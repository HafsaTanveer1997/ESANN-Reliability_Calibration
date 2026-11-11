"""Temperature-anchored sparse affine calibration for development studies."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def _inputs(logits, labels):
    z = torch.as_tensor(logits).detach().to(device="cpu", dtype=torch.float64)
    y = torch.as_tensor(labels).detach().to(device="cpu", dtype=torch.long)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2:
        raise ValueError("Expected nonempty logits [N, C], C >= 2.")
    if y.shape != (len(z),) or not torch.isfinite(z).all():
        raise ValueError("Invalid labels or nonfinite logits.")
    if y.min() < 0 or y.max() >= z.shape[1]:
        raise ValueError("Labels outside the class range.")
    return z, y


def _optimize(parameters, objective, max_iter, learning_rate=1.0):
    if max_iter <= 0 or learning_rate <= 0:
        raise ValueError("Optimizer budget and learning rate must be positive.")
    optimizer = torch.optim.LBFGS(
        parameters, lr=learning_rate, max_iter=max_iter,
        history_size=20, line_search_fn="strong_wolfe",
        tolerance_grad=1e-9, tolerance_change=1e-12,
    )

    def checked_loss():
        loss = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite fitting objective.")
        return loss

    initial = float(checked_loss().detach())

    def closure():
        optimizer.zero_grad()
        loss = checked_loss()
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in parameters
        ):
            raise RuntimeError("Nonfinite fitting gradient.")
        return loss

    optimizer.step(closure)
    final = closure()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise RuntimeError("Nonfinite fitted parameter.")

    gradients = torch.cat([
        p.grad.reshape(-1) for p in parameters if p.grad is not None
    ])
    state = optimizer.state[parameters[0]]
    iterations = int(state.get("n_iter", 0))
    return {
        "objective_initial": initial,
        "objective_final": float(final.detach()),
        "gradient_norm": float(gradients.norm()),
        "gradient_max_abs": float(gradients.abs().max()),
        "iterations": iterations,
        "function_evaluations": int(state.get("func_evals", 0)),
        "hit_iteration_limit": iterations >= max_iter,
        "gradient_tolerance_met": float(gradients.abs().max()) <= 1e-9,
        "finite": True,
    }


def fit_temperature(logits, labels, max_iter=200):
    z, y = _inputs(logits, labels)
    log_t = torch.zeros(1, dtype=z.dtype, requires_grad=True)

    def objective():
        return F.cross_entropy(z / log_t.exp(), y)

    info = _optimize([log_t], objective, max_iter, learning_rate=0.5)
    temperature = float(log_t.detach().exp())
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("Invalid fitted temperature.")
    info["temperature"] = temperature
    return temperature, info


@torch.no_grad()
def select_links(logits, labels, k=2, min_support=1.0, mode="hard"):
    classes = logits.shape[1]
    if not 0 < k < classes or min_support <= 0:
        raise ValueError("Invalid link count or support threshold.")
    support = torch.zeros(classes, classes, dtype=logits.dtype)
    if mode == "hard":
        support.index_put_(
            (labels, logits.argmax(1)),
            torch.ones(len(labels), dtype=logits.dtype),
            accumulate=True,
        )
    elif mode == "soft":
        support.index_add_(0, labels, logits.softmax(1))
    else:
        raise ValueError("Link mode must be hard or soft.")
    support.fill_diagonal_(0)
    mask = torch.zeros(classes, classes, dtype=torch.bool)
    for true_class in range(classes):
        ranked = sorted(
            (j for j in range(classes) if j != true_class),
            key=lambda j: (-float(support[true_class, j]), j),
        )
        for predicted_class in ranked[:k]:
            if support[true_class, predicted_class] >= min_support:
                mask[true_class, predicted_class] = True

    selected_support = support[mask]
    info = {
        "links": int(mask.sum()),
        "classes_without_links": int((mask.sum(1) == 0).sum()),
        "max_links_per_row": int(mask.sum(1).max()),
        "support_min": float(selected_support.min()) if mask.any() else 0.0,
        "support_median": float(selected_support.median()) if mask.any() else 0.0,
        "support_max": float(selected_support.max()) if mask.any() else 0.0,
    }
    return mask, info


@dataclass
class TACGSRH:
    temperature: float
    matrix: torch.Tensor
    bias: torch.Tensor
    mask: torch.Tensor
    info: dict

    @torch.no_grad()
    def apply(self, logits):
        z = torch.as_tensor(logits).to(device="cpu", dtype=self.matrix.dtype)
        result = z @ self.matrix.T + self.bias
        if not torch.isfinite(result).all():
            raise RuntimeError("Nonfinite calibrated logits.")
        return result

    @torch.no_grad()
    def fold(self, weight, bias=None):
        matrix = self.matrix.to(device=weight.device, dtype=weight.dtype)
        shift = self.bias.to(device=weight.device, dtype=weight.dtype)
        if bias is None:
            bias = weight.new_zeros(weight.shape[0])
        return matrix @ weight, matrix @ bias + shift


def fit_ta_cgsrh(
    logits, labels, *, alpha=0.0, tau=0.6, k=2,
    min_support=1.0, link_mode="hard", hinge="hard", beta=50.0,
    lam_g=0.01, lam_r=0.01, lam_b=0.001,
    anchor=True, max_iter=2000,
    learn_scales=True, learn_bias=True, learn_residual=True,
):
    z, y = _inputs(logits, labels)
    if alpha < 0 or min(lam_g, lam_r, lam_b) < 0:
        raise ValueError("Loss weights must be nonnegative.")
    if not 0 <= tau <= 1 or beta <= 0:
        raise ValueError("Invalid hinge threshold or smoothing parameter.")
    if hinge not in ("hard", "smooth"):
        raise ValueError("Hinge must be hard or smooth.")

    temperature, temperature_info = (
        fit_temperature(z, y) if anchor else (1.0, {})
    )
    scaled = z / temperature
    mask, link_info = select_links(
        scaled, y, k=k, min_support=min_support, mode=link_mode,
    )
    row, column = mask.nonzero(as_tuple=True)
    classes = z.shape[1]
    g = torch.zeros(classes, dtype=z.dtype, requires_grad=learn_scales)
    residual = torch.zeros(len(row), dtype=z.dtype, requires_grad=learn_residual)
    bias = torch.zeros(classes, dtype=z.dtype, requires_grad=learn_bias)
    targets = F.one_hot(y, classes).to(z.dtype)

    def transformed():
        result = scaled * g.exp()
        if len(row):
            result = result.index_add(
                1, row, scaled[:, column] * residual,
            )
        return result + bias

    def components():
        result = transformed()
        probabilities = result.softmax(1)
        nll = F.cross_entropy(result, y)
        brier = (probabilities - targets).square().sum(1).mean()
        wrong = probabilities.masked_fill(targets.bool(), -1).max(1).values
        excess = wrong - tau
        hce_loss = (
            excess.clamp_min(0).mean() if hinge == "hard"
            else F.softplus(beta * excess).mean() / beta
        )
        regularization = lam_g*g.square().mean() + lam_b*bias.square().mean()
        if len(row):
            regularization = regularization + lam_r*residual.square().mean()
        return nll, brier, hce_loss, regularization

    def objective():
        nll, brier, hce_loss, regularization = components()
        return 0.5*nll + 0.5*brier + alpha*hce_loss + regularization

    parameters = []
    if learn_scales:
        parameters.append(g)
    if learn_bias:
        parameters.append(bias)
    if learn_residual and len(row):
        parameters.append(residual)
    if not parameters:
        raise ValueError('Use plain temperature scaling when no corrections are learnable.')
    fit_info = _optimize(parameters, objective, max_iter)

    with torch.no_grad():
        matrix = torch.diag(g.exp())
        matrix[row, column] = residual
        matrix = matrix / temperature
        nll, brier, hce_loss, regularization = components()
        info = {
            **link_info, **fit_info,
            **{f"temperature_{key}": value
               for key, value in temperature_info.items()},
            "anchor": bool(anchor),
            "parameter_count": sum(p.numel() for p in parameters) + int(anchor),
            "learn_scales": bool(learn_scales),
            "learn_bias": bool(learn_bias),
            "learn_residual": bool(learn_residual),
            "effective_scale_min": float(g.exp().min()/temperature),
            "effective_scale_max": float(g.exp().max()/temperature),
            "residual_abs_max": float(residual.abs().max()) if len(row) else 0.0,
            "bias_abs_max": float(bias.abs().max()),
            "fitting_nll": float(nll),
            "fitting_brier": float(brier),
            "fitting_hce_surrogate": float(hce_loss),
            "regularization": float(regularization),
        }
        model = TACGSRH(
            temperature, matrix.detach().clone(),
            bias.detach().clone(), mask, info,
        )
        assert torch.allclose(model.apply(z), transformed(), atol=1e-10, rtol=1e-10)
    return model
