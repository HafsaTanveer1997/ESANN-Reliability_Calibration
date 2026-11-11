"""Reusable class-wise affine calibration for classification logits."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _validate_inputs(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    """Validate the common [N, C] logits and [N] labels interface."""

    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor.")

    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor.")

    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [N, C], but got {tuple(logits.shape)}."
        )

    if labels.ndim != 1:
        raise ValueError(
            f"labels must have shape [N], but got {tuple(labels.shape)}."
        )

    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "logits and labels must contain the same number of samples."
        )

    if logits.shape[0] == 0:
        raise ValueError("logits and labels cannot be empty.")

    if logits.shape[1] < 2:
        raise ValueError("logits must contain at least two classes.")

    if not torch.is_floating_point(logits):
        raise TypeError("logits must use a floating-point dtype.")

    if not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinite values.")

    if labels.min().item() < 0:
        raise ValueError("labels cannot contain negative class indices.")

    if labels.max().item() >= logits.shape[1]:
        raise ValueError(
            "A label index is greater than or equal to the number of classes."
        )


def _brier_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Calculate the multiclass Brier loss."""

    probabilities = torch.softmax(logits, dim=1)

    targets = F.one_hot(
        labels,
        num_classes=logits.shape[1],
    ).to(
        device=logits.device,
        dtype=logits.dtype,
    )

    return torch.mean(
        torch.sum(
            (probabilities - targets) ** 2,
            dim=1,
        )
    )


def apply_classwise_affine(
    logits: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Apply class-wise affine calibration.

    The transformation is:

        corrected_logits[:, c] =
            gamma[c] * logits[:, c] + beta[c]

    Parameters
    ----------
    logits:
        Classification logits with shape [N, C].
    gamma:
        Positive class-wise scaling values with shape [C].
    beta:
        Class-wise offsets with shape [C].
    """

    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor.")

    if not isinstance(gamma, torch.Tensor):
        raise TypeError("gamma must be a torch.Tensor.")

    if not isinstance(beta, torch.Tensor):
        raise TypeError("beta must be a torch.Tensor.")

    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, C].")

    if gamma.ndim != 1:
        raise ValueError("gamma must have shape [C].")

    if beta.ndim != 1:
        raise ValueError("beta must have shape [C].")

    num_classes = logits.shape[1]

    if gamma.shape[0] != num_classes:
        raise ValueError(
            "gamma length must equal the number of logit classes."
        )

    if beta.shape[0] != num_classes:
        raise ValueError(
            "beta length must equal the number of logit classes."
        )

    gamma = gamma.to(
        device=logits.device,
        dtype=logits.dtype,
    )

    beta = beta.to(
        device=logits.device,
        dtype=logits.dtype,
    )

    if not torch.isfinite(gamma).all():
        raise ValueError("gamma contains NaN or infinite values.")

    if not torch.isfinite(beta).all():
        raise ValueError("beta contains NaN or infinite values.")

    if torch.any(gamma <= 0):
        raise ValueError("All gamma values must be positive.")

    return logits * gamma.unsqueeze(0) + beta.unsqueeze(0)


def fit_classwise_affine(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    learning_rate: float = 0.01,
    steps: int = 1500,
    nll_weight: float = 1.0,
    brier_weight: float = 0.5,
    gamma_regularization: float = 0.01,
    beta_regularization: float = 0.01,
    print_interval: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Fit positive class-wise scales and class-wise offsets.

    The objective combines:

    - negative log-likelihood;
    - Brier loss;
    - regularization of gamma around one;
    - regularization of beta around zero.

    Returns
    -------
    gamma:
        Learned positive scaling tensor with shape [C].
    beta:
        Learned offset tensor with shape [C].
    best_loss:
        Lowest optimization-objective value observed during fitting.
    """

    _validate_inputs(logits, labels)

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if steps <= 0:
        raise ValueError("steps must be positive.")

    if nll_weight < 0:
        raise ValueError("nll_weight cannot be negative.")

    if brier_weight < 0:
        raise ValueError("brier_weight cannot be negative.")

    if gamma_regularization < 0:
        raise ValueError("gamma_regularization cannot be negative.")

    if beta_regularization < 0:
        raise ValueError("beta_regularization cannot be negative.")

    if nll_weight == 0 and brier_weight == 0:
        raise ValueError(
            "At least one of nll_weight or brier_weight must be positive."
        )

    if print_interval < 0:
        raise ValueError("print_interval cannot be negative.")

    fit_logits = logits.detach()

    fit_labels = labels.detach().to(
        device=fit_logits.device,
        dtype=torch.long,
    )

    num_classes = fit_logits.shape[1]

    log_gamma = torch.nn.Parameter(
        torch.zeros(
            num_classes,
            dtype=fit_logits.dtype,
            device=fit_logits.device,
        )
    )

    beta = torch.nn.Parameter(
        torch.zeros(
            num_classes,
            dtype=fit_logits.dtype,
            device=fit_logits.device,
        )
    )

    optimizer = torch.optim.Adam(
        [log_gamma, beta],
        lr=learning_rate,
    )

    best_loss = float("inf")
    best_log_gamma = None
    best_beta = None

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        gamma = torch.exp(log_gamma)

        transformed_logits = (
            fit_logits * gamma.unsqueeze(0)
            + beta.unsqueeze(0)
        )

        nll = F.cross_entropy(
            transformed_logits,
            fit_labels,
        )

        brier = _brier_loss(
            transformed_logits,
            fit_labels,
        )

        gamma_penalty = torch.mean(
            (gamma - 1.0) ** 2
        )

        beta_penalty = torch.mean(
            beta ** 2
        )

        loss = (
            nll_weight * nll
            + brier_weight * brier
            + gamma_regularization * gamma_penalty
            + beta_regularization * beta_penalty
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite optimization loss encountered at step {step}."
            )

        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().item())

        if loss_value < best_loss:
            best_loss = loss_value
            best_log_gamma = log_gamma.detach().clone()
            best_beta = beta.detach().clone()

        if print_interval > 0 and (
            step == 1
            or step % print_interval == 0
            or step == steps
        ):
            print(
                f"Step {step:4d}/{steps}: "
                f"loss={loss_value:.8f}, "
                f"nll={nll.item():.8f}, "
                f"brier={brier.item():.8f}"
            )

    if best_log_gamma is None or best_beta is None:
        raise RuntimeError(
            "No valid class-wise affine parameters were produced."
        )

    best_gamma = torch.exp(best_log_gamma)

    return best_gamma, best_beta, best_loss
