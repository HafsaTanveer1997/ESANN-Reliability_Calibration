"""Temperature scaling for classification logits."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F


OptimizerName = Literal["lbfgs", "adam"]


def apply_temperature(
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """Scale logits by a positive temperature."""
    temperature = torch.as_tensor(
        temperature,
        dtype=logits.dtype,
        device=logits.device,
    )
    return logits / temperature


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    minimum_temperature: float = 0.05,
    maximum_temperature: float = 20.0,
    initial_temperature: float = 1.0,
    optimizer_name: OptimizerName = "lbfgs",
    learning_rate: float = 0.1,
    max_iterations: int = 100,
) -> float:
    """Fit one temperature by minimizing cross-entropy."""

    if optimizer_name not in {"lbfgs", "adam"}:
        raise ValueError("optimizer_name must be 'lbfgs' or 'adam'.")

    fit_logits = logits.detach()
    fit_labels = labels.detach().to(
        device=fit_logits.device,
        dtype=torch.long,
    )

    log_temperature = torch.nn.Parameter(
        torch.tensor(
            math.log(initial_temperature),
            dtype=fit_logits.dtype,
            device=fit_logits.device,
        )
    )

    minimum_log_temperature = math.log(minimum_temperature)
    maximum_log_temperature = math.log(maximum_temperature)

    def temperature() -> torch.Tensor:
        return log_temperature.clamp(
            minimum_log_temperature,
            maximum_log_temperature,
        ).exp()

    def loss() -> torch.Tensor:
        return F.cross_entropy(
            apply_temperature(fit_logits, temperature()),
            fit_labels,
        )

    if optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [log_temperature],
            lr=learning_rate,
            max_iter=max_iterations,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            objective = loss()
            objective.backward()
            return objective

        optimizer.step(closure)

    else:
        optimizer = torch.optim.Adam(
            [log_temperature],
            lr=learning_rate,
        )

        for _ in range(max_iterations):
            optimizer.zero_grad()
            objective = loss()
            objective.backward()
            optimizer.step()

            with torch.no_grad():
                log_temperature.clamp_(
                    minimum_log_temperature,
                    maximum_log_temperature,
                )

    return temperature().detach().item()
