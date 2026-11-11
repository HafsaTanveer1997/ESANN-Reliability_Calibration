"""Fit and apply the calibrators used by the Stage 2 Optuna studies."""

from __future__ import annotations

from typing import Any

import torch

from .classwise_affine import apply_classwise_affine, fit_classwise_affine
from .temperature_scaling import apply_temperature, fit_temperature


def fit_and_apply_calibration(
    fit_logits: torch.Tensor,
    fit_labels: torch.Tensor,
    evaluation_logits: torch.Tensor,
    model_params: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    """Fit one calibration method and apply it to evaluation logits."""

    method = str(model_params["calibration_method"])

    if method == "temperature":
        temperature = fit_temperature(
            logits=fit_logits,
            labels=fit_labels,
            minimum_temperature=float(
                model_params["minimum_temperature"]
            ),
            maximum_temperature=float(
                model_params["maximum_temperature"]
            ),
            initial_temperature=float(
                model_params["initial_temperature"]
            ),
            optimizer_name=str(
                model_params["optimizer_name"]
            ),
            learning_rate=float(
                model_params["learning_rate"]
            ),
            max_iterations=int(
                model_params["max_iterations"]
            ),
        )

        calibrated_logits = apply_temperature(
            evaluation_logits,
            temperature,
        )
        learned_parameters = {
            "temperature": float(temperature),
        }
        method_results = {
            "temperature": float(temperature),
        }

    elif method == "classwise_affine":
        gamma, beta, best_training_loss = (
            fit_classwise_affine(
                logits=fit_logits,
                labels=fit_labels,
                learning_rate=float(
                    model_params["learning_rate"]
                ),
                steps=int(
                    model_params["max_iterations"]
                ),
                nll_weight=float(
                    model_params["fit_nll_weight"]
                ),
                brier_weight=float(
                    model_params["fit_brier_weight"]
                ),
                gamma_regularization=float(
                    model_params[
                        "gamma_regularization"
                    ]
                ),
                beta_regularization=float(
                    model_params[
                        "beta_regularization"
                    ]
                ),
                print_interval=0,
            )
        )

        calibrated_logits = apply_classwise_affine(
            evaluation_logits,
            gamma,
            beta,
        )
        learned_parameters = {
            "gamma": gamma.detach().cpu().tolist(),
            "beta": beta.detach().cpu().tolist(),
        }
        method_results = {
            "best_training_loss": float(
                best_training_loss
            ),
            "gamma_min": float(gamma.min().item()),
            "gamma_max": float(gamma.max().item()),
            "gamma_mean": float(gamma.mean().item()),
            "beta_min": float(beta.min().item()),
            "beta_max": float(beta.max().item()),
            "beta_mean": float(beta.mean().item()),
        }

    else:
        raise ValueError(f"Unsupported Stage 2 calibration method: {method}")

    return calibrated_logits, learned_parameters, method_results
