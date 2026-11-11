"""Reusable classification accuracy and reliability metrics."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def _validate_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Validate logits and return labels aligned to their device."""

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

    aligned_labels = labels.to(
        device=logits.device,
        dtype=torch.long,
    )

    if aligned_labels.min().item() < 0:
        raise ValueError("labels cannot contain negative class indices.")

    if aligned_labels.max().item() >= logits.shape[1]:
        raise ValueError(
            "A label index is greater than or equal to the number of classes."
        )

    return aligned_labels


def top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 1,
) -> float:
    """Compute classification accuracy within the top-k predictions."""

    labels = _validate_logits_and_labels(logits, labels)

    if k <= 0:
        raise ValueError("k must be positive.")

    effective_k = min(k, logits.shape[1])

    predictions = logits.topk(
        effective_k,
        dim=1,
    ).indices

    correct = (
        predictions.eq(labels.unsqueeze(1))
        .any(dim=1)
        .float()
        .mean()
    )

    return float(correct.item())


def negative_log_likelihood(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute mean multiclass negative log-likelihood."""

    labels = _validate_logits_and_labels(logits, labels)

    nll = F.cross_entropy(
        logits,
        labels,
        reduction="mean",
    )

    return float(nll.item())


def multiclass_brier_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute the multiclass Brier score."""

    labels = _validate_logits_and_labels(logits, labels)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    targets = F.one_hot(
        labels,
        num_classes=logits.shape[1],
    ).to(
        device=logits.device,
        dtype=logits.dtype,
    )

    score = torch.mean(
        torch.sum(
            (probabilities - targets) ** 2,
            dim=1,
        )
    )

    return float(score.item())


def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
) -> float:
    """Compute fixed-width Expected Calibration Error."""

    labels = _validate_logits_and_labels(logits, labels)

    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    confidences, predictions = probabilities.max(
        dim=1,
    )

    correctness = predictions.eq(
        labels
    ).float()

    bin_boundaries = torch.linspace(
        0.0,
        1.0,
        steps=num_bins + 1,
        device=logits.device,
        dtype=logits.dtype,
    )

    ece = torch.zeros(
        (),
        device=logits.device,
        dtype=logits.dtype,
    )

    for bin_index in range(num_bins):
        lower = bin_boundaries[bin_index]
        upper = bin_boundaries[bin_index + 1]

        if bin_index == 0:
            in_bin = (
                confidences.ge(lower)
                & confidences.le(upper)
            )
        else:
            in_bin = (
                confidences.gt(lower)
                & confidences.le(upper)
            )

        bin_fraction = in_bin.float().mean()

        if in_bin.any():
            bin_accuracy = correctness[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()

            ece = ece + (
                torch.abs(
                    bin_confidence - bin_accuracy
                )
                * bin_fraction
            )

    return float(ece.item())


def evaluate_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_ece_bins: int = 15,
    confidence_threshold: float = 0.60,
) -> Dict[str, float | int]:
    """Evaluate prediction and probability-quality metrics."""

    labels = _validate_logits_and_labels(
        logits,
        labels,
    )

    return {
        "samples": int(labels.numel()),
        "top1": top_k_accuracy(
            logits,
            labels,
            k=1,
        ),
        "top5": top_k_accuracy(
            logits,
            labels,
            k=5,
        ),
        "nll": negative_log_likelihood(
            logits,
            labels,
        ),
        "brier": multiclass_brier_score(
            logits,
            labels,
        ),
        "ece": expected_calibration_error(
            logits,
            labels,
            num_bins=num_ece_bins,
        ),
        "high_confidence_error": high_confidence_error_rate(
            logits,
            labels,
            confidence_threshold=confidence_threshold,
        ),
    }


def high_confidence_error_rate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    confidence_threshold: float = 0.60,
) -> float:
    """Return the fraction of all samples that are confidently misclassified."""

    labels = _validate_logits_and_labels(
        logits,
        labels,
    )

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            "confidence_threshold must lie between 0 and 1."
        )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    confidences, predictions = probabilities.max(
        dim=1,
    )

    confident_errors = (
        predictions.ne(labels)
        & confidences.ge(confidence_threshold)
    )

    return float(
        confident_errors.float().mean().item()
    )


def accuracy_guard_passed(
    baseline_top1: float,
    calibrated_top1: float,
    maximum_allowed_drop: float = 0.005,
) -> bool:
    """Check the common accuracy constraint for any calibration method."""

    if not 0.0 <= baseline_top1 <= 1.0:
        raise ValueError(
            "baseline_top1 must lie between 0 and 1."
        )

    if not 0.0 <= calibrated_top1 <= 1.0:
        raise ValueError(
            "calibrated_top1 must lie between 0 and 1."
        )

    if maximum_allowed_drop < 0.0:
        raise ValueError(
            "maximum_allowed_drop cannot be negative."
        )

    minimum_accepted_top1 = (
        baseline_top1 - maximum_allowed_drop
    )

    return calibrated_top1 >= minimum_accepted_top1



def compute_reliability_risk(
    metrics: Dict[str, float | int],
    num_classes: int,
    *,
    nll_weight: float = 0.25,
    brier_weight: float = 0.25,
    ece_weight: float = 0.25,
    high_confidence_error_weight: float = 0.25,
) -> Dict[str, float]:
    """Return separate normalized metrics and their combined risk score."""

    if num_classes < 2:
        raise ValueError(
            "num_classes must be at least two."
        )

    weights = {
        "nll": float(nll_weight),
        "brier": float(brier_weight),
        "ece": float(ece_weight),
        "high_confidence_error": float(
            high_confidence_error_weight
        ),
    }

    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError(
            "Reliability-risk weights cannot be negative."
        )

    if not math.isclose(
        sum(weights.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Reliability-risk weights must sum to 1.0."
        )

    required_metrics = {
        "nll",
        "brier",
        "ece",
        "high_confidence_error",
    }

    missing_metrics = required_metrics.difference(metrics)

    if missing_metrics:
        raise KeyError(
            "Missing reliability metrics: "
            f"{sorted(missing_metrics)}"
        )

    nll = float(metrics["nll"])
    brier = float(metrics["brier"])
    ece = float(metrics["ece"])
    high_confidence_error = float(
        metrics["high_confidence_error"]
    )

    values = {
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "high_confidence_error": high_confidence_error,
    }

    if any(
        not math.isfinite(value)
        for value in values.values()
    ):
        raise ValueError(
            "Reliability metrics must be finite."
        )

    if nll < 0.0:
        raise ValueError("NLL cannot be negative.")

    if not 0.0 <= brier <= 2.0 + 1e-6:
        raise ValueError(
            "Brier score must lie between 0 and 2."
        )

    if not 0.0 <= ece <= 1.0:
        raise ValueError(
            "ECE must lie between 0 and 1."
        )

    if not 0.0 <= high_confidence_error <= 1.0:
        raise ValueError(
            "High-confidence error must lie between 0 and 1."
        )

    normalized_nll = nll / math.log(num_classes)
    normalized_brier = brier / 2.0

    weighted_nll = weights["nll"] * normalized_nll
    weighted_brier = (
        weights["brier"] * normalized_brier
    )
    weighted_ece = weights["ece"] * ece
    weighted_hce = (
        weights["high_confidence_error"]
        * high_confidence_error
    )

    reliability_risk = (
        weighted_nll
        + weighted_brier
        + weighted_ece
        + weighted_hce
    )

    return {
        "raw_nll": nll,
        "raw_brier": brier,
        "raw_ece": ece,
        "raw_high_confidence_error": high_confidence_error,
        "normalized_nll": normalized_nll,
        "normalized_brier": normalized_brier,
        "normalized_ece": ece,
        "normalized_high_confidence_error": (
            high_confidence_error
        ),
        "weighted_nll": weighted_nll,
        "weighted_brier": weighted_brier,
        "weighted_ece": weighted_ece,
        "weighted_high_confidence_error": weighted_hce,
        "reliability_risk": reliability_risk,
    }
