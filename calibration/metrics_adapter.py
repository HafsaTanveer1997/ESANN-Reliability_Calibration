"""Use the existing pipeline metrics for calibration comparisons."""

import numpy as np
import torch

from .reliability_metrics import (
    evaluate_logits,
    compute_reliability_risk,
)


def evaluate(logits, labels, tau=0.6):
    values = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    targets = torch.as_tensor(np.asarray(labels), dtype=torch.long)

    if values.ndim != 2 or targets.shape != (values.shape[0],):
        raise ValueError("Expected logits [N, C] and labels [N].")
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite logits.")

    metrics = evaluate_logits(
        values, targets, num_ece_bins=15, confidence_threshold=tau,
    )

    def risk(weight, hce_weight):
        return compute_reliability_risk(
            metrics,
            num_classes=values.shape[1],
            nll_weight=weight,
            brier_weight=weight,
            ece_weight=weight,
            high_confidence_error_weight=hce_weight,
        )["reliability_risk"]

    return {
        "top1": metrics["top1"],
        "nll": metrics["nll"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
        "hce": metrics["high_confidence_error"],
        "r3": risk(1/3, 0.0),
        "r4": risk(0.25, 0.25),
        "coverage_at_tau": (
            values.softmax(1).max(1).values.ge(tau).float().mean().item()
        ),
    }
