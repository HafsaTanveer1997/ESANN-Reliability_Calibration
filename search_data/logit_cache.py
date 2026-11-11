"""Save and load cached classification logits for Stage-2 calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def save_logit_cache(
    output_path: str | Path,
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    """Save logits, labels, and optional metadata to a PyTorch file."""

    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [N, C], got {tuple(logits.shape)}."
        )

    if labels.ndim != 1:
        raise ValueError(
            f"labels must have shape [N], got {tuple(labels.shape)}."
        )

    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "logits and labels must contain the same number of samples."
        )

    if logits.shape[0] == 0:
        raise ValueError(
            "logits and labels cannot be empty."
        )

    if not torch.is_floating_point(logits):
        raise TypeError(
            "logits must use a floating-point dtype."
        )

    if not torch.isfinite(logits).all():
        raise ValueError(
            "logits contain NaN or infinite values."
        )

    labels = labels.to(
        dtype=torch.long,
        device="cpu",
    )

    logits = logits.detach().to(
        dtype=torch.float32,
        device="cpu",
    )

    if labels.min().item() < 0:
        raise ValueError(
            "labels cannot contain negative values."
        )

    if labels.max().item() >= logits.shape[1]:
        raise ValueError(
            "A label index exceeds the number of classes."
        )

    if metadata is None:
        metadata = {}

    if not isinstance(metadata, dict):
        raise TypeError(
            "metadata must be a dictionary."
        )

    output_path = Path(
        output_path
    ).expanduser().resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "format_version": 1,
        "logits": logits,
        "labels": labels,
        "metadata": metadata,
    }

    torch.save(
        payload,
        output_path,
    )

    return output_path


def load_logit_cache(
    input_path: str | Path,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Load and validate a cached logit dataset."""

    input_path = Path(
        input_path
    ).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Logit cache does not exist: {input_path}"
        )

    payload = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Logit cache must contain a dictionary."
        )

    required_keys = {
        "logits",
        "labels",
        "metadata",
    }

    missing_keys = required_keys.difference(payload)

    if missing_keys:
        raise ValueError(
            f"Logit cache is missing keys: {sorted(missing_keys)}"
        )

    logits = payload["logits"]
    labels = payload["labels"]
    metadata = payload["metadata"]

    if not isinstance(logits, torch.Tensor):
        raise TypeError(
            "Cached logits must be a torch.Tensor."
        )

    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            "Cached labels must be a torch.Tensor."
        )

    if not isinstance(metadata, dict):
        raise TypeError(
            "Cached metadata must be a dictionary."
        )

    if logits.ndim != 2:
        raise ValueError(
            "Cached logits must have shape [N, C]."
        )

    if labels.ndim != 1:
        raise ValueError(
            "Cached labels must have shape [N]."
        )

    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "Cached logits and labels have different sample counts."
        )

    return (
        logits.to(dtype=torch.float32),
        labels.to(dtype=torch.long),
        metadata,
    )
