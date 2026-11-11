"""Read and validate saved experiment partitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple


@dataclass(frozen=True)
class ExperimentDataSplit:
    """Indices for all training and calibration stages."""

    training_indices: Tuple[int, ...]
    stage1_validation_indices: Tuple[int, ...]
    ptq_calibration_indices: Tuple[int, ...]
    stage2_fit_indices: Tuple[int, ...]
    stage2_validation_indices: Tuple[int, ...]
    unused_indices: Tuple[int, ...]


def load_experiment_data_split(
    input_path: str | Path,
) -> ExperimentDataSplit:
    """Load and validate a saved experiment split."""

    input_path = Path(
        input_path
    ).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Split file does not exist: {input_path}"
        )

    payload = json.loads(
        input_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Split file must contain a JSON object."
        )

    splits = payload.get("splits")

    if not isinstance(splits, dict):
        raise ValueError(
            "Split file has no valid 'splits' object."
        )

    required_names = {
        "training",
        "stage1_validation",
        "ptq_calibration",
        "stage2_fit",
        "stage2_validation",
        "unused",
    }

    missing_names = required_names.difference(
        splits
    )

    if missing_names:
        raise ValueError(
            "Split file is missing partitions: "
            f"{sorted(missing_names)}"
        )

    parsed = {}

    for name in required_names:
        values = splits[name]

        if not isinstance(values, list):
            raise ValueError(
                f"Partition '{name}' must be a list."
            )

        parsed[name] = tuple(
            int(value)
            for value in values
        )

        if any(
            value < 0
            for value in parsed[name]
        ):
            raise ValueError(
                f"Partition '{name}' contains negative indices."
            )

    combined = []

    for name in required_names:
        combined.extend(parsed[name])

    if len(combined) != len(set(combined)):
        raise ValueError(
            "Saved experiment partitions overlap."
        )

    return ExperimentDataSplit(
        training_indices=parsed["training"],
        stage1_validation_indices=(
            parsed["stage1_validation"]
        ),
        ptq_calibration_indices=(
            parsed["ptq_calibration"]
        ),
        stage2_fit_indices=parsed["stage2_fit"],
        stage2_validation_indices=(
            parsed["stage2_validation"]
        ),
        unused_indices=parsed["unused"],
    )
