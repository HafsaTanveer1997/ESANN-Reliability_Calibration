"""Utilities for creating reproducible calibration-data splits."""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import torch


@dataclass(frozen=True)
class CalibrationDataSplit:
    """Non-overlapping indices used during Stage-2 calibration."""

    fit_indices: Tuple[int, ...]
    validation_indices: Tuple[int, ...]
    unused_indices: Tuple[int, ...]


def create_calibration_data_split(
    available_indices: Sequence[int],
    fit_size: int,
    validation_size: int,
    *,
    seed: int = 42,
) -> CalibrationDataSplit:
    """Create deterministic calibration-fit and validation splits.

    The supplied indices must come from a dataset partition reserved for
    calibration. The final test-set indices should not be supplied here.
    """

    indices = [int(index) for index in available_indices]

    if not indices:
        raise ValueError(
            "available_indices cannot be empty."
        )

    if len(indices) != len(set(indices)):
        raise ValueError(
            "available_indices contains duplicate values."
        )

    if any(index < 0 for index in indices):
        raise ValueError(
            "available_indices cannot contain negative values."
        )

    if fit_size <= 0:
        raise ValueError(
            "fit_size must be greater than zero."
        )

    if validation_size <= 0:
        raise ValueError(
            "validation_size must be greater than zero."
        )

    requested_size = fit_size + validation_size

    if requested_size > len(indices):
        raise ValueError(
            "fit_size plus validation_size exceeds the number "
            "of available indices."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    permutation = torch.randperm(
        len(indices),
        generator=generator,
    ).tolist()

    shuffled_indices = [
        indices[position]
        for position in permutation
    ]

    fit_end = fit_size
    validation_end = fit_size + validation_size

    fit_indices = tuple(
        shuffled_indices[:fit_end]
    )

    validation_indices = tuple(
        shuffled_indices[
            fit_end:validation_end
        ]
    )

    unused_indices = tuple(
        shuffled_indices[validation_end:]
    )

    return CalibrationDataSplit(
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        unused_indices=unused_indices,
    )


def create_stratified_fit_validation_split(
    labels: Sequence[int] | torch.Tensor,
    validation_size: int,
    *,
    seed: int = 42,
) -> CalibrationDataSplit:
    """Split labeled samples into deterministic stratified fit/validation sets.

    Returned indices are positions within ``labels``. All samples not assigned
    to validation are assigned to fit, so no samples are discarded.
    """

    labels_tensor = torch.as_tensor(
        labels,
        dtype=torch.long,
    ).detach().cpu().reshape(-1)

    num_samples = int(labels_tensor.numel())

    if num_samples == 0:
        raise ValueError("labels cannot be empty.")

    if validation_size <= 0:
        raise ValueError(
            "validation_size must be greater than zero."
        )

    if validation_size >= num_samples:
        raise ValueError(
            "validation_size must be smaller than the number of samples."
        )

    classes, class_counts = torch.unique(
        labels_tensor,
        sorted=True,
        return_counts=True,
    )

    # Allocate validation samples proportionally. Largest fractional
    # remainders receive any samples left after flooring so the requested
    # validation size is matched exactly.
    exact_counts = (
        class_counts.to(dtype=torch.float64)
        * float(validation_size)
        / float(num_samples)
    )
    validation_counts = torch.floor(
        exact_counts
    ).to(dtype=torch.long)

    remaining = (
        int(validation_size)
        - int(validation_counts.sum().item())
    )

    if remaining > 0:
        fractional = exact_counts - validation_counts
        remainder_order = sorted(
            range(len(classes)),
            key=lambda index: (
                -float(fractional[index].item()),
                int(classes[index].item()),
            ),
        )

        for index in remainder_order[:remaining]:
            validation_counts[index] += 1

    if int(validation_counts.sum().item()) != int(
        validation_size
    ):
        raise RuntimeError(
            "Failed to construct the requested stratified validation size."
        )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    fit_positions: list[int] = []
    validation_positions: list[int] = []

    for class_value, class_count, validation_count in zip(
        classes,
        class_counts,
        validation_counts,
    ):
        class_positions = torch.nonzero(
            labels_tensor == class_value,
            as_tuple=False,
        ).reshape(-1)

        permutation = torch.randperm(
            int(class_count.item()),
            generator=generator,
        )
        class_positions = class_positions[permutation]

        split_point = int(validation_count.item())

        validation_positions.extend(
            class_positions[:split_point].tolist()
        )
        fit_positions.extend(
            class_positions[split_point:].tolist()
        )

    # Remove class-group ordering while preserving deterministic membership.
    fit_permutation = torch.randperm(
        len(fit_positions),
        generator=generator,
    ).tolist()
    validation_permutation = torch.randperm(
        len(validation_positions),
        generator=generator,
    ).tolist()

    fit_positions = [
        fit_positions[index]
        for index in fit_permutation
    ]
    validation_positions = [
        validation_positions[index]
        for index in validation_permutation
    ]

    if len(fit_positions) + len(validation_positions) != num_samples:
        raise RuntimeError(
            "Stratified split did not preserve every input sample."
        )

    if set(fit_positions) & set(validation_positions):
        raise RuntimeError(
            "Stratified fit and validation partitions overlap."
        )

    return CalibrationDataSplit(
        fit_indices=tuple(fit_positions),
        validation_indices=tuple(validation_positions),
        unused_indices=(),
    )


@dataclass(frozen=True)
class ExperimentDataSplit:
    """Indices for all training and calibration stages."""

    training_indices: Tuple[int, ...]
    stage1_validation_indices: Tuple[int, ...]
    ptq_calibration_indices: Tuple[int, ...]
    stage2_fit_indices: Tuple[int, ...]
    stage2_validation_indices: Tuple[int, ...]
    unused_indices: Tuple[int, ...]


def create_experiment_data_split(
    available_indices: Sequence[int],
    training_size: int,
    stage1_validation_size: int,
    ptq_calibration_size: int,
    stage2_fit_size: int,
    stage2_validation_size: int,
    *,
    seed: int = 42,
) -> ExperimentDataSplit:
    """Create deterministic, mutually exclusive experiment partitions."""

    indices = [int(index) for index in available_indices]

    if not indices:
        raise ValueError(
            "available_indices cannot be empty."
        )

    if len(indices) != len(set(indices)):
        raise ValueError(
            "available_indices contains duplicate values."
        )

    if any(index < 0 for index in indices):
        raise ValueError(
            "available_indices cannot contain negative values."
        )

    split_sizes = {
        "training_size": training_size,
        "stage1_validation_size": stage1_validation_size,
        "ptq_calibration_size": ptq_calibration_size,
        "stage2_fit_size": stage2_fit_size,
        "stage2_validation_size": stage2_validation_size,
    }

    for name, size in split_sizes.items():
        if size <= 0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

    requested_size = sum(split_sizes.values())

    if requested_size > len(indices):
        raise ValueError(
            "Requested split sizes exceed the number of "
            "available indices."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    permutation = torch.randperm(
        len(indices),
        generator=generator,
    ).tolist()

    shuffled_indices = [
        indices[position]
        for position in permutation
    ]

    start = 0

    def take(size: int) -> Tuple[int, ...]:
        nonlocal start

        end = start + size
        selected = tuple(
            shuffled_indices[start:end]
        )
        start = end

        return selected

    training_indices = take(training_size)

    stage1_validation_indices = take(
        stage1_validation_size
    )

    ptq_calibration_indices = take(
        ptq_calibration_size
    )

    stage2_fit_indices = take(
        stage2_fit_size
    )

    stage2_validation_indices = take(
        stage2_validation_size
    )

    unused_indices = tuple(
        shuffled_indices[start:]
    )

    return ExperimentDataSplit(
        training_indices=training_indices,
        stage1_validation_indices=(
            stage1_validation_indices
        ),
        ptq_calibration_indices=(
            ptq_calibration_indices
        ),
        stage2_fit_indices=stage2_fit_indices,
        stage2_validation_indices=(
            stage2_validation_indices
        ),
        unused_indices=unused_indices,
    )


def save_experiment_data_split(
    split: ExperimentDataSplit,
    output_path: str | Path,
    *,
    dataset_name: str,
    seed: int,
) -> Path:
    """Save an experiment split as a human-readable JSON file."""

    if not isinstance(split, ExperimentDataSplit):
        raise TypeError(
            "split must be an ExperimentDataSplit."
        )

    dataset_name = dataset_name.strip()

    if not dataset_name:
        raise ValueError(
            "dataset_name cannot be empty."
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
        "dataset_name": dataset_name,
        "seed": int(seed),
        "splits": {
            "training": list(
                split.training_indices
            ),
            "stage1_validation": list(
                split.stage1_validation_indices
            ),
            "ptq_calibration": list(
                split.ptq_calibration_indices
            ),
            "stage2_fit": list(
                split.stage2_fit_indices
            ),
            "stage2_validation": list(
                split.stage2_validation_indices
            ),
            "unused": list(
                split.unused_indices
            ),
        },
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path


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


@dataclass(frozen=True)
class ExperimentDatasetSubsets:
    """Dataset subsets used throughout the complete experiment."""

    training: torch.utils.data.Subset
    stage1_validation: torch.utils.data.Subset
    ptq_calibration: torch.utils.data.Subset
    stage2_fit: torch.utils.data.Subset
    stage2_validation: torch.utils.data.Subset
    unused: torch.utils.data.Subset


def create_experiment_dataset_subsets(
    training_dataset: torch.utils.data.Dataset,
    evaluation_dataset: torch.utils.data.Dataset,
    split: ExperimentDataSplit,
) -> ExperimentDatasetSubsets:
    """Apply saved experiment indices to training and evaluation datasets.

    Both datasets must represent the same underlying samples in the same
    order. They may use different transforms.
    """

    if len(training_dataset) != len(evaluation_dataset):
        raise ValueError(
            "training_dataset and evaluation_dataset must have "
            "the same number of samples."
        )

    dataset_size = len(training_dataset)

    if dataset_size == 0:
        raise ValueError(
            "Datasets cannot be empty."
        )

    partitions = {
        "training": split.training_indices,
        "stage1_validation": (
            split.stage1_validation_indices
        ),
        "ptq_calibration": (
            split.ptq_calibration_indices
        ),
        "stage2_fit": split.stage2_fit_indices,
        "stage2_validation": (
            split.stage2_validation_indices
        ),
        "unused": split.unused_indices,
    }

    all_indices = []

    for name, indices in partitions.items():
        if any(
            index < 0 or index >= dataset_size
            for index in indices
        ):
            raise ValueError(
                f"Partition '{name}' contains an index outside "
                f"the dataset range [0, {dataset_size - 1}]."
            )

        all_indices.extend(indices)

    if len(all_indices) != len(set(all_indices)):
        raise ValueError(
            "Experiment dataset partitions overlap."
        )

    return ExperimentDatasetSubsets(
        training=torch.utils.data.Subset(
            training_dataset,
            split.training_indices,
        ),
        stage1_validation=torch.utils.data.Subset(
            evaluation_dataset,
            split.stage1_validation_indices,
        ),
        ptq_calibration=torch.utils.data.Subset(
            evaluation_dataset,
            split.ptq_calibration_indices,
        ),
        stage2_fit=torch.utils.data.Subset(
            evaluation_dataset,
            split.stage2_fit_indices,
        ),
        stage2_validation=torch.utils.data.Subset(
            evaluation_dataset,
            split.stage2_validation_indices,
        ),
        unused=torch.utils.data.Subset(
            evaluation_dataset,
            split.unused_indices,
        ),
    )


@dataclass(frozen=True)
class ExperimentDataLoaders:
    """DataLoaders used throughout Stage 1 and Stage 2."""

    training: torch.utils.data.DataLoader
    stage1_validation: torch.utils.data.DataLoader
    ptq_calibration: torch.utils.data.DataLoader
    stage2_fit: torch.utils.data.DataLoader
    stage2_validation: torch.utils.data.DataLoader
    unused: torch.utils.data.DataLoader


def create_experiment_dataloaders(
    subsets: ExperimentDatasetSubsets,
    *,
    training_batch_size: int = 96,
    evaluation_batch_size: int = 64,
    ptq_batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> ExperimentDataLoaders:
    """Create deterministic experiment DataLoaders.

    Only the model-training loader is shuffled. Calibration and validation
    loaders preserve their saved index order for reproducibility.
    """

    batch_sizes = {
        "training_batch_size": training_batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "ptq_batch_size": ptq_batch_size,
    }

    for name, batch_size in batch_sizes.items():
        if batch_size <= 0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative."
        )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    def build_loader(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool,
    ) -> torch.utils.data.DataLoader:
        arguments = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "persistent_workers": False,
            "pin_memory": bool(pin_memory),
        }

        if num_workers > 0:
            arguments["prefetch_factor"] = 2

        return torch.utils.data.DataLoader(
            **arguments
        )

    return ExperimentDataLoaders(
        training=build_loader(
            subsets.training,
            training_batch_size,
            shuffle=True,
        ),
        stage1_validation=build_loader(
            subsets.stage1_validation,
            evaluation_batch_size,
            shuffle=False,
        ),
        ptq_calibration=build_loader(
            subsets.ptq_calibration,
            ptq_batch_size,
            shuffle=False,
        ),
        stage2_fit=build_loader(
            subsets.stage2_fit,
            evaluation_batch_size,
            shuffle=False,
        ),
        stage2_validation=build_loader(
            subsets.stage2_validation,
            evaluation_batch_size,
            shuffle=False,
        ),
        unused=build_loader(
            subsets.unused,
            evaluation_batch_size,
            shuffle=False,
        ),
    )
