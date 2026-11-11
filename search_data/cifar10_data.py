"""CIFAR-10 adapter for the reliability-calibration framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from utils.data import (
    _EVAL_TRANSFORM,
    _TRAIN_TRANSFORM,
    _load_cifar10_dataset,
)

from .calibration_data import (
    ExperimentDataLoaders,
    create_experiment_dataloaders,
    create_experiment_dataset_subsets,
    load_experiment_data_split,
)


@dataclass(frozen=True)
class CIFAR10DataBundle:
    """All CIFAR-10 loaders used by the two-stage experiment."""

    experiment: ExperimentDataLoaders
    final_test: torch.utils.data.DataLoader


def create_cifar10_data_bundle(
    cifar_10_dir: str,
    split_path: str | Path,
    *,
    training_batch_size: int = 96,
    evaluation_batch_size: int = 64,
    ptq_batch_size: int = 16,
    num_workers: int = 0,
    allow_download: bool = False,
) -> CIFAR10DataBundle:
    """Create clean, non-overlapping CIFAR-10 experiment loaders."""

    split = load_experiment_data_split(
        split_path
    )

    training_dataset = _load_cifar10_dataset(
        cifar_10_dir=cifar_10_dir,
        train=True,
        transform=_TRAIN_TRANSFORM,
        allow_download=allow_download,
    )

    evaluation_dataset = _load_cifar10_dataset(
        cifar_10_dir=cifar_10_dir,
        train=True,
        transform=_EVAL_TRANSFORM,
        allow_download=allow_download,
    )

    subsets = create_experiment_dataset_subsets(
        training_dataset=training_dataset,
        evaluation_dataset=evaluation_dataset,
        split=split,
    )

    experiment_loaders = create_experiment_dataloaders(
        subsets=subsets,
        training_batch_size=training_batch_size,
        evaluation_batch_size=evaluation_batch_size,
        ptq_batch_size=ptq_batch_size,
        num_workers=num_workers,
    )

    final_test_dataset = _load_cifar10_dataset(
        cifar_10_dir=cifar_10_dir,
        train=False,
        transform=_EVAL_TRANSFORM,
        allow_download=allow_download,
    )

    final_test_loader = torch.utils.data.DataLoader(
        final_test_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return CIFAR10DataBundle(
        experiment=experiment_loaders,
        final_test=final_test_loader,
    )
