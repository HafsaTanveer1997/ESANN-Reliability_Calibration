#!/usr/bin/env python3
"""Stage-2 Optuna search for reliability calibration.

Hyperparameters are fitted on an inner-fit partition derived from Stage-2 fit
and selected on a disjoint inner-validation partition. Stage-2 validation and
the official CIFAR-10 test set are not evaluated during Optuna search.

Deployment refitting, PT2E quantization, LiteRT/TFLite export, and Stage-2
validation evaluation are performed only after hyperparameter search.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


# Make benchmark/util, image_classification, and sibling vit imports available.
IMAGE_CLASSIFICATION_DIR = Path(__file__).resolve().parents[1]
TRAINING_TORCH_DIR = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = Path(__file__).resolve().parents[3]

for import_root in (
    BENCHMARK_ROOT,
    IMAGE_CLASSIFICATION_DIR,
    TRAINING_TORCH_DIR,
):
    import_root_str = str(import_root)
    while import_root_str in sys.path:
        sys.path.remove(import_root_str)
    sys.path.insert(0, import_root_str)


from util.optuna_search import (  # noqa: E402
    OptunaTuner,
    add_tuning_args,
    load_ssd,
)

from search_data.calibration_data import (  # noqa: E402
    create_stratified_fit_validation_split,
)
from calibration.calibration_methods import (  # noqa: E402
    fit_and_apply_calibration,
)
from search_data.cifar10_data import (  # noqa: E402
    CIFAR10DataBundle,
    create_cifar10_data_bundle,
)
from search_data.logit_cache import (  # noqa: E402
    load_logit_cache,
)
from calibration.reliability_metrics import (  # noqa: E402
    accuracy_guard_passed,
    compute_reliability_risk,
    evaluate_logits,
)
from deployment.reproducibility import (  # noqa: E402
    set_reproducibility_seed,
)


class EmptyHistory:
    """Minimal history object required by OptunaTuner."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {}


_STAGE2_DATA_BUNDLE: CIFAR10DataBundle | None = None
_STAGE2_DATA_BUNDLE_KEY: tuple[Any, ...] | None = None


def get_stage2_data_bundle(
    flags,
) -> CIFAR10DataBundle:
    """Create or reuse the clean Stage-2 image DataLoaders."""

    if getattr(flags, "dataset", "cifar10") == "cifar100":
        from search_data.cifar100_data import (
            create_cifar100_data_bundle,
        )
        if int(flags.ptq_batch_size) != 1:
            raise ValueError("CIFAR-100 PTQ batch size must be 1.")
        return create_cifar100_data_bundle(
            cifar_100_dir=flags.cifar100_path,
            split_path=flags.split_path,
            evaluation_batch_size=int(flags.evaluation_batch_size),
            ptq_batch_size=1,
            num_workers=int(flags.num_workers),
            allow_download=bool(flags.allow_download),
        )

    global _STAGE2_DATA_BUNDLE
    global _STAGE2_DATA_BUNDLE_KEY

    if int(flags.ptq_batch_size) != 1:
        raise ValueError(
            "--ptq_batch_size must be 1 because post-search deployment "
            "uses a static batch-1 LiteRT/TFLite model."
        )

    bundle_key = (
        str(Path(flags.cifar10_path).expanduser().resolve()),
        str(Path(flags.split_path).expanduser().resolve()),
        int(flags.evaluation_batch_size),
        int(flags.ptq_batch_size),
        int(flags.num_workers),
        bool(flags.allow_download),
    )

    if (
        _STAGE2_DATA_BUNDLE is None
        or _STAGE2_DATA_BUNDLE_KEY != bundle_key
    ):
        _STAGE2_DATA_BUNDLE = create_cifar10_data_bundle(
            cifar_10_dir=flags.cifar10_path,
            split_path=flags.split_path,
            evaluation_batch_size=int(
                flags.evaluation_batch_size
            ),
            ptq_batch_size=int(flags.ptq_batch_size),
            num_workers=int(flags.num_workers),
            allow_download=bool(flags.allow_download),
        )
        _STAGE2_DATA_BUNDLE_KEY = bundle_key

    return _STAGE2_DATA_BUNDLE


def _collect_loader_labels(dataloader) -> torch.Tensor:
    """Collect labels in deterministic DataLoader order."""

    labels = []

    for _, batch_labels in dataloader:
        labels.append(
            batch_labels.detach().cpu().reshape(-1)
        )

    if not labels:
        raise ValueError(
            "The Stage-2 DataLoader did not provide any labels."
        )

    return torch.cat(labels, dim=0).to(dtype=torch.long)


def prepare_stage2_data(ssd, flags):
    """Load cached logits and Stage-2 validation images for one candidate."""

    inner_validation_size = int(
        ssd["stage2_inner_validation_size"]
    )
    inner_split_seed = int(
        ssd["stage2_inner_split_seed"]
    )

    candidate_directory = (
        Path(flags.cache_root).expanduser().resolve()
        / flags.candidate_id
    )

    fit_path = candidate_directory / "stage2_fit_logits.pt"
    if not fit_path.is_file():
        raise FileNotFoundError(fit_path)

    fit_logits, fit_labels, _ = load_logit_cache(
        fit_path
    )
    data_bundle = get_stage2_data_bundle(flags)
    stage2_fit_loader = data_bundle.experiment.stage2_fit
    if len(stage2_fit_loader.dataset) != int(
        fit_labels.numel()
    ):
        raise RuntimeError(
            "Cached Stage-2 fit sample count does not match the split "
            "DataLoader. Check --split_path and --cache_root."
        )

    # Confirm that the cached fit logits use exactly the saved Stage-2 fit
    # partition and deterministic DataLoader order.
    loader_fit_labels = _collect_loader_labels(
        stage2_fit_loader
    )
    if not torch.equal(
        fit_labels.detach().cpu().to(dtype=torch.long),
        loader_fit_labels,
    ):
        raise RuntimeError(
            "Cached Stage-2 fit labels do not match the Stage-2 fit "
            "DataLoader order."
        )

    full_fit_size = int(fit_labels.numel())
    method = str(ssd.get("stage2_selection_method", "full"))
    budget = ssd.get("stage2_selection_budget")
    seed = int(ssd.get("stage2_selection_seed", 42))
    independent_inner_validation = bool(
        ssd.get("stage2_independent_inner_validation", False)
    )
    independent_validation_size = int(
        ssd.get("stage2_independent_inner_validation_size", 102)
    )

    if independent_inner_validation:
        if independent_validation_size <= 0:
            raise ValueError(
                "Independent inner-validation size must be greater than zero."
            )
        if independent_validation_size >= full_fit_size:
            raise ValueError(
                "Independent inner-validation size must be smaller than "
                "the Stage-2 fit pool."
            )

        common_split = create_stratified_fit_validation_split(
            fit_labels,
            validation_size=independent_validation_size,
            seed=inner_split_seed,
        )

        pool_positions = list(common_split.fit_indices)
        validation_positions = list(common_split.validation_indices)

        validation_indices = torch.tensor(
            validation_positions,
            dtype=torch.long,
        )
        inner_validation_data = (
            fit_logits.index_select(0, validation_indices),
            fit_labels.index_select(0, validation_indices),
        )

        if method == "full":
            selected_positions = pool_positions

        elif method == "random":
            if budget is None:
                raise ValueError("Random selection requires a budget.")

            fit_budget = int(budget) - independent_validation_size
            if fit_budget <= 0:
                raise ValueError(
                    "Selection budget must exceed the independent "
                    "inner-validation size."
                )
            if fit_budget > len(pool_positions):
                raise ValueError(
                    "Random fit budget exceeds the available fit pool."
                )

            selected_local_positions = np.random.default_rng(seed).choice(
                len(pool_positions),
                size=fit_budget,
                replace=False,
            ).tolist()
            selected_positions = [
                pool_positions[index]
                for index in selected_local_positions
            ]



        else:
            raise ValueError(
                f"Unknown Stage-2 selection method: {method!r}"
            )

        if not selected_positions:
            raise RuntimeError("Stage-2 selection returned no samples.")

        selected_indices = torch.tensor(
            selected_positions,
            dtype=torch.long,
        )
        inner_fit_data = (
            fit_logits.index_select(0, selected_indices),
            fit_labels.index_select(0, selected_indices),
        )

        inner_fit_count = len(selected_positions)
        inner_validation_count = len(validation_positions)

        print(
            "Stage-2 independent inner-validation protocol:",
            f"fit={inner_fit_count}",
            f"validation={inner_validation_count}",
            f"total={inner_fit_count + inner_validation_count}",
        )
        print(
            "Stage-2 selection:",
            method,
            f"{inner_fit_count} fit samples from "
            f"{len(pool_positions)} available after validation reservation",
        )

    else:
        if method == "full":
            selected_positions = list(range(full_fit_size))
        elif method == "random":
            if budget is None:
                raise ValueError("Random selection requires a budget.")
            selected_positions = np.random.default_rng(seed).choice(
                full_fit_size,
                size=int(budget),
                replace=False,
            ).tolist()

        else:
            raise ValueError(
                f"Unknown Stage-2 selection method: {method!r}"
            )

        if not selected_positions:
            raise RuntimeError("Stage-2 selection returned no samples.")

        selected_indices = torch.tensor(
            selected_positions,
            dtype=torch.long,
        )
        selected_fit_logits = fit_logits.index_select(
            0, selected_indices
        )
        selected_fit_labels = fit_labels.index_select(
            0, selected_indices
        )

        selected_fit_size = int(selected_fit_labels.numel())
        scaled_validation_size = round(
            inner_validation_size
            * selected_fit_size
            / full_fit_size
        )

        print(
            "Stage-2 selection:",
            method,
            f"{selected_fit_size}/{full_fit_size}",
        )

        inner_split = create_stratified_fit_validation_split(
            selected_fit_labels,
            validation_size=scaled_validation_size,
            seed=inner_split_seed,
        )

        inner_fit_indices = torch.tensor(
            inner_split.fit_indices,
            dtype=torch.long,
        )
        inner_validation_indices = torch.tensor(
            inner_split.validation_indices,
            dtype=torch.long,
        )

        inner_fit_data = (
            selected_fit_logits.index_select(0, inner_fit_indices),
            selected_fit_labels.index_select(0, inner_fit_indices),
        )
        inner_validation_data = (
            selected_fit_logits.index_select(
                0, inner_validation_indices
            ),
            selected_fit_labels.index_select(
                0, inner_validation_indices
            ),
        )

        inner_fit_count = len(inner_split.fit_indices)
        inner_validation_count = len(inner_split.validation_indices)

    print(
        "Stage-2 Optuna inner-fit samples:",
        inner_fit_count,
    )
    print(
        "Stage-2 Optuna inner-validation samples:",
        inner_validation_count,
    )
    print(
        "Stage-2 inner split seed:",
        inner_split_seed,
    )
    print(
        "Stage-2 validation is not loaded during Optuna search."
    )

    # Optuna receives only partitions derived from Stage-2 fit.
    # Stage-2 validation is intentionally unavailable during hyperparameter
    # search and is reconstructed only during post-search evaluation.
    return (
        inner_fit_data,
        inner_validation_data,
        None,
    )


def _create_trial_directory(tuner, trial) -> Path:
    """Create a persistent trial directory when available."""

    if tuner.files_root is not None:
        trial_directory = (
            Path(tuner.files_root)
            / f"{trial.number:04d}"
        )
        trial_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        return trial_directory

    # This fallback mirrors hyper_search_train.py's temporary export path.
    return Path(
        tempfile.mkdtemp(
            prefix=(
                "stage2_calibration_"
                f"trial_{trial.number:04d}_"
            )
        )
    ).resolve()



def calibrate_and_test(
    tuner,
    trial,
    flags,
    model_params,
):
    """Fit and evaluate one calibration configuration on the inner split."""

    fit_logits, fit_labels = tuner.ds_train
    validation_logits, validation_labels = tuner.ds_val

    calibration_seed = int(tuner.ssd["calibration_seed"])
    trial_seed = set_reproducibility_seed(
        calibration_seed + int(trial.number)
    )

    method = str(model_params["calibration_method"])
    num_ece_bins = int(model_params["num_ece_bins"])
    confidence_threshold = float(
        model_params["confidence_threshold"]
    )

    baseline_metrics = evaluate_logits(
        validation_logits,
        validation_labels,
        num_ece_bins=num_ece_bins,
        confidence_threshold=confidence_threshold,
    )

    risk_arguments = {
        "num_classes": int(validation_logits.shape[1]),
        "nll_weight": float(model_params["nll_weight"]),
        "brier_weight": float(
            model_params["brier_weight"]
        ),
        "ece_weight": float(model_params["ece_weight"]),
        "high_confidence_error_weight": float(
            model_params[
                "high_confidence_error_weight"
            ]
        ),
    }

    baseline_risk = compute_reliability_risk(
        baseline_metrics,
        **risk_arguments,
    )

    (
        calibrated_logits,
        learned_parameters,
        method_results,
    ) = fit_and_apply_calibration(
        fit_logits=fit_logits,
        fit_labels=fit_labels,
        evaluation_logits=validation_logits,
        model_params=model_params,
    )

    calibrated_metrics = evaluate_logits(
        calibrated_logits,
        validation_labels,
        num_ece_bins=num_ece_bins,
        confidence_threshold=confidence_threshold,
    )
    calibrated_risk = compute_reliability_risk(
        calibrated_metrics,
        **risk_arguments,
    )

    maximum_allowed_accuracy_drop = float(
        model_params["maximum_allowed_accuracy_drop"]
    )

    inner_guard_passed = accuracy_guard_passed(
        baseline_top1=float(baseline_metrics["top1"]),
        calibrated_top1=float(calibrated_metrics["top1"]),
        maximum_allowed_drop=maximum_allowed_accuracy_drop,
    )

    real_risk = float(
        calibrated_risk["reliability_risk"]
    )
    objective_risk = real_risk

    if not inner_guard_passed:
        objective_risk += float(
            model_params["accuracy_guard_penalty"]
        )

    test_results = {
        **method_results,
        "inner_baseline_top1": float(
            baseline_metrics["top1"]
        ),
        "inner_calibrated_top1": float(
            calibrated_metrics["top1"]
        ),
        "inner_accuracy_drop": float(
            baseline_metrics["top1"]
            - calibrated_metrics["top1"]
        ),
        "inner_accuracy_guard_passed": bool(
            inner_guard_passed
        ),
        "inner_baseline_nll": float(
            baseline_metrics["nll"]
        ),
        "inner_calibrated_nll": float(
            calibrated_metrics["nll"]
        ),
        "inner_baseline_brier": float(
            baseline_metrics["brier"]
        ),
        "inner_calibrated_brier": float(
            calibrated_metrics["brier"]
        ),
        "inner_baseline_ece": float(
            baseline_metrics["ece"]
        ),
        "inner_calibrated_ece": float(
            calibrated_metrics["ece"]
        ),
        "inner_baseline_high_confidence_error": float(
            baseline_metrics["high_confidence_error"]
        ),
        "inner_calibrated_high_confidence_error": float(
            calibrated_metrics["high_confidence_error"]
        ),
        "inner_baseline_reliability_risk": float(
            baseline_risk["reliability_risk"]
        ),
        "inner_calibrated_reliability_risk": real_risk,
        "inner_reliability_risk_improvement": float(
            baseline_risk["reliability_risk"]
            - real_risk
        ),
        "objective_reliability_risk": float(
            objective_risk
        ),
    }

    trial_directory = _create_trial_directory(
        tuner,
        trial,
    )
    result_path = (
        trial_directory
        / "search_result.json"
    )

    result = {
        "trial_number": int(trial.number),
        "candidate_id": str(flags.candidate_id),
        "calibration_method": method,
        "hyperparameters": {
            key: value
            for key, value in model_params.items()
            if key != "baseline"
        },
        "baseline_trial": bool(
            model_params["baseline"]
        ),
        "search_data": {
            "fit_partition": "stage2_fit_inner_fit",
            "validation_partition": (
                "stage2_fit_inner_validation"
            ),
            "fit_samples": int(
                fit_labels.numel()
            ),
            "validation_samples": int(
                validation_labels.numel()
            ),
            "inner_split_seed": int(
                tuner.ssd["stage2_inner_split_seed"]
            ),
            "calibration_seed": calibration_seed,
            "trial_seed": trial_seed,
        },
        # These parameters are retained only to make each search trial
        # auditable. The selected hyperparameters will be refitted later
        # on the complete Stage-2 fit partition.
        "search_learned_parameters": learned_parameters,
        "metrics": test_results,
        "stage2_validation_status": (
            "not_used_during_optuna_search"
        ),
        "deployment_status": (
            "not_run_during_optuna_search"
        ),
    }

    result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "Stage-2 inner-validation calibrated top-1:",
        f"{test_results['inner_calibrated_top1']:.6f}",
    )
    print(
        "Stage-2 inner-validation reliability risk:",
        f"{real_risk:.6f}",
    )
    print(
        "Stage-2 inner-validation accuracy guard passed:",
        inner_guard_passed,
    )

    model_info = {
        "candidate_id": flags.candidate_id,
        "calibration_method": method,
        "search_learned_parameters": learned_parameters,
        "result_path": str(result_path),
    }

    return (
        model_info,
        EmptyHistory(),
        test_results,
    )


def run_hp(parser):
    """Run the calibration hyperparameter search."""

    add_tuning_args(parser)
    flags, _ = parser.parse_known_args()

    if flags.dataset == "cifar100" and not flags.cifar100_path:
        parser.error("--cifar100_path is required for CIFAR-100 runs.")
    if not flags.candidate_id:
        parser.error("--candidate_id is required")

    ssd = load_ssd(flags.ssd)
    # IFX requires this key for validation. The classifier stays fixed.
    ssd.setdefault("qat_bits", None)
    if flags.hyperparam_trials is not None:
        ssd["trials"] = flags.hyperparam_trials
    if flags.study_name is not None:
        ssd["study_name"] = flags.study_name

    ssd["stage2_selection_method"] = flags.selection_method
    ssd["stage2_selection_budget"] = flags.selection_budget
    ssd["stage2_selection_seed"] = flags.selection_seed
    ssd["stage2_independent_inner_validation"] = (
        flags.independent_inner_validation
    )
    ssd["stage2_independent_inner_validation_size"] = (
        flags.independent_inner_validation_size
    )


    # Keep Full backward-compatible and isolate subset-selection studies.
    selection_suffix = ""
    if flags.selection_method != "full":
        selection_suffix = (
            f"_{flags.selection_method}"
            f"{flags.selection_budget}"
            f"_seed{flags.selection_seed}"
        )
    protocol_suffix = ""
    if flags.independent_inner_validation:
        protocol_suffix = (
            f"_indval{flags.independent_inner_validation_size}"
        )
    ssd["study_name"] = (
        f"{ssd['study_name']}_"
        f"{flags.candidate_id}"
        f"{selection_suffix}"
        f"{protocol_suffix}"
    )

    # Keep the selection and protocol suffixes when IFX applies CLI overrides.
    if flags.study_name is not None:
        flags.study_name = ssd["study_name"]

    tuner = OptunaTuner(
        ssd=ssd,
        flags=flags,
        do_trial=calibrate_and_test,
        callback_functions={
            "prepare_stage2_data": (
                prepare_stage2_data
            ),
        },
        verbose=True,
    )
    study = tuner.run_search_space()
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Nested Optuna Stage-2 reliability calibration search using "
            "inner partitions derived exclusively from Stage-2 fit."
        )
    )

    parser.add_argument(
        "--candidate_id",
        type=str,
        default=None,
        help=(
            "Stage-1 candidate ID, for example resnet8_trial_1."
        ),
    )
    parser.add_argument(
        "--candidate_manifest",
        type=str,
        default=(
            "reliability_calibration/"
            "stage1_pareto_candidates.json"
        ),
        help=(
            "Stage-1 Pareto candidate manifest containing the float model "
            "used for calibration folding."
        ),
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default=(
            "reliability_calibration/"
            "outputs/stage2_logits"
        ),
        help=(
            "Directory containing cached Stage-2 logits."
        ),
    )
    parser.add_argument(
        "--split_path",
        type=str,
        default=(
            "reliability_calibration/"
            "cifar10_experiment_split_seed42.json"
        ),
        help=(
            "Saved clean split used to reconstruct PTQ and Stage-2 "
            "validation image loaders."
        ),
    )
    parser.add_argument(
        "--dataset", choices=("cifar10", "cifar100"), default="cifar10",
    )
    parser.add_argument(
        "--cifar100_path",
    )
    parser.add_argument(
        "--cifar10_path",
        type=str,
        default="cifar-10-batches-py",
    )
    parser.add_argument(
        "--evaluation_batch_size",
        type=int,
        default=64,
        help=(
            "Batch size for Stage-2 image loaders."
        ),
    )
    parser.add_argument(
        "--ptq_batch_size",
        type=int,
        default=1,
        help=(
            "Must remain 1 for static batch-1 TFLite deployment evaluation."
        ),
    )
    parser.add_argument(
        "--selection_method",
        choices=("full", "random"),
        default="full",
    )
    parser.add_argument(
        "--selection_budget",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--independent_inner_validation",
        action="store_true",
        help=(
            "Reserve the inner-validation partition before Stage-2 "
            "subset selection."
        ),
    )
    parser.add_argument(
        "--independent_inner_validation_size",
        type=int,
        default=102,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--allow_download",
        action="store_true",
    )

    run_hp(parser)
