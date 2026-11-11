"""Fit, export, and evaluate recorded CIFAR-100 calibration studies."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from deployment.splits import load_experiment_data_split
from models import create_model
from models.checkpoint import load_checkpoint
from torchvision.datasets import CIFAR100
from deployment.pt2e_export import export_pt2e_and_tflite
from deployment.reproducibility import set_reproducibility_seed
from deployment.tflite_runner import collect_tflite_logits
from torchvision.transforms import ToTensor
from ai_edge_litert import schema_py_generated as schema

from calibration.anchored_components import fit_temperature, fit_ta_cgsrh
from calibration.metrics_adapter import evaluate



def sha(path):
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """Write JSON via a temporary file to avoid partial result records."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def operators(path):
    """Count TFLite operators across all subgraphs."""
    model = schema.Model.GetRootAsModel(Path(path).read_bytes(), 0)
    return sum(
        model.Subgraphs(i).OperatorsLength()
        for i in range(model.SubgraphsLength())
    )


def metrics(logits, labels):
    """Compute development metrics and high-confidence error counts."""
    z = torch.as_tensor(logits).detach().cpu().float()
    y = torch.as_tensor(labels).detach().cpu().long()
    result = evaluate(z.numpy(), y.numpy())
    confidence, prediction = z.softmax(1).max(1)
    high = confidence.ge(0.6)
    errors = int((high & prediction.ne(y)).sum())
    count = int(high.sum())
    result.update({
        "confident_errors": errors,
        "high_confidence_predictions": count,
        "conditional_confident_error_rate": errors/count if count else None,
    })
    return result


def fit(job, protocol, fit_logits, labels):
    """Fit the configured calibrator and return its affine logit transform."""
    configuration = job["configuration"]
    if configuration["temperature_only"]:
        temperature, info = fit_temperature(fit_logits, labels)
        matrix = torch.eye(fit_logits.shape[1], dtype=torch.float64) / temperature
        bias = torch.zeros(fit_logits.shape[1], dtype=torch.float64)
        return matrix, bias, info

    arguments = dict(protocol["correction_settings"])
    arguments.update({
        key: value for key, value in configuration.items()
        if key not in ("name", "temperature_only")
    })
    model = fit_ta_cgsrh(fit_logits, labels, **arguments)
    return model.matrix, model.bias, model.info


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_path(root, name, expected_hash, expected_size=None):
    """Resolve a manifest path within its declared root and verify its bytes."""
    root = root.resolve()
    relative = Path(name)
    path = (root / relative).resolve()
    require(not relative.is_absolute() and root in path.parents,
            f"Expected a relative path inside {root}: {name}")
    require(path.is_file(), f"Missing frozen input: {path}")
    if expected_size is not None:
        require(path.stat().st_size == expected_size, f"Size mismatch: {path}")
    require(sha(path) == expected_hash, f"SHA-256 mismatch: {path}")
    return path


def _baseline_tflite(args, candidate_id):
    """Verify the original INT8 artifact referenced by the model manifest."""
    records = json.loads((args.artifact_root / "manifest.json").read_text())["models"]
    matches = [row for row in records if row["candidate_id"] == candidate_id]
    require(len(matches) == 1, f"Expected one bundle entry for {candidate_id}.")
    artifact = matches[0]["artifacts"]["tflite_model"]
    return str(verified_path(
        args.artifact_root, artifact["path"], artifact["sha256"], artifact["bytes"]
    ))


def _resnet32_plan(args, settings):
    """Load the explicit job list for the recorded ResNet-32 repeat study."""
    project = args.input_root.expanduser().resolve()
    split_path = verified_path(project, settings["split"], settings["split_sha256"])
    cid = "resnet32_cifar100_trial_23"
    candidates = {cid: {"artifacts": {"tflite_model": _baseline_tflite(args, cid)}}}
    jobs = [dict(row) for row in settings[args.phase]]
    if len(jobs) != 20 or any(job["candidate_id"] != cid for job in jobs):
        raise ValueError("Unexpected ResNet-32 repeat plan.")

    verified = {}
    for job in jobs:
        name = job["configuration"]["name"]
        if job["selector"] != "random" or not name or name in (".", "..") or Path(name).name != name:
            raise ValueError("Invalid selector or configuration name.")
        key = (job["development_export"], job["development_export_sha256"])
        if key not in verified:
            verified[key] = str(verified_path(project, *key))
        job["development_export"] = verified[key]
    return jobs, candidates, split_path


def _mobilenet_plan(args, settings):
    """Build deployment jobs from the fixed configurations and fitting seeds."""
    cid = "mobilenet_v2_cifar100_trial_26"
    require(settings["candidate_id"] == cid and settings["num_classes"] == 100,
            "Expected the primary MobileNetV2/CIFAR-100 settings.")
    require(settings["selector"] == "random" and settings["selection_seed"] == 42
            and settings["reselection_allowed"] is False,
            "Expected the frozen random-selection protocol.")
    seeds = settings["repeated_fitting_seeds"]
    configs = settings["development_configurations"]
    selected = settings["selected_configuration"]
    temperature = [c for c in configs if c["temperature_only"]]
    require(seeds == list(range(1, 11)) and len(configs) == 9
            and len({c["name"] for c in configs}) == 9 and len(temperature) == 1
            and not selected["temperature_only"],
            "Unexpected frozen configurations or fitting seeds.")

    exports = {row["selection_seed"]: row for row in settings["input_exports"]}
    require(len(settings["input_exports"]) == 11 and set(exports) == {42, *seeds},
            "Expected one verified export for each frozen fitting seed.")
    data = settings["data"]
    input_root = args.input_root.expanduser().resolve()

    def project_reference(name):
        # Normalize settings-relative paths to the project layout.
        path = Path(name)
        if path.parts[:1] == ("..",):
            return (args.settings.parent / path).resolve().relative_to(input_root)
        return path

    split_path = verified_path(
        input_root, project_reference(data["split"]), data["split_sha256"]
    )

    artifacts = {"tflite_model": _baseline_tflite(args, cid)}

    active_seeds, active_configs = (
        ([42], configs) if args.phase == "development" else (seeds, [selected, *temperature])
    )
    jobs = []
    for seed in active_seeds:
        row = exports[seed]
        export = verified_path(
            input_root, project_reference(row["path"]), row["sha256"]
        )
        for configuration in active_configs:
            name = configuration["name"]
            require(name not in ("", ".", "..") and Path(name).name == name,
                    f"Invalid configuration name: {name!r}")
            jobs.append({
                "candidate_id": cid,
                "configuration": configuration,
                "selector": settings["selector"],
                "selection_seed": seed,
                "development_export": str(export),
                "development_export_sha256": row["sha256"],
            })
    return jobs, {cid: {"artifacts": artifacts}}, split_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("mobilenetv2", "resnet32"))
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--phase", choices=("development", "repeat"), required=True)
    parser.add_argument("--run", action="store_true", help="Fit and export after validation.")
    args = parser.parse_args(argv)
    if args.model == "resnet32" and args.phase != "repeat":
        parser.error("ResNet-32 supports only --phase repeat.")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Model weights not found: {checkpoint}")
    if not __debug__:
        parser.error("Run without Python optimization; deployment assertions are required.")
    require(args.data_root.is_dir(), f"Missing CIFAR-100 data directory: {args.data_root}")

    settings_bytes = args.settings.read_bytes()
    settings = json.loads(settings_bytes)
    if args.model == "resnet32":
        jobs, candidates, split_path = _resnet32_plan(args, settings)
        model_name, classifier_path = "resnet32_cifar100", "fc"
        output_namespace = model_name
        drop_limit = settings["accuracy_drop_limit"]
    else:
        jobs, candidates, split_path = _mobilenet_plan(args, settings)
        model_name, classifier_path = "mobilenet_v2_cifar100", "classifier.1"
        output_namespace = settings["candidate_id"]
        drop_limit = settings["eligibility"]["deployed_development_accuracy_drop_max"]
    print(f"PASS: {len(jobs)} {args.phase} jobs verified.", flush=True)
    if not args.run:
        print("Validation only; no fitting or export.")
        return

    # Store development and repeated fitting outputs in separate directories.
    torch.set_num_threads(1)
    protocol = {
        "correction_settings": settings["correction_settings"],
        "accuracy_drop_limit": drop_limit,
    }
    output_root = args.output_root.expanduser().resolve() / output_namespace / args.phase
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = {
            "protocol_sha256": hashlib.sha256(settings_bytes).hexdigest(),
            "runner_sha256": sha(Path(__file__).resolve()),
            "checkpoint_sha256": sha(checkpoint),
            "artifact_manifest_sha256": sha(args.artifact_root / "manifest.json"),
        }
        audit_path = output_root / "runner_audit.json"
        if audit_path.exists():
            require(json.loads(audit_path.read_text()) == audit, "Output audit changed.")
        else:
            save(audit_path, audit)

        # Load training data only, using the recorded PTQ and development indices.
        split = load_experiment_data_split(split_path)
        data_root = args.data_root.expanduser().resolve()
        if data_root.name == "cifar-100-python":
            data_root = data_root.parent
        dataset = CIFAR100(
            root=str(data_root), train=True, transform=ToTensor(), download=False,
        )
        ptq = DataLoader(
            Subset(dataset, split.ptq_calibration_indices),
            batch_size=1, shuffle=False, num_workers=0,
        )
        development = DataLoader(
            Subset(dataset, split.stage2_validation_indices),
            batch_size=64, shuffle=False, num_workers=0,
        )
        assert len(ptq.dataset) == 500
        assert len(development.dataset) == 5000
        expected_labels = torch.tensor(
            [dataset.targets[i] for i in split.stage2_validation_indices],
            dtype=torch.long,
        )

        for index, job in enumerate(jobs, 1):
            cid = job["candidate_id"]
            name = job["configuration"]["name"]
            folder = (
                output_root / cid / job["selector"]
                / f"seed_{job['selection_seed']}" / name
            )
            folder.mkdir(parents=True, exist_ok=True)
            result_path = folder / "result.json"
            print(f"[{index}/{len(jobs)}] {folder.relative_to(output_root)}", flush=True)

            if result_path.exists():
                previous = json.loads(result_path.read_text())
                assert previous["job"] == job
                assert previous["protocol_sha256"] == audit["protocol_sha256"]
                assert sha(previous["tflite_path"]) == previous["tflite_sha256"]
                assert sha(previous["fitted_parameters_path"]) == previous[
                    "fitted_parameters_sha256"
                ]
                print("REUSED", flush=True)
                continue

            assert sha(job["development_export"]) == job["development_export_sha256"]
            with np.load(job["development_export"], allow_pickle=False) as data:
                fz = torch.from_numpy(data["fit_logits"].copy())
                fy = torch.from_numpy(data["fit_labels"].copy()).long()
                dz = torch.from_numpy(data["dev_logits"].copy())
                dy = torch.from_numpy(data["dev_labels"].copy()).long()
                rz = torch.from_numpy(data["reserve_logits"].copy())
                ry = torch.from_numpy(data["reserve_labels"].copy()).long()
            assert torch.equal(dy, expected_labels)
            assert fz.shape == (1500, 100) and rz.shape == (500, 100)
            assert dz.shape == (5000, 100)

            set_reproducibility_seed(42)
            matrix, bias, diagnostics = fit(job, protocol, fz, fy)
            baseline = metrics(dz, dy)
            posthoc = metrics(dz.double() @ matrix.T + bias, dy)
            reserve_before = metrics(rz, ry)
            reserve_after = metrics(rz.double() @ matrix.T + bias, ry)

            parameters_path = folder / "fitted_parameters.pt"
            torch.save({
                "matrix": matrix, "bias": bias,
                "diagnostics": diagnostics,
                "configuration": job["configuration"],
            }, parameters_path)

            artifacts = candidates[cid]["artifacts"]
            # Fold the affine logit transform into the final linear classifier.
            model = load_checkpoint(create_model(model_name), checkpoint)
            classifier = model.get_submodule(classifier_path)
            assert isinstance(classifier, torch.nn.Linear)
            assert classifier.out_features == 100
            images, _ = next(iter(ptq))

            with torch.no_grad():
                original_logits = model(images)
                a = matrix.to(classifier.weight.dtype)
                b = bias.to(classifier.weight.dtype)
                expected = original_logits @ a.T + b
                old_weight = classifier.weight.detach().clone()
                old_bias = (
                    classifier.bias.detach().clone()
                    if classifier.bias is not None
                    else old_weight.new_zeros(100)
                )
                classifier.weight.copy_(a @ old_weight)
                folded_bias = a @ old_bias + b
                if classifier.bias is None:
                    classifier.bias = torch.nn.Parameter(folded_bias)
                else:
                    classifier.bias.copy_(folded_bias)
                actual = model(images)
                maximum_difference = float((actual - expected).abs().max())
                assert torch.allclose(
                    actual, expected, atol=1e-5, rtol=1e-5,
                ), f"FP32 folding check failed: {maximum_difference}"

            exported = export_pt2e_and_tflite(
                model=model,
                calibration_dataloader=ptq,
                output_directory=folder,
                pt2e_filename="calibrated_model_pt2e.pt2",
                tflite_filename="calibrated_model_int8.tflite",
            )
            tflite = Path(exported["tflite_model_path"]).resolve()
            deployed_logits, deployed_labels = collect_tflite_logits(
                model_path=tflite, dataloader=development,
            )
            assert deployed_logits.shape == (5000, 100)
            assert torch.equal(deployed_labels.cpu().long(), expected_labels)
            deployed = metrics(deployed_logits, deployed_labels)

            drop_limit = protocol["accuracy_drop_limit"]
            result = {
                "job": job,
                "protocol_sha256": audit["protocol_sha256"],
                "official_test_used": False,
                "baseline_deployed": baseline,
                "posthoc": posthoc,
                "calibrated_deployed": deployed,
                "reserve_guard_before_export": (
                    reserve_before["top1"] - reserve_after["top1"]
                    <= drop_limit + 1e-7
                ),
                "deployed_development_guard": (
                    baseline["top1"] - deployed["top1"]
                    <= drop_limit + 1e-7
                ),
                "fp32_folding_max_absolute_difference": maximum_difference,
                "fold_verification_images": int(images.shape[0]),
                "diagnostics": diagnostics,
                "fitted_parameters_path": str(parameters_path),
                "fitted_parameters_sha256": sha(parameters_path),
                "source_tflite_sha256": sha(artifacts["tflite_model"]),
                "tflite_path": str(tflite),
                "tflite_sha256": sha(tflite),
                "baseline_tflite_bytes": Path(artifacts["tflite_model"]).stat().st_size,
                "calibrated_tflite_bytes": tflite.stat().st_size,
                "baseline_graph_operators": operators(artifacts["tflite_model"]),
                "calibrated_graph_operators": operators(tflite),
            }
            save(result_path, result)
            print(
                f"SAVED: HCE={100*deployed['hce']:.3f}% "
                f"accuracy={100*deployed['top1']:.3f}% "
                f"guard={result['deployed_development_guard']}",
                flush=True,
            )

        print(f"PASS: requested deployment jobs completed: {len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
