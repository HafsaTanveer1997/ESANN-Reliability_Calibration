"""Fit, export, and validate frozen CIFAR-10 calibration configurations."""

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
from deployment.pt2e_export import export_pt2e_and_tflite
from deployment.reproducibility import set_reproducibility_seed
from deployment.tflite_runner import collect_tflite_logits
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from ai_edge_litert import schema_py_generated as schema

from calibration.anchored_components import fit_temperature, fit_ta_cgsrh
from calibration.metrics_adapter import evaluate

MODEL_NAMES = {
    "resnet8_trial_33": "resnet8",
    "vit_tiny_cifar10_trial_4": "vit_tiny_cifar10",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def operators(path):
    model = schema.Model.GetRootAsModel(Path(path).read_bytes(), 0)
    return sum(
        model.Subgraphs(i).OperatorsLength()
        for i in range(model.SubgraphsLength())
    )


def metrics(logits, labels):
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


def verify(root, name, digest, size=None):
    """Resolve an input under its declared root and verify its contents."""
    root = root.resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or root not in path.parents or not path.is_file():
        raise ValueError(f"Invalid or missing input: {path}")
    if (size is not None and path.stat().st_size != size) or sha(path) != digest:
        raise ValueError(f"Frozen input mismatch: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("resnet8", "vit_tiny"))
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--weights-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("development", "repeat"))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Run without Python optimization to retain deployment checks.")
    if not args.data_root.expanduser().is_dir():
        parser.error(f"CIFAR-10 data directory not found: {args.data_root}")

    input_root = args.input_root.expanduser().resolve()
    protocol = json.loads(args.settings.read_text())
    split_path = verify(input_root, protocol["split"], protocol["split_sha256"])
    manifest = args.artifact_root / "manifest.json"
    records = json.loads(manifest.read_text())["models"]
    heads = protocol["classifier_paths"]
    expected = {"resnet8_trial_33": "fc", "vit_tiny_cifar10_trial_4": "head"}
    if heads != expected:
        raise ValueError("Unexpected CIFAR-10 classifier mapping.")
    selected_candidate = {
        "resnet8": "resnet8_trial_33",
        "vit_tiny": "vit_tiny_cifar10_trial_4",
    }[args.model]
    candidates = {}
    for cid in (selected_candidate,):
        rows = [row for row in records if row["candidate_id"] == cid]
        if len(rows) != 1:
            raise ValueError(f"Expected one model bundle entry for {cid}.")
        row = rows[0]["artifacts"]["tflite_model"]
        baseline_tflite = verify(
            args.artifact_root, row["path"], row["sha256"], row["bytes"]
        )
        checkpoint = (
            args.weights_root.expanduser() / f"{MODEL_NAMES[cid]}.pt"
        ).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing model weights: {checkpoint}")
        candidates[cid] = {
            "tflite_model": baseline_tflite,
            "checkpoint": checkpoint,
            "checkpoint_sha256": sha(checkpoint),
        }

    jobs = [dict(row) for row in protocol[args.phase]]
    count = 9 if args.phase == "development" else 20
    if len(jobs) != 2 * count or any(
        sum(j["candidate_id"] == cid for j in jobs) != count for cid in heads
    ):
        raise ValueError(f"Unexpected number of {args.phase} jobs.")
    jobs = [job for job in jobs if job["candidate_id"] == selected_candidate]
    verified = {}
    for job in jobs:
        name = job["configuration"]["name"]
        if job["selector"] != "random" or not name or name in (".", "..") or Path(name).name != name:
            raise ValueError("Invalid selector or configuration name.")
        key = (job["development_export"], job["development_export_sha256"])
        if key not in verified:
            verified[key] = str(verify(input_root, *key))
        job["development_export"] = verified[key]

    print(f"PASS: {len(jobs)} CIFAR-10 {args.model} {args.phase} jobs verified.", flush=True)
    if not args.run:
        return

    torch.set_num_threads(1)
    ROOT = args.output_root.expanduser().resolve() / "cifar10" / args.phase / args.model
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = {
            "protocol_sha256": sha(args.settings),
            "runner_sha256": sha(Path(__file__).resolve()),
            "artifact_manifest_sha256": sha(manifest),
            "checkpoint_sha256": {
                cid: item["checkpoint_sha256"] for cid, item in candidates.items()
            },
        }
        audit_path = ROOT / "runner_audit.json"
        if audit_path.exists():
            assert json.loads(audit_path.read_text()) == audit, "Runner inputs changed."
        else:
            save(audit_path, audit)

        split = load_experiment_data_split(split_path)
        data_root = args.data_root.expanduser().resolve()
        if data_root.name == "cifar-10-batches-py":
            data_root = data_root.parent
        dataset = CIFAR10(
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
        assert len(development.dataset) == 2250
        expected_labels = torch.tensor(
            [dataset.targets[i] for i in split.stage2_validation_indices],
            dtype=torch.long,
        )

        for index, job in enumerate(jobs, 1):
            cid = job["candidate_id"]
            name = job["configuration"]["name"]
            folder = (
                ROOT / cid / job["selector"]
                / f"seed_{job['selection_seed']}" / name
            )
            folder.mkdir(parents=True, exist_ok=True)
            result_path = folder / "result.json"
            print(f"[{index}/{len(jobs)}] {folder.relative_to(ROOT)}", flush=True)

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
            assert fz.shape == (410, 10) and rz.shape == (102, 10)
            assert dz.shape == (2250, 10)

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

            artifacts = candidates[cid]
            classifier_path = heads[cid]
            model = load_checkpoint(create_model(MODEL_NAMES[cid]), artifacts["checkpoint"])
            classifier = model.get_submodule(classifier_path)
            assert isinstance(classifier, torch.nn.Linear)
            assert classifier.out_features == 10
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
                    else old_weight.new_zeros(10)
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
            assert deployed_logits.shape == (2250, 10)
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
