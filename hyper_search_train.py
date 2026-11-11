"""Optuna search for PyTorch CIFAR-10 image-classification models.

The Stage-1 search trains each candidate from scratch on the saved training
split, selects its best validation weights, quantizes it with PT2E, and
optimizes deployed INT8 accuracy together with compressed model size.
"""

import argparse
import copy
import logging
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

# Suppress warnings from litert
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
import litert_torch
import torchao
from litert_torch.quantize.pt2e_quantizer import (
    PT2EQuantizer,
    get_symmetric_quantization_config,
)
from litert_torch.quantize.quant_config import QuantConfig
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from util.optuna_search import (
    OptunaTuner,
    add_tuning_args,
    load_ssd,
)
from util.tflite_helpers import compressed_size, tflite_evaluate
from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter
from search_data.cifar10_data import create_cifar10_data_bundle
from utils.training import LOSS_FUNCTION
from util_torch.training_helpers_torch import evaluation, train_one_epoch


class _TfliteDatasetAdapter:
    """Wraps a PyTorch DataLoader to the .shapes / .tflite_gen() interface
    required by tflite_evaluate in tflite_helpers.

    Input images (float32 [0,1]) are quantized to INT8 using the tflite
    model's input tensor quantization parameters before being fed to the
    interpreter.  Labels are wrapped in a one-element list to match the
    single-output convention expected by the loss function.
    """
    def __init__(self, dataloader, tflite_bytes):
        interp = LiteRTInterpreter(model_content=tflite_bytes)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        self._in_scale, self._in_zp = inp['quantization']
        self._in_shape = list(inp['shape'])  # e.g. [1, 3, 32, 32]
        out = interp.get_output_details()[0]
        self._out_scale, self._out_zp = out['quantization']
        self._dataloader = dataloader
        # shapes: ([input_shape], [output_shape]) — resize_tensor_input uses input shapes
        self.shapes = ([self._in_shape], [])

    def tflite_gen(self):
        for images, labels in self._dataloader:
            images_np = images.numpy()
            labels_np = labels.numpy()
            for img, label in zip(images_np, labels_np):
                if self._in_scale != 0:
                    img_q = (img / self._in_scale + self._in_zp).clip(-128, 127).astype(np.int8)
                else:
                    img_q = img.astype(np.int8)
                # x: list indexed by input tensor index; y: label as array
                yield ([img_q[np.newaxis]], [np.array([label], dtype=np.int64)])


def tflite_deployed_accuracy(tflite_bytes, evaluation_dataset):
    """Return categorical accuracy for a fully quantized LiteRT model.

    The LiteRT model exposes quantized INT8 outputs, so they are dequantized
    before computing argmax predictions against the sparse integer labels.
    """
    tf_dataset = _TfliteDatasetAdapter(
        evaluation_dataset,
        tflite_bytes,
    )

    def acc_fn(y_true, y_pred):
        out = y_pred[0]
        # Dequantize INT8 output (litert_torch model has fully quantized I/O)
        if tf_dataset._out_scale != 0:
            out = (out.astype(np.float32) - tf_dataset._out_zp) * tf_dataset._out_scale
        predicted = np.argmax(out, axis=-1).flatten()
        true_labels = y_true[0].flatten()
        return float(np.mean(predicted == true_labels))

    return tflite_evaluate(tflite_bytes, acc_fn, tf_dataset.tflite_gen())


class TrainHistory:
    """Minimal drop-in for Keras History, required by OptunaTuner.objective.

    OptunaTuner looks up ``train_hist.history["val_<metric>"][-1]`` to log
    per-epoch validation curves.  We accumulate them here during training.
    """

    def __init__(self):
        self.history: dict[str, list] = {}

    def log(self, key: str, value: float) -> None:
        self.history.setdefault(key, []).append(float(value))


_CLEAN_CIFAR10_BUNDLE = None
_CLEAN_CIFAR10_BUNDLE_KEY = None



_CLEAN_CIFAR100_CACHE = {}

def get_clean_cifar100_bundle(ssd, flags):
    """Load CIFAR-100 using its separate experiment split."""
    import json
    from search_data.cifar100_data import (
        create_cifar100_data_bundle,
    )

    split_path = Path(flags.split_path)
    payload = json.loads(split_path.read_text())
    if payload.get("dataset_name") != "CIFAR100":
        raise ValueError("ResNet-32 requires a CIFAR100 split file.")

    key = (
        str(Path(flags.cifar100_path).resolve()),
        str(split_path.resolve()),
        int(ssd["fixed_params"]["batch_size"]),
        flags.evaluation_batch_size,
        flags.ptq_batch_size,
        flags.num_workers,
    )
    if key not in _CLEAN_CIFAR100_CACHE:
        _CLEAN_CIFAR100_CACHE[key] = create_cifar100_data_bundle(
            cifar_100_dir=flags.cifar100_path,
            split_path=split_path,
            training_batch_size=key[2],
            evaluation_batch_size=flags.evaluation_batch_size,
            ptq_batch_size=flags.ptq_batch_size,
            num_workers=flags.num_workers,
            allow_download=False,
        )
    return _CLEAN_CIFAR100_CACHE[key]


def prepare_cifar100_datasets(ssd, flags):
    """Use Stage-1 validation for both search evaluation slots."""
    bundle = get_clean_cifar100_bundle(ssd, flags)
    training = bundle.experiment.training
    validation = bundle.experiment.stage1_validation
    print(
        f"CIFAR-100: training={len(training.dataset)}, "
        f"validation={len(validation.dataset)}, "
        f"PTQ={len(bundle.experiment.ptq_calibration.dataset)}"
    )
    return training, validation, validation

def get_clean_cifar10_bundle(ssd, flags):
    """Create or reuse the clean two-stage CIFAR-10 data bundle."""

    if flags.model_name in CIFAR100_MODELS:
        return get_clean_cifar100_bundle(ssd, flags)

    global _CLEAN_CIFAR10_BUNDLE
    global _CLEAN_CIFAR10_BUNDLE_KEY

    training_batch_size = int(
        ssd["fixed_params"]["batch_size"]
    )

    evaluation_batch_size = int(
        flags.evaluation_batch_size
    )

    ptq_batch_size = int(
        flags.ptq_batch_size
    )

    num_workers = int(
        flags.num_workers
    )

    bundle_key = (
        str(Path(flags.cifar10_path).expanduser().resolve()),
        str(Path(flags.split_path).expanduser().resolve()),
        training_batch_size,
        evaluation_batch_size,
        ptq_batch_size,
        num_workers,
    )

    if (
        _CLEAN_CIFAR10_BUNDLE is None
        or _CLEAN_CIFAR10_BUNDLE_KEY != bundle_key
    ):
        _CLEAN_CIFAR10_BUNDLE = create_cifar10_data_bundle(
            cifar_10_dir=flags.cifar10_path,
            split_path=flags.split_path,
            training_batch_size=training_batch_size,
            evaluation_batch_size=evaluation_batch_size,
            ptq_batch_size=ptq_batch_size,
            num_workers=num_workers,
            allow_download=False,
        )

        _CLEAN_CIFAR10_BUNDLE_KEY = bundle_key

    return _CLEAN_CIFAR10_BUNDLE


CIFAR100_MODELS = (
    "resnet32_cifar100",
    "mobilenet_v2_cifar100",
)

SUPPORTED_MODELS = (
    "mobilenet_v2_cifar100",
    "resnet32_cifar100",
    "resnet8",
    "vit_tiny_cifar10",
)


def create_model(model_name: str) -> torch.nn.Module:
    """Create an image classifier through the shared model factory."""
    from models import create_model as build_model
    return build_model(model_name)


def pt_quantize(model: torch.nn.Module, cal_dataloader):
    """PT2E export -> prepare -> calibrate -> convert.
    """
    cpu = torch.device("cpu")
    model_cpu = model.to(cpu).eval()

    single_input = (next(iter(cal_dataloader))[0].to(cpu),)

    ep = torch.export.export(model_cpu, single_input)
    pt2e_model = ep.module()

    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    )
    pt2e_model = prepare_pt2e(pt2e_model, quantizer)
    with torch.no_grad():
        for images, _ in cal_dataloader:
            pt2e_model(images.to(cpu))

    torchao.quantization.pt2e.move_exported_model_to_eval(pt2e_model)
    pt2e_model = convert_pt2e(pt2e_model, fold_quantize=False)
    torchao.quantization.pt2e.move_exported_model_to_eval(pt2e_model)

    return pt2e_model, quantizer, single_input


# ---------------------------------------------------------------------------
# Trial function
# ---------------------------------------------------------------------------
def train_and_test(tuner, trial, flags, model_params):

    if flags.model_name in CIFAR100_MODELS:
        import random
        seed = int(tuner.ssd.get("training_seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        trial.set_user_attr("training_seed", seed)

    ckpt_dir = f"{tuner.files_root}/ckpts/{trial.number}/"
    ckpt_path = ckpt_dir + "{epoch}.pt"
    if flags.save_ckpt:
        os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(flags.model_name).to(device)
    if flags.model_name == "resnet8":
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    if tuner.files_root is not None:
        per_trial_dir = os.path.join(tuner.files_root, f"{trial.number:04d}")
        os.makedirs(per_trial_dir, exist_ok=True)
    else:
        per_trial_dir = None

    optimizer = Adam(
        params=model.parameters(),
        lr=model_params["learning_rate"],
        weight_decay=0.0,
        decoupled_weight_decay=False,
    )
    # These affect the train loop not the model definition
    regularizer = model_params.get("regularizer")
    l1 = model_params.get("l1", 0.0)
    l2 = model_params.get("l2", 0.0)

    # Create lr_scheduler with model_params learning rate
    decay_per_epoch = 0.99

    def lr_schedule(epoch):
        return decay_per_epoch ** epoch

    # LambdaLR expects a multiplicative factor
    train_scheduler = LambdaLR(
        optimizer=optimizer, lr_lambda=lr_schedule
    )

    if flags.from_ckpt:
        logging.info(f'Loading weights from checkpoint {flags.from_ckpt}')
        print(f'Loading weights from checkpoint {flags.from_ckpt}')
        model = torch.load(flags.from_ckpt, map_location=device)
        train_hist = None
    else:
        logging.info('Training from scratch.')
        print('Training from scratch.')

        train_hist = TrainHistory()
        best_val_accuracy = 0.0
        best_state_dict = None

        for epoch in range(1, tuner.ssd["epochs"] + 1):
            model.train()
            train_one_epoch(
                model=model,
                train_dataloader=tuner.ds_train,
                optimizer=optimizer,
                loss_fn=LOSS_FUNCTION,
                epoch=epoch,
                writer=None,
                device=device,
                regularizer=regularizer,
                l1=l1,
                l2=l2,
            )

            model.eval()
            val_accuracy, val_loss, val_precision, val_recall = evaluation(
                model=model,
                dataloader=tuner.ds_val,
                loss_fn=LOSS_FUNCTION,
                device=device,
                num_classes=100 if flags.model_name in CIFAR100_MODELS else 10,
            )
            train_hist.log("val_categorical_accuracy", val_accuracy)
            train_hist.log("val_loss", val_loss)
            train_hist.log("val_precision", val_precision)
            train_hist.log("val_recall", val_recall)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_state_dict = copy.deepcopy(model.state_dict())
                if flags.save_ckpt:
                    torch.save(
                        model,
                        ckpt_path.format(epoch=epoch),
                    )

            train_scheduler.step()

        logging.info('Training complete.')

        # Restore best validation weights (always tracked in memory)
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        if per_trial_dir is not None:
            pt_model_path = os.path.join(per_trial_dir, "ic_model.pt")
            torch.save(model, pt_model_path)

    print("Measuring performance on Stage-1 validation set...")
    model.eval()
    (
        validation_accuracy,
        validation_loss,
        validation_precision,
        validation_recall,
    ) = evaluation(
        model=model,
        dataloader=tuner.ds_test,
        loss_fn=LOSS_FUNCTION,
        device=device,
        num_classes=100 if flags.model_name in CIFAR100_MODELS else 10,
    )

    test_results = {
        "loss": float(validation_loss),
        "categorical_accuracy": float(validation_accuracy),
        "precision": float(validation_precision),
        "recall": float(validation_recall),
    }

    print("Quantizing model for compressed size estimate...")
    clean_data_bundle = get_clean_cifar10_bundle(
        tuner.ssd,
        flags,
    )
    cal_dataloader = (
        clean_data_bundle.experiment.ptq_calibration
    )
    pt2e_model, quantizer, single_input = pt_quantize(model, cal_dataloader)
    aiet_model = litert_torch.convert(
        pt2e_model,
        single_input,
        quant_config=QuantConfig(pt2e_quantizer=quantizer),
    )

    if per_trial_dir is not None:
        pt2e_model_path = os.path.join(per_trial_dir, "ic_model_pt2e.pt2")
        torch.export.save(torch.export.export(pt2e_model, single_input), pt2e_model_path)

        tflite_model_path = os.path.join(per_trial_dir, "ic_model.tflite")
        aiet_model.export(tflite_model_path)
        with open(tflite_model_path, "rb") as model_file:
            tflite_bytes = model_file.read()
    else:
        with tempfile.NamedTemporaryFile(
            suffix=".tflite",
            delete=False,
        ) as temporary_file:
            tmp_path = temporary_file.name
        try:
            aiet_model.export(tmp_path)
            with open(tmp_path, "rb") as model_file:
                tflite_bytes = model_file.read()
        finally:
            os.remove(tmp_path)

    print(
        "Evaluating deployed (INT8 LiteRT) model on "
        "Stage-1 validation..."
    )
    test_results["deployed_accuracy"] = tflite_deployed_accuracy(
        tflite_bytes,
        tuner.ds_test,
    )

    print(
        f"Accuracy: {test_results['categorical_accuracy']:.4f}  "
        f"Deployed accuracy: {test_results['deployed_accuracy']:.4f}"
    )

    print("Derive compressibility of INT8 LiteRT model...")
    packings = compressed_size(bytearray(tflite_bytes))
    return (packings, train_hist, test_results)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_cifar10_datasets(ssd, flags):
    """Prepare clean Stage-1 CIFAR-10 loaders.

    The third returned loader occupies OptunaTuner's historical ds_test
    slot, but it is the Stage-1 validation loader. The official CIFAR-10
    test set is not used during Stage 1.
    """

    if flags.model_name in CIFAR100_MODELS:
        return prepare_cifar100_datasets(ssd, flags)

    print("Preparing clean CIFAR-10 Stage-1 DataLoaders")
    print(20 * "-")

    bundle = get_clean_cifar10_bundle(
        ssd,
        flags,
    )

    train_loader = bundle.experiment.training
    validation_loader = bundle.experiment.stage1_validation
    ptq_loader = bundle.experiment.ptq_calibration

    print(
        f"Dataset sizes — "
        f"training: {len(train_loader.dataset)}, "
        f"Stage-1 validation: {len(validation_loader.dataset)}, "
        f"PTQ calibration: {len(ptq_loader.dataset)}"
    )

    # OptunaTuner requires three loaders. Its third slot is used for the
    # final per-trial objective evaluation. We intentionally reuse the
    # Stage-1 validation loader here.
    return (
        train_loader,
        validation_loader,
        validation_loader,
    )
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_hp(parser):
    add_tuning_args(parser)
    flags, _ = parser.parse_known_args()
    if not flags.split_path:
        parser.error("--split_path is required for Stage 1 searches.")
    if flags.model_name in CIFAR100_MODELS and not flags.cifar100_path:
        parser.error("--cifar100_path is required for CIFAR-100 models.")

    ssd = load_ssd(flags.ssd)
    # IFX requires this key for validation. Stage 1 uses PTQ.
    ssd.setdefault("qat_bits", None)
    # The tuner reads the trial limit from the YAML key 'trials'.
    if flags.hyperparam_trials is not None:
        ssd['trials'] = flags.hyperparam_trials

    tuner = OptunaTuner(
        ssd=ssd,
        flags=flags,
        do_trial=train_and_test,
        callback_functions={
            # rg_compressed_size is the estimated GR-compressed INT8 TFLite size.
            "rg_compressed_size": lambda model_info, _hist, _results: model_info["rg_size"],
            "zs_compressed_size": lambda model_info, _hist, _results: model_info["zs_size"],
            "prepare_cifar10_datasets": prepare_cifar10_datasets,
            "prepare_cifar100_datasets": prepare_cifar100_datasets,
        },
        verbose=True,
    )

    tuner.run_search_space()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for PyTorch CIFAR-10 and CIFAR-100 classifiers"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=SUPPORTED_MODELS,
        default="resnet8",
        help="Model architecture to train and quantize.",
    )

    parser.add_argument(
        "--split_path",
        type=str,
        help="Path to the saved experiment split JSON for the selected dataset.",
    )

    parser.add_argument(
        "--evaluation_batch_size",
        type=int,
        default=64,
        help="Batch size for Stage-1 validation.",
    )

    parser.add_argument(
        "--ptq_batch_size",
        type=int,
        default=1,
        help="Batch size used during PT2E calibration.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )

    parser.add_argument(
        '--cifar10_path',
        type=str,
        default="cifar-10-batches-py",
        help="Path to CIFAR-10 data directory.",
    )
    parser.add_argument(
        '--from_ckpt',
        type=str,
        help='Path to checkpoint to load weights from, if not training from scratch.',
        default=None,
        required=False,
    )
    parser.add_argument(
        '--save_ckpt',
        action='store_true',
        help='Save best checkpoint during training.',
        default=False,
        required=False,
    )
    parser.add_argument(
        "--cifar100_path",
        type=str,
        help="Root containing cifar-100-python. Required for CIFAR-100 models.",
    )
    run_hp(parser)
