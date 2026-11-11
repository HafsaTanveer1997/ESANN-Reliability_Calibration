"""PT2E quantization and TFLite export helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import litert_torch
import torch
import torchao
from litert_torch.quantize.pt2e_quantizer import (
    PT2EQuantizer,
    get_symmetric_quantization_config,
)
from litert_torch.quantize.quant_config import QuantConfig
from torchao.quantization.pt2e.quantize_pt2e import (
    convert_pt2e,
    prepare_pt2e,
)


def quantize_model_pt2e(
    model: torch.nn.Module,
    calibration_dataloader,
) -> Tuple[torch.nn.Module, PT2EQuantizer, tuple]:
    """Quantize a float PyTorch model through the PT2E flow."""

    cpu = torch.device("cpu")
    model_cpu = model.to(cpu).eval()

    first_batch = next(iter(calibration_dataloader))
    single_input = (first_batch[0].to(cpu),)

    exported_program = torch.export.export(
        model_cpu,
        single_input,
    )
    pt2e_model = exported_program.module()

    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,
        )
    )

    pt2e_model = prepare_pt2e(
        pt2e_model,
        quantizer,
    )

    with torch.no_grad():
        for images, _ in calibration_dataloader:
            pt2e_model(images.to(cpu))

    torchao.quantization.pt2e.move_exported_model_to_eval(
        pt2e_model
    )

    pt2e_model = convert_pt2e(
        pt2e_model,
        fold_quantize=False,
    )

    torchao.quantization.pt2e.move_exported_model_to_eval(
        pt2e_model
    )

    return pt2e_model, quantizer, single_input


def export_pt2e_and_tflite(
    model: torch.nn.Module,
    calibration_dataloader,
    output_directory: str | Path,
    *,
    pt2e_filename: str = "calibrated_model_pt2e.pt2",
    tflite_filename: str = "calibrated_model.tflite",
) -> Dict[str, Any]:
    """Quantize and export a standalone PT2E and TFLite model.

    The exported TFLite file is read back in binary mode.  Returning the bytes
    here makes the Stage-2 trial flow identical to the Stage-1 deployment
    verification flow: export -> read bytes -> tflite_deployed_accuracy().
    """

    output_directory = Path(
        output_directory
    ).expanduser().resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pt2e_model, quantizer, single_input = (
        quantize_model_pt2e(
            model,
            calibration_dataloader,
        )
    )

    pt2e_path = output_directory / pt2e_filename

    quantized_exported_program = torch.export.export(
        pt2e_model,
        single_input,
    )

    torch.export.save(
        quantized_exported_program,
        str(pt2e_path),
    )

    converted_model = litert_torch.convert(
        pt2e_model,
        single_input,
        quant_config=QuantConfig(
            pt2e_quantizer=quantizer
        ),
    )

    tflite_path = (
        output_directory / tflite_filename
    )

    converted_model.export(
        str(tflite_path)
    )

    if not pt2e_path.is_file():
        raise RuntimeError(
            f"PT2E export was not created: {pt2e_path}"
        )

    if not tflite_path.is_file():
        raise RuntimeError(
            f"TFLite export was not created: {tflite_path}"
        )

    # This is the exact operation used in hyper_search_train.py.
    tflite_bytes = tflite_path.read_bytes()

    if not tflite_bytes:
        raise RuntimeError(
            f"TFLite export is empty: {tflite_path}"
        )

    return {
        "pt2e_model_path": pt2e_path,
        "tflite_model_path": tflite_path,
        "tflite_bytes": tflite_bytes,
        "tflite_size_bytes": len(tflite_bytes),
    }
