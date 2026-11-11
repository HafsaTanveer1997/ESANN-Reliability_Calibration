"""LiteRT/TFLite inference utilities for collecting classification logits."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter


class TFLiteLogitRunner:
    """Run a single-input, single-output TFLite classification model."""

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:
        model_path = Path(
            model_path
        ).expanduser().resolve()

        if not model_path.is_file():
            raise FileNotFoundError(
                f"TFLite model does not exist: {model_path}"
            )

        self.model_path = model_path

        self.interpreter = Interpreter(
            model_path=str(model_path)
        )

        self.interpreter.allocate_tensors()

        input_details = (
            self.interpreter.get_input_details()
        )

        output_details = (
            self.interpreter.get_output_details()
        )

        if len(input_details) != 1:
            raise ValueError(
                "Expected exactly one TFLite input tensor."
            )

        if len(output_details) != 1:
            raise ValueError(
                "Expected exactly one TFLite output tensor."
            )

        self.input_detail = input_details[0]
        self.output_detail = output_details[0]

        self.input_index = int(
            self.input_detail["index"]
        )

        self.output_index = int(
            self.output_detail["index"]
        )

        self.input_dtype = np.dtype(
            self.input_detail["dtype"]
        )

        self.output_dtype = np.dtype(
            self.output_detail["dtype"]
        )

        self.input_scale, self.input_zero_point = (
            self.input_detail["quantization"]
        )

        self.output_scale, self.output_zero_point = (
            self.output_detail["quantization"]
        )

    def _prepare_input(
        self,
        image: torch.Tensor,
    ) -> np.ndarray:
        """Convert one floating-point PyTorch image to model input format."""

        if not isinstance(image, torch.Tensor):
            raise TypeError(
                "image must be a torch.Tensor."
            )

        image_array = (
            image.detach()
            .cpu()
            .numpy()
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        if np.issubdtype(
            self.input_dtype,
            np.integer,
        ):
            if self.input_scale == 0:
                raise ValueError(
                    "Integer TFLite input has zero quantization scale."
                )

            quantized = np.rint(
                image_array / self.input_scale
                + self.input_zero_point
            )

            dtype_limits = np.iinfo(
                self.input_dtype
            )

            image_array = np.clip(
                quantized,
                dtype_limits.min,
                dtype_limits.max,
            ).astype(
                self.input_dtype
            )
        else:
            image_array = image_array.astype(
                self.input_dtype
            )

        return image_array

    def _dequantize_output(
        self,
        output: np.ndarray,
    ) -> np.ndarray:
        """Convert a TFLite output tensor to floating-point logits."""

        if np.issubdtype(
            self.output_dtype,
            np.integer,
        ):
            if self.output_scale == 0:
                raise ValueError(
                    "Integer TFLite output has zero quantization scale."
                )

            output = (
                output.astype(np.float32)
                - self.output_zero_point
            ) * self.output_scale
        else:
            output = output.astype(
                np.float32
            )

        return output

    def run_image(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Run inference for one image and return one-dimensional logits."""

        prepared_input = self._prepare_input(
            image
        )

        self.interpreter.set_tensor(
            self.input_index,
            prepared_input,
        )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_index
        )

        output = self._dequantize_output(
            output
        )

        output = np.asarray(
            output
        ).reshape(-1)

        return torch.from_numpy(
            output.copy()
        ).to(
            dtype=torch.float32
        )


def collect_tflite_logits(
    model_path: str | Path,
    dataloader: torch.utils.data.DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect dequantized TFLite logits and labels from a DataLoader."""

    runner = TFLiteLogitRunner(
        model_path=model_path
    )

    collected_logits = []
    collected_labels = []

    for images, labels in dataloader:
        if images.ndim < 2:
            raise ValueError(
                "Expected a batch of input images."
            )

        if labels.ndim != 1:
            labels = labels.reshape(-1)

        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                "Image and label batch sizes do not match."
            )

        for image, label in zip(
            images,
            labels,
        ):
            logits = runner.run_image(
                image
            )

            collected_logits.append(
                logits
            )

            collected_labels.append(
                int(label.item())
            )

    if not collected_logits:
        raise ValueError(
            "The DataLoader did not provide any samples."
        )

    logits_tensor = torch.stack(
        collected_logits,
        dim=0,
    )

    labels_tensor = torch.tensor(
        collected_labels,
        dtype=torch.long,
    )

    return logits_tensor, labels_tensor
