"""Model factory for the four CIFAR image-classification architectures."""

from __future__ import annotations

import torch

SUPPORTED_MODELS = (
    "mobilenet_v2_cifar100",
    "resnet32_cifar100",
    "resnet8",
    "vit_tiny_cifar10",
)


def create_model(model_name: str) -> torch.nn.Module:
    """Create the requested PyTorch CIFAR classifier."""

    if model_name == "mobilenet_v2_cifar100":
        from models.mobilenet_cifar100 import mobilenet_v2_cifar100
        return mobilenet_v2_cifar100()

    if model_name == "resnet32_cifar100":
        from models.resnet32_cifar100 import ResNet32
        return ResNet32(num_classes=100)

    if model_name == "resnet8":
        from models.resnet8 import resnet_v1_eembc
        return resnet_v1_eembc()

    if model_name == "vit_tiny_cifar10":
        from models.vit_cifar10 import vit_tiny_cifar10

        return vit_tiny_cifar10(
            image_size=32,
            patch_size=4,
            num_layers=4,
            num_heads=4,
            hidden_dim=128,
            mlp_dim=256,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=10,
        )

    raise ValueError(
        f"Unsupported model_name={model_name!r}. "
        f"Supported models: {SUPPORTED_MODELS}"
    )
