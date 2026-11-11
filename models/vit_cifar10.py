"""Local PyTorch Vision Transformer implementation.

Reference source:
https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py

This file follows the Torchvision VisionTransformer structure and adapts it for
the MLPerf Tiny PyTorch flow and CIFAR-10 experiments.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convert input image into patch embeddings using Conv2d projection."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.conv_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_proj(x)      # [B, hidden_dim, H/P, W/P]
        x = x.flatten(2)           # [B, hidden_dim, num_patches]
        x = x.transpose(1, 2)      # [B, num_patches, hidden_dim]
        return x


class MLPBlock(nn.Module):
    """Transformer MLP block."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    ("linear_1", nn.Linear(hidden_dim, mlp_dim)),
                    ("activation", nn.GELU()),
                    ("dropout_1", nn.Dropout(dropout)),
                    ("linear_2", nn.Linear(mlp_dim, hidden_dim)),
                    ("dropout_2", nn.Dropout(dropout)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EncoderBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBlock(
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln_1(x)
        y, _ = self.self_attention(y, y, y, need_weights=False)
        x = x + self.dropout(y)

        y = self.ln_2(x)
        y = self.mlp(y)
        x = x + y

        return x


class Encoder(nn.Module):
    """Transformer encoder stack."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()

        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
        )
        self.dropout = nn.Dropout(dropout)

        layers = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
            )

        self.layers = nn.Sequential(layers)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)
        return x


class VisionTransformer(nn.Module):
    """Small local Vision Transformer for CIFAR-10."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 128,
        mlp_dim: int = 256,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 10,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
        )

        seq_length = self.patch_embed.num_patches + 1

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.encoder = Encoder(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        self.head = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.class_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)

        x = torch.cat([class_token, x], dim=1)
        x = self.encoder(x)

        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vit_tiny_cifar10(
    image_size: int = 32,
    patch_size: int = 4,
    num_layers: int = 4,
    num_heads: int = 4,
    hidden_dim: int = 128,
    mlp_dim: int = 256,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
    num_classes: int = 10,
) -> VisionTransformer:
    """Create a small local ViT model for CIFAR-10."""
    return VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        dropout=dropout,
        attention_dropout=attention_dropout,
        num_classes=num_classes,
    )
