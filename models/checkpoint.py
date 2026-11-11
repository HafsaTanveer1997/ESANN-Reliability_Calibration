"""Load externally supplied model weights for calibration deployment."""

from pathlib import Path

import torch
from torch import nn


def load_checkpoint(model: nn.Module, checkpoint: str | Path) -> nn.Module:
    """Load a plain state dict into its matching architecture."""
    path = Path(checkpoint).expanduser()
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.cpu()
    model.load_state_dict(state_dict, strict=True)
    return model.eval()
