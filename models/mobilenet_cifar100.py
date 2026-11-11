"""Torchvision MobileNetV2 adapted for 32-by-32 CIFAR-100 images."""

from torchvision.models import mobilenet_v2


def mobilenet_v2_cifar100():
    """Create MobileNetV2 with a CIFAR stem and 100-class classifier."""
    model = mobilenet_v2(
        weights=None,
        num_classes=100,
        width_mult=1.0,
        dropout=0.2,
    )
    # Preserve spatial resolution at the input for small CIFAR images.
    model.features[0][0].stride = (1, 1)
    return model
