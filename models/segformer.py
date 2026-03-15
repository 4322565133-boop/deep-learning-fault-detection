import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise ImportError(
        "segmentation-models-pytorch is not installed. "
        "Please run: pip install -U segmentation-models-pytorch"
    ) from e


class SegFormer(nn.Module):
    """
    SegFormer wrapper based on segmentation_models_pytorch.

    This class matches your build_model() signature:
      SegFormer(n_channels=1, n_classes=1, pretrained=True, **kwargs)

    Recommended kwargs:
      encoder_name: "mit_b0" / "mit_b1" / ...
      encoder_weights: "imagenet" (default if pretrained=True) or None
    """

    def __init__(self, n_channels=1, n_classes=1, pretrained=True, **kwargs):
        super().__init__()

        # Use MiT encoder names from SMP
        encoder_name = kwargs.pop("encoder_name", "mit_b0")

        # If user explicitly provides encoder_weights, respect it
        # Otherwise follow pretrained flag
        if "encoder_weights" in kwargs:
            encoder_weights = kwargs.pop("encoder_weights")
        else:
            encoder_weights = "imagenet" if pretrained else None

        # activation=None => return logits (good for BCE/Dice loss in your pipeline)
        self.model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes,
            activation=None,
        )

        # If there are unexpected kwargs, raise early (helps catch YAML typos)
        if len(kwargs) > 0:
            raise ValueError(f"Unknown kwargs for SegFormer(SMP): {list(kwargs.keys())}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
