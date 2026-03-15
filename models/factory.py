from .segformer import SegFormer
from .maxvitunet import MaxViTUNet
from .deit3seg import DeiT3Seg
from .convnextunet import ConvNeXtUNet
from .unet import UNet
from .resnet50unet import ResNet50_UNet
from .transunet import TransUNet
from .transattunet import TransAttUNet
from .swinunet import SwinUNet
from .transdeeplab import TransDeepLab

def build_model(name: str, n_channels=1, n_classes=1, pretrained=True, **kwargs):
    name = name.lower()
    if name == "unet":
        return UNet(n_channels=n_channels, n_classes=n_classes, **kwargs)
    if name in ["resnet50_unet", "resnet50unet"]:
        return ResNet50_UNet(n_channels=n_channels, n_classes=n_classes, pretrained=pretrained)
    if name == "transunet":
        return TransUNet(n_channels=n_channels, n_classes=n_classes, pretrained=pretrained, **kwargs)
    if name in ["transattunet", "trans_attunet"]:
        return TransAttUNet(n_channels=n_channels, n_classes=n_classes, **kwargs)
    if name == "swinunet":
        return SwinUNet(n_channels=n_channels, n_classes=n_classes, pretrained=pretrained, **kwargs)
    if name in ["transdeeplab", "swindeeplab"]:
        return TransDeepLab(n_channels=n_channels, n_classes=n_classes, pretrained=pretrained, **kwargs)
    if name in ["maxvitunet", "maxvit_unet"]:
        return MaxViTUNet(n_channels=n_channels, n_classes=n_classes, pretrained=pretrained, **kwargs)
    if name in ["deit3", "deit3seg", "deit3_seg"]:
        return DeiT3Seg(n_channels=n_channels, n_classes=n_classes, pretrained=pretrained, **kwargs)
    if name in ["convnextunet", "convnext_unet"]:
        return ConvNeXtUNet(n_channels=n_channels, n_classes=n_classes, pretrained=pretrained, **kwargs)
    if name == "segformer":
        return SegFormer(n_channels=n_channels, n_classes=n_classes, pretrained=pretrained, **kwargs)


    raise ValueError(f"Unknown model name: {name}")
