import torch
import torch.nn.functional as F

def resize(img: torch.Tensor, size: int):
    if img.dim() == 3:
        out = F.interpolate(img.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)
        return out.squeeze(0)
    if img.dim() == 4:
        return F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
    raise ValueError(f"Unexpected tensor dim: {img.dim()}")
