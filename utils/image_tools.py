import torch

def norm_minmax(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = torch.min(img)
    mx = torch.max(img)
    return (img - mn) / (mx - mn + eps)

def ensure_binary_mask(mask: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
    if mask.max() > 1.0:
        mask = mask / (mask.max() + 1e-6)
    return (mask > thresh).float()

def faultseg_augmentation(img: torch.Tensor, mask: torch.Tensor, p: float = 0.7):
    # lightweight augmentation (no extra deps)
    if torch.rand(1).item() > p:
        return img, mask

    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=[2])
        mask = torch.flip(mask, dims=[2])
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=[1])
        mask = torch.flip(mask, dims=[1])

    if torch.rand(1).item() < 0.5:
        k = int(torch.randint(0, 4, (1,)).item())
        img = torch.rot90(img, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[1, 2])

    if torch.rand(1).item() < 0.3:
        scale = 0.9 + 0.2 * torch.rand(1).item()
        shift = -0.05 + 0.1 * torch.rand(1).item()
        img = img * scale + shift
    return img, mask
