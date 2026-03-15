import torch.nn as nn
import torch.nn.functional as F
from .dice import DiceLoss

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = logits.sigmoid()
        dice = self.dice(probs, targets)
        return self.bce_weight * bce + self.dice_weight * dice
