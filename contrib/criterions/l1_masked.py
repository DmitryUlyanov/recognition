import torch.nn as nn


class L1Masked(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask=None):
        if mask == None:
            eps = 1e-9
            mask = target.sum(1, True) > eps
            mask = mask.float()

        return nn.functional.l1_loss(input*mask, target*mask)
