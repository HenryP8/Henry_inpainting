import torch.nn.functional as F
import torch.nn as nn


class L1_Loss(nn.Module):
    def __init__(self):
        super(L1_Loss, self).__init__()

    def forward(self, pred, target, mask):
        loss = F.l1_loss(pred, target, reduction='none')
        mask_weights = mask * 10
        return (loss * mask_weights).mean()
