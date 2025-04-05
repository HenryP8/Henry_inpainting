import torch.nn.functional as F
import torch.nn as nn
import torch

from models.generator import FFCGenerator
from models.discriminator import Discriminator


class FeatureMatchingLoss(nn.Module):
    def __init__(self, discriminator):
        super(FeatureMatchingLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, fake, real, mask):
        _, real_features = self.discriminator(real)
        _, fake_features = self.discriminator(fake)

        res = 0
        norm = 0
        for fake_feat, target_feat in zip(fake_features, real_features):
            cur_mask = F.interpolate(mask, size=fake_feat.shape[-2:], mode='bilinear', align_corners=False)
            error_weights = 1 - cur_mask
            cur_val = ((fake_feat - target_feat).pow(2) * error_weights).mean()
            res = res + cur_val
            norm += 1
        res = res / norm
        return res
    