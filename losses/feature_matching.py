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

        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            interp_mask = F.interpolate(1-mask, size=fake_feat.shape[-2:], mode='bilinear', align_corners=False)
            mask_weights = 1 - interp_mask
            loss += (F.mse_loss(real_feat, fake_feat, reduction='none') * mask_weights).mean()
            
        return loss / len(fake_features)
    