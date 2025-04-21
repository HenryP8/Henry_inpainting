import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import torch


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.device = 'cuda'
        self.vgg_layers = vgg16(weights=VGG16_Weights.DEFAULT).features.to(self.device)
        for param in self.vgg_layers.parameters():
            param.require_grad = False

    def forward(self, fake, real):
        loss = torch.zeros(fake.shape[0]).to(self.device)
        cnt = 0
        for _, module in self.vgg_layers._modules.items():
            fake = module(fake)
            real = module(real)

            if module.__class__.__name__ == 'ReLU':
                part_loss = F.mse_loss(fake, real, reduction='none')

                loss += part_loss.mean(dim=tuple(range(4)[1:]))
                cnt += 1

        return (loss/cnt).sum()