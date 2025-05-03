import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights
import torch


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.device = 'cuda'
        self.resnet_dilated = resnet50(weights=ResNet50_Weights.DEFAULT, 
                                       replace_stride_with_dilation=[False, True, True]).to(self.device)
        self.model = nn.Sequential(*list(self.resnet_dilated.children())[:-2])
        for param in self.model.parameters():
            param.require_grad = False

    def forward(self, fake, real, mask):
        loss = torch.zeros(fake.shape[0]).to(self.device)
        new_loss = 0
        c = 0
        
        for _, module in self.model._modules.items():
            fake = module(fake)
            real = module(real)

            if module.__class__.__name__ == 'ReLU':
                interp_mask = F.interpolate(mask, size=fake.shape[-2:], mode='bilinear', align_corners=False)
                loss += (F.mse_loss(fake, real, reduction='none') * interp_mask).mean()

                c += 1

        return loss / c
    

if __name__ == '__main__':
    model = PerceptualLoss()
