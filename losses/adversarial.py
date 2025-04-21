import torch.nn.functional as F
import torch.nn as nn
import torch


class GeneratorAdversarialLoss(nn.Module):
    def __init__(self, discriminator):
        super(GeneratorAdversarialLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, fake, mask):
        disc_fake, _ = self.discriminator(fake)
        loss = F.softplus(-disc_fake)

        interp_mask = F.interpolate(1-mask, size=loss.shape[-2:], mode='bilinear', align_corners=False)
        loss *= interp_mask

        return loss.mean()


class DiscriminatorAdversarialLoss(nn.Module):
    def __init__(self, discriminator):
        super(DiscriminatorAdversarialLoss, self).__init__()
        self.discriminator = discriminator

    def forward(self, fake, real, mask):
        real.requires_grad = True

        disc_real, _ = self.discriminator(real)
        disc_fake, _ = self.discriminator(fake)

        if torch.is_grad_enabled():
            grad_real = torch.autograd.grad(outputs=disc_real.sum(), inputs=real, create_graph=True)[0]
            gradient_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
        else:
            gradient_penalty = 0
        real.requires_grad = False

        real_loss = F.softplus(-disc_real)
        fake_loss = F.softplus(disc_fake)

        interp_mask = F.interpolate(1-mask, size=fake_loss.shape[-2:], mode='bilinear', align_corners=False)
        fake_loss *= interp_mask

        return (real_loss + 0.001 *gradient_penalty + fake_loss).mean()
