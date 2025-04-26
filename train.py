import torch
import numpy as np
from torchvision.datasets import Places365
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.generator import FFCGenerator
from models.discriminator import Discriminator
from dataloader.dataset import MaskedImgDataset
from losses.l1_loss import L1_Loss
from losses.adversarial import GeneratorAdversarialLoss, DiscriminatorAdversarialLoss
from losses.feature_matching import FeatureMatchingLoss
from losses.perceptual import PerceptualLoss

import os
import time


torch.manual_seed(0)

subset_data = MaskedImgDataset('data/places_subset_15000')
train_loader = DataLoader(subset_data, shuffle=True, batch_size=8)


def imsave(img, fn):
    img = 0.5 * img + 0.5
    npimg = img.numpy()
    plt.imsave(fn, np.transpose(npimg, (1, 2, 0)))


num_epochs = 25
warmup_epochs = 1

model_name = time.strftime("%Y%m%d-%H%M%S")
device = 'cuda'
generator = FFCGenerator().to(device)
discriminator = Discriminator().to(device)

optimizer_g = optim.AdamW(generator.parameters(), lr=1e-3)
optimizer_d = optim.AdamW(discriminator.parameters(), lr=1e-4)

# lr_warmup_g = optim.lr_scheduler.LinearLR(optimizer_g, start_factor=0.01, end_factor=1, total_iters=warmup_epochs)
# lr_cos_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=num_epochs-warmup_epochs, eta_min=1e-4)
# lr_scheduler_g = optim.lr_scheduler.SequentialLR(optimizer_g, schedulers=[lr_warmup_g, lr_cos_g], milestones=[warmup_epochs])
# lr_scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=5, gamma=0.5)

criterion_l1 = L1_Loss().to(device)
criterion_g = GeneratorAdversarialLoss(discriminator).to(device)
criterion_d = DiscriminatorAdversarialLoss(discriminator).to(device)
criterion_f = FeatureMatchingLoss(discriminator).to(device)
criterion_p = PerceptualLoss().to(device)

for epoch in range(num_epochs):
    for batch_idx, (images, reals, masks, _) in enumerate(tqdm(train_loader)):

        images, reals, masks = images.to(device), reals.to(device), masks.to(device)
        fakes = generator(images)

        loss_l1 = criterion_l1(fakes, reals, masks)

        loss_adv = criterion_g(fakes, masks)
        fakes = fakes.detach()
        loss_d = criterion_d(fakes, reals, masks)

        loss_feature_match = criterion_f(fakes, reals, masks)

        perceptual_loss = criterion_p(fakes, reals, masks)

        loss_g = loss_l1 + 10 * loss_adv + 100 * loss_feature_match + 30 * perceptual_loss

        if batch_idx % 1000 == 0 or batch_idx == len(train_loader) - 1:
            imsave(fakes.cpu().detach()[0], f'results/{epoch}_{batch_idx}.png')

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

    # lr_scheduler_g.step()
    # lr_scheduler_d.step()

    # print(f'Epoch: {epoch+1}\nG lr: {optimizer_g.param_groups[0]['lr']}\nD lr: {optimizer_d.param_groups[0]['lr']}')
    # print(f'Loss g: {loss_g.item()}\nLoss d: {loss_d.item()}')

    if epoch % 3 == 0:
        torch.save(generator.state_dict(), f'saved_models/generator/{model_name}_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'saved_models/discriminator/{model_name}_epoch_{epoch}.pth')
        torch.save(optimizer_g.state_dict(), f'saved_models/optimizer_g/{model_name}_epoch_{epoch}.pth')
        torch.save(optimizer_d.state_dict(), f'saved_models/optimizer_d/{model_name}_epoch_{epoch}.pth')

torch.save(generator.state_dict(), f'saved_models/generator/{model_name}_final.pth')
torch.save(discriminator.state_dict(), f'saved_models/discriminator/{model_name}_final.pth')
torch.save(optimizer_g.state_dict(), f'saved_models/optimizer_g/{model_name}_final.pth')
torch.save(optimizer_d.state_dict(), f'saved_models/optimizer_d/{model_name}_final.pth')