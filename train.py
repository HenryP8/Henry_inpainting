import torch
import numpy as np
import cv2
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

import os
import time


torch.manual_seed(0)

all_data = MaskedImgDataset('data/data_256_standard')
train_data, test_data, _ = random_split(all_data, [50_000, 15_000, len(all_data) - 65_000])
train_loader = DataLoader(train_data, shuffle=True)
test_loader = DataLoader(test_data, shuffle=True)

subset_data = MaskedImgDataset('data/places_subset_15000')
train_loader = DataLoader(subset_data, shuffle=True)

def imshow(img):
    img = 0.5 * img + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')


def imsave(img, fn):
    img = 0.5 * img + 0.5
    npimg = img.numpy()
    plt.imsave(fn, np.transpose(npimg, (1, 2, 0)))


device= 'cuda'
generator = FFCGenerator().to(device)
discriminator = Discriminator().to(device)

optimizer_g = optim.AdamW(generator.parameters(), lr=0.001)
optimizer_d = optim.AdamW(discriminator.parameters(), lr=0.001)

criterion_l1 = L1_Loss().to(device)
criterion_g = GeneratorAdversarialLoss(discriminator).to(device)
criterion_d = DiscriminatorAdversarialLoss(discriminator).to(device)
criterion_f = FeatureMatchingLoss(discriminator).to(device)

num_epochs = 1

for epoch in range(num_epochs):
    for batch_idx, (images, reals, masks, _) in enumerate(tqdm(train_loader)):

        images, reals, masks = images.to(device), reals.to(device), masks.to(device)
        fakes = generator(images)
        
        loss_l1 = criterion_l1(fakes, reals, masks)

        loss_adv = criterion_g(fakes, masks)
        fakes = fakes.detach()
        loss_d = criterion_d(fakes, reals, masks)

        loss_feature_match = criterion_f(fakes, reals, masks)

        loss_g = loss_l1 + 10 * loss_adv + 100 * loss_feature_match

        if batch_idx % 1000 == 0 or batch_idx == len(train_loader) - 1:
            imsave(fakes.cpu().detach()[0], f'results/{epoch}_{batch_idx}.png')

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

model_name = time.strftime("%Y%m%d-%H%M%S")
torch.save(generator.state_dict(), f'saved_models/{model_name}')

model = FFCGenerator()
model.load_state_dict(torch.load(f'saved_models/{model_name}', weights_only=True))
model.eval()
model = model.to(device)

for batch_idx, (input_images, targets, _, images) in enumerate(test_loader):
    for target, img in zip(targets, images):
        imshow(img)
        plt.show()
        pred = model(input_images.to(device))
        imshow(pred.cpu().detach()[0])
        plt.show()
        imshow(target)
        plt.show()
