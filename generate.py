import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from models.generator import FFCGenerator
from dataloader.dataset import MaskedImgDataset


subset_data = MaskedImgDataset('data/val_256', num_masks=1, min_mask_l=100, max_mask_l=100)
test_loader = DataLoader(subset_data, shuffle=True)

def imshow(img, target, masked):
    img = 0.5 * img + 0.5
    target = 0.5 * target + 0.5
    masked = 0.5 * masked + 0.5
    npimg = img.numpy()
    nptgt = target.numpy()
    npmask = masked.numpy()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))

    axes[0].imshow(np.transpose(nptgt, (1, 2, 0)))
    axes[0].set_title('Ground Truth')
    axes[1].imshow(np.transpose(npmask, (1, 2, 0)))
    axes[1].set_title('Masked Input')
    axes[2].imshow(np.transpose(npimg, (1, 2, 0)))
    axes[2].set_title('Generated Image')
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')


device = 'cuda'
model_name = 'saved_models/generator/20250426-204017_epoch_15.pth'

model = FFCGenerator()
model.load_state_dict(torch.load(f'{model_name}'))
model = model.to(device)
model.eval()

for batch_idx, (images, reals, masks, masked) in enumerate(test_loader):

    images = images.to(device)
    fakes = model(images)

    imshow(fakes.cpu().detach()[0], reals[0], masked[0])
    plt.show()
