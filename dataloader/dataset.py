import glob
import os

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

from dataloader.masks import gen_mask


class MaskedImgDataset(Dataset):
    def __init__(self, path):
        self.img_paths = list(glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=0.5, std=0.5)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        target = cv2.imread(self.img_paths[idx])
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        mask = gen_mask(256, 256, num_masks='random')
        masked_img = (target * mask).astype(np.uint8)

        masked_img_4d = torch.Tensor(
            torch.cat([
                torch.Tensor(masked_img), 
                torch.Tensor(1-mask)], 
                dim=-1))

        masked_img_4d = self.transform(masked_img_4d.numpy())
        masked_img_4d[3, :, :] = (masked_img_4d[3, :, :] + 1) // 2
        target = self.transform(target)
        masked_img = self.transform(masked_img)

        return masked_img_4d, target, (1-mask).reshape(1, 256, 256), masked_img
    

if __name__ == '__main__':
    subset_data = MaskedImgDataset('data/places_subset_15000')
    loader = DataLoader(subset_data, shuffle=True)

    for _, (masked_img, _, mask, _) in enumerate(loader):
        print(mask, mask.min(), mask.max(), np.unique(mask))
        print((1-mask), (1-mask).min(), (1-mask).max(), np.unique((1-mask)))
        # print(masked_img[0, 3, :, :], masked_img[0, 3, :, :].min(), masked_img[0, 3, :, :].max())
        # print(np.unique(masked_img[0, 3, :, :]))
        break
