import glob
import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch

from dataloader.masks import gen_mask


class MaskedImgDataset(Dataset):
    def __init__(self, path):
        self.img_paths = list(glob.glob(os.path.join(path, '*', '*', '*.jpg')))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=0.5, std=0.5)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        target = cv2.imread(self.img_paths[idx])
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        mask = gen_mask(256, 256)
        masked_img = (target * mask).astype(np.uint8)

        masked_img_4d = torch.Tensor(
            torch.cat([
                torch.Tensor(masked_img), 
                torch.zeros((256, 256, 1))], 
                dim=-1))
        
        masked_img_4d = self.transform(masked_img_4d.numpy())
        target = self.transform(target)
        masked_img = self.transform(masked_img)

        return masked_img_4d, target, mask.reshape(1, 256, 256), masked_img
    