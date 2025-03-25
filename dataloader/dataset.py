import glob
import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from dataloader.masks import gen_mask


class MaskedImgDataset(Dataset):
    def __init__(self, path):
        self.img_paths = list(glob.glob(os.path.join(path, '*', '*', '*.jpg')))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = gen_mask(256, 256)
        img = (img * mask).astype(np.uint8)

        img = self.transform(img)

        return img
    