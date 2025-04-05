import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        features = []

        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x)


        return features[-1], features[:-1]


if __name__ == '__main__':
    disc = Discriminator()
    img = cv2.imread('data/places_subset_50000/0-1000/00001.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.Tensor([img])
    res = disc(img)
    for f in res[1]:
        print(f.shape)
    print(res[0].shape)
