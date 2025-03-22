import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ffc import FFC
# from models.ffc_deprecated import FFC_BN_ACT


class FFCGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        # Downsample
        self.downsample = [nn.ReflectionPad2d(3),
                           nn.Conv2d(4, 64, kernel_size=7, padding=0, padding_mode='reflect')]
        for i in range(2):
            self.downsample += [
                nn.Conv2d(64*(2**i), 128*(2**i), kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
                nn.BatchNorm2d(128*(2**i)),
                nn.ReLU()
            ]
        self.downsample += [#FFC_BN_ACT(64*(2**(i+1)), 128*(2**(i+1)), kernel_size=3, stride=2, padding=1, ratio_gin=0, ratio_gout=0.75, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)
                            Splitter(64*(2**(i+1)), 128*(2**(i+1)), 0.75, kernel_size=3, stride=2, padding=1)]
        self.downsample = nn.Sequential(*self.downsample)

        # FCC Blocks
        self.ffc = []
        for _ in range(9):
            self.ffc += [FFC_block(512)]
        self.ffc = nn.Sequential(*self.ffc)

        # Join local + global
        self.joiner = Joiner()

        # Upsample
        self.upsample = []
        for i in range(3):
            self.upsample += [
                nn.ConvTranspose2d(128*(2**(2-i)), 64*(2**(2-i)), kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64*(2**(2-i))),
                nn.ReLU()
            ]
        self.upsample += [nn.ReflectionPad2d(3),
                          nn.Conv2d(64, 3, kernel_size=7, padding=0)]
        self.upsample = nn.Sequential(*self.upsample)

    def forward(self, x):
        x = self.downsample(x)
        x = self.ffc(x)
        x = self.joiner(x)
        x = self.upsample(x)
        return x


class FFC_block(nn.Module):
    def __init__(self, dims):
        super().__init__()

        #self.ffc1 = FFC_BN_ACT(dims, dims, kernel_size=3, padding=1, ratio_gin=0.75, ratio_gout=0.75)
        #self.ffc2 = FFC_BN_ACT(dims, dims, kernel_size=3, padding=1, ratio_gin=0.75, ratio_gout=0.75)
        self.ffc1 = FFC(dims, 0.75)
        self.ffc2 = FFC(dims, 0.75)

    def forward(self, x):
        x_l, x_g = x
        x_l_t, x_g_t = x_l, x_g

        x_l, x_g = self.ffc1((x_l, x_g))
        x_l, x_g = self.ffc2((x_l, x_g))

        x_l, x_g = x_l + x_l_t, x_g + x_g_t
        return x_l, x_g


class Splitter(nn.Module):
    def __init__(self, in_dim, out_dim, ratio_g, kernel_size=3, stride=2, padding=1):
        super().__init__()

        out_dim_g = int(out_dim * ratio_g)
        out_dim_l = out_dim - out_dim_g

        self.norm_l = nn.BatchNorm2d(out_dim_l)
        self.activation_l = nn.ReLU()
        self.conv_l = nn.Conv2d(in_dim, out_dim_l, kernel_size=kernel_size, stride=stride, 
                                padding=padding, padding_mode='reflect')

        self.norm_g = nn.BatchNorm2d(out_dim_g)
        self.activation_g = nn.ReLU()
        self.conv_g = nn.Conv2d(in_dim, out_dim_g, kernel_size=kernel_size, stride=stride, 
                                padding=padding, padding_mode='reflect')

    def forward(self, x):
        x_l = self.activation_l(self.norm_l(self.conv_l(x)))
        x_g = self.activation_g(self.norm_g(self.conv_g(x)))

        return x_l, x_g
    

class Joiner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat(x, dim=1)
    