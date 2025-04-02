import torch
import torch.nn as nn


class FFC(nn.Module):
    def __init__(self, dims, ratio_g, kernel_size=3, stride=1, padding=1):
        super(FFC, self).__init__()

        dim_g = int(ratio_g * dims)
        dim_l = dims - dim_g

        self.f_l2l = nn.Conv2d(dim_l, dim_l, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect', bias=False)
        self.f_l2g = nn.Conv2d(dim_l, dim_g, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect', bias=False)
        self.f_g2l = nn.Conv2d(dim_g, dim_l, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect', bias=False)
        self.f_g2g = SpectralTransform(dim_g, dim_g, stride=1)

        self.norm_l = nn.BatchNorm2d(dim_l)
        self.activation_l = nn.ReLU()
        self.norm_g = nn.BatchNorm2d(dim_g)
        self.activation_g = nn.ReLU()

    def forward(self, x):
        x_l, x_g = x

        x_l2l = self.f_l2l(x_l)
        x_l2g = self.f_l2g(x_l)
        x_g2l = self.f_g2l(x_g)
        x_g2g = self.f_g2g(x_g)

        x_l, x_g = x_l2l + x_g2l, x_l2g + x_g2g
        x_l, x_g = self.activation_l(self.norm_l(x_l)), self.activation_g(self.norm_g(x_g))

        return x_l, x_g


class SpectralTransform(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super(SpectralTransform, self).__init__()

        self.chennel_reduction = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_dim // 2),
            nn.ReLU()
        )

        self.fu = FourierUnit(out_dim // 2, out_dim // 2)

        self.channel_promotion = nn.Conv2d(out_dim // 2, out_dim, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        pre_fft = self.chennel_reduction(x)
        post_fft = self.fu(pre_fft)
        return self.channel_promotion(pre_fft + post_fft)


class FourierUnit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FourierUnit, self).__init__()

        self.conv = nn.Conv2d(in_dim*2, out_dim*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_dim * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y_ri = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')
        y_r, y_i = y_ri.real, y_ri.imag
        y = torch.stack((y_r, y_i), dim=-1).permute(0, 1, 4, 2, 3).contiguous()
        y = y.view((x.shape[0], -1, y.shape[3], y.shape[4]))
        y = self.relu(self.bn(self.conv(y)))
        y = y.view((x.shape[0], -1, 2, y.shape[2], y.shape[3]))
        y = y.permute(0, 1, 3, 4, 2).contiguous()
        y_r, y_i = y[..., 0], y[..., 1]
        y_ri = torch.complex(y_r, y_i)
        z = torch.fft.irfftn(y_ri, dim=(-2, -1), norm='ortho')
        return z

        # Pseudo code from FFC paper for fourier unit
        # y_r, y_i = FFT(x) # y_r/y_i: [N,C,H,b floor(W/2)+1]
        # y = Concatenate([y_r, y_i], dim=1) # [N,C∗2,H,b floor(W/2)+1]
        # y = ReLU(BN(Conv(y))) # [N,C∗2,H,b floor(W/2)+1]
        # y_r, y_i = Split(y, dim=1) # y_r/y_i: [N,C,H,b floor(W/2)+1]
        # z = iFFT(y_r, y_i) # [N,C,H,W]
        # return z
