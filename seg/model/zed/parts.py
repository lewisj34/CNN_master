import torch
import torch.nn as nn 
import torch.nn.functional as F

from torchsummary import summary

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => SiLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class DownASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            ASPP(
                inplanes = out_channels,
                outplanes = out_channels,
            )
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)

class NouveauAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, AvgPoolKernelSize=31, AvgPoolPadding=15):
        super().__init__()
        self.cSE = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.AvgPool2d(kernel_size=AvgPoolKernelSize, stride=1, padding=AvgPoolPadding),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class UpAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        mode='bilinear',
    ):
        super().__init__()
        self.conv1 = DoubleConv(
            in_channels=in_channels + skip_channels,
            out_channels=out_channels,
            mid_channels=in_channels // 2,
        )
        self.attention1 = SCSEModule(in_channels=in_channels + skip_channels)
        # self.conv2 = DoubleConv(
        #     in_channels=out_channels,
        #     out_channels=out_channels,
        #     mid_channels=in_channels // 2,
        # )
        # self.attention2 = SCSEModule(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True,)
        if skip is not None:
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            # print(f'x.shape: {x.shape}')
            # print(f'skip.shape: {skip.shape}')
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)

            x = self.attention1(x)
        x = self.conv1(x)
        return x


    # def forward(self, x1, x2=None):
    #     x1 = self.up(x1)
    #     # input is CHW
    #     if x2 is not None:
    #         diffY = x2.size()[2] - x1.size()[2]
    #         diffX = x2.size()[3] - x1.size()[3]

    #         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
    #                         diffY // 2, diffY - diffY // 2])
    #         x = torch.cat([x2, x1], dim=1)
    #     else:
    #         x = x1
    #     return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

if __name__ == '__main__':
    model = Up(
        in_channels = 64, 
        out_channels = 100, 
    )

    from torchsummary import summary 
    summary(
        model = model.cuda(), 
        input_size = (64, 128, 128),
        batch_size = 10
    )