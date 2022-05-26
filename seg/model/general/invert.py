import torch
import torch.nn as nn
import torch.nn.functional as F 

from .DW_sep import SeparableConv2D

class NegativeInvert(nn.Module):
    def __init__(
        self, 
        expand_channels,
        out_chans=64,
        scale_factor=2,
    ):
        super(NegativeInvert, self).__init__()
        """
        Calls the negative inverse sigmoid function that appears in PraNet
        Takes in a smaller representation of the desired target and multiples it 
        with a same size feature map from somewhere else in the network.  
            1. Scales it up to output size. 
            2. Expands it to the number of channels the output has. 
            3. Multiples the two features.
            @expand_channels: number of channels it needs to be expanded to. 
            @scale_factor: the size to scale it up to. 
        """

        self.scale_factor = scale_factor
        self.chans = expand_channels
        self.out_chans = out_chans

        self.PWconv1 = nn.Sequential(
            nn.Conv2d(self.chans, self.out_chans, kernel_size=1),
            nn.BatchNorm2d(self.out_chans),
        )

        self.conv2 = nn.Sequential(
            SeparableConv2D(self.out_chans, self.out_chans, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(self.out_chans),
            nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            SeparableConv2D(self.out_chans, self.out_chans, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(self.out_chans),
            nn.ReLU(True),
        )

    def forward(self, x_small, x_in):
        inv = F.interpolate(
            x_small, 
            scale_factor=self.scale_factor, 
            mode='bilinear',
            align_corners=False,
        )

        x = -1 * (torch.sigmoid(inv)) + 1
        x = x.expand(-1, self.chans, -1, -1).mul(x_in)
        x = self.PWconv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x * inv   
        return x 

if __name__ == '__main__':
    x_small = torch.randn((2, 1, 11, 11))
    x_in = torch.randn((2, 1024, 22, 22))

    model = NegativeInvert(
        expand_channels=1024, 
        scale_factor=2
    )

    print(f'[x_small]: \t{x_small.shape}')
    print(f'[x_in]: \t{x_in.shape}')
    x_out = model(x_small, x_in)
    print(f'[x_out]: \t{x_out.shape}')
    print(f'COMPLETE.')
