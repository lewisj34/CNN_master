import torch 
import torch.nn as nn 
from einops import rearrange

from .blocks import Block
from .utils import init_weights

from seg.model.CNN.CNN_parts import Up, DoubleConv, OutConv
import torch.nn.functional as F

class UpMod(nn.Module):
    """Upscaling but with modifiable scale_factor  then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

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

class DecoderPlus(nn.Module):
    def __init__(
        self,
        input_size=(16, 16),
        in_chans=1,
        output_size=(256, 256),
        inter_chans=32,
        out_chans=1,
    ):
        super(DecoderPlus, self).__init__()
        assert input_size == (16, 16), \
            f'input_size must be corresponding to [N, 1, 16, 16]' 
            # haven't done for [N, 1, 32, 32] yet

        self.output_size = output_size
        self.up1 = UpMod(
            in_channels = in_chans, 
            out_channels = inter_chans,
            bilinear=True,
            scale_factor=4,
        )
        self.up2 = UpMod(
            in_channels = inter_chans, 
            out_channels = inter_chans,
            bilinear=True,
            scale_factor=4,
        )
        self.conv = nn.Conv2d(inter_chans, out_chans, kernel_size=1)
        self.final_conv = nn.Conv2d(2, 1, kernel_size=1)
    def forward(self, x):
        x_final_dec = F.upsample_bilinear(x, size=self.output_size) # use this for attn
        x = self.up1(x)
        x = self.up2(x)
        x = self.conv(x)
        x = torch.cat([x, x_final_dec], dim=1)
        x = self.final_conv(x)
        return x 



if __name__ == '__main__':
    input = (1, 16, 16)
    batch_size = 5

    from torchsummary import summary 
    model = DecoderPlus(
        input_size=(16,16),
        output_size=(256,256),
        inter_chans=32, 
        out_chans=1
    )

    summary(
        model=model.cuda(),
        input_size=input,
        batch_size=batch_size
    )
