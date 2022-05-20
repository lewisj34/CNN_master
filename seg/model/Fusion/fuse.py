from multiprocessing.sharedctypes import Value
import numpy as np
from sklearn.preprocessing import scale
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from seg.model.CNN.CNN_parts import Down, Up
from seg.model.siddnet.parts import SuperficialModule

from seg.model.zed.parts import DownDWSep, UpDWSep

'''
Okay so we want two fusion modules here. One where 
we decrease the CNN input DOWN to the transformer input (c_dim = 1)

And then we want another where 
we increase the transformer input UP to the CNN input (c_dim = 64 - 512) 
and then we would decrease it back down again to a single channel dimension 

or something... brain is broken right now and having a hard time thinking about 
this l0l
'''

class MiniEncoderFuseV2(nn.Module):
    def __init__(
        self, 
        in_chan_CNN, 
        in_chan_TRANS, 
        intermediate_chan,
        out_chan=1,
        stage=None,
        drop_rate = 0.5
        ):
        super(MiniEncoderFuseV2, self).__init__()

        stages = ['1_2', '1_4', '1_8', '1_16', '1_32']
        self.fuse_stage = stage
        assert self.fuse_stage in stages

        if self.fuse_stage == '1_2':
            self.scale_factor = 2
        elif self.fuse_stage == '1_4':
            self.scale_factor = 4
        elif self.fuse_stage == '1_8':
            self.scale_factor = 8 
        elif self.fuse_stage == '1_16':
            self.scale_factor = 16
        elif self.fuse_stage == '1_32':
            self.scale_factor = 32
        else:
            raise ValueError(f'Valid stages for fusion: {stages}')

        self.down1 = Down(in_chan_CNN + in_chan_TRANS, intermediate_chan)
        self.super1 = SuperficialModule(nIn=intermediate_chan)
        self.down2 = Down(intermediate_chan, intermediate_chan)
        self.up1 = Up(intermediate_chan, intermediate_chan)
        self.super2 = SuperficialModule(nIn=intermediate_chan)
        self.up2 = Up(intermediate_chan, out_chan)

    def forward(self, x_CNN, x_TRANS):
        assert(x_CNN.shape[0] == x_TRANS.shape[0] 
            and x_CNN.shape[2] == x_TRANS.shape[2] 
            and x_CNN.shape[3] == x_TRANS.shape[3])
        x = torch.cat([x_CNN, x_TRANS], dim=1)
        x = self.down1(x)
        # x = self.super1(x)
        x = self.down2(x)
        x = self.up1(x)
        # x = self.super2(x)
        x = self.up2(x)
        seg_map = F.interpolate(
            x, 
            scale_factor = self.scale_factor, 
            mode='bilinear') 
        return seg_map

class SimpleFusion(nn.Module):
    '''
    Extremely low memory fusion module. 
    Example input from Transformer, x_trans, and CNN, x_cnn: 
        [1, 1, 64, 64] and [1, 512, 64, 64]
    Convolve CNN input down such that x_trans and x_cnn are the same
        [1, 1, 64 64] and [1, 1, 64, 64]
    Concatenate such that combined output, x, is:
        [1, 2, 64, 64] 
    Convolve and interpolate such that we are at same dimension as output 
    segmentation map, x is:
        [1, 1, 64, 64]
    then after interpolating up to output segmentation map size, x is:
        [1, 1, 256, 256] or [1, 1, 512, 512] (given that these are two sizes
        we've been working with)
    Parameters: 
        @in_chan_CNN,        # input number of channels from CNN
        @in_chan_TRANS,      # input number of channels from Transformer
        @out_chan=1,         # number of output channels 
        @fuse_stage=None,    # stage in network str({1_2, 1_4, 1_8, 1_16, 1_32})
    '''
    def __init__(
        self,
        in_chan_CNN,        # input number of channels from CNN
        in_chan_TRANS,      # input number of channels from Transformer
        out_chan=1,         # number of output channels 
        fuse_stage=None,    # stage in network str({1_2, 1_4, 1_8, 1_16, 1_32})
    ):
        super(SimpleFusion, self).__init__()

        self.in_chan_CNN = in_chan_CNN
        self.in_chan_TRANS = in_chan_TRANS
        self.out_chan = out_chan

        stages = ['1_2', '1_4', '1_8', '1_16', '1_32']
        assert fuse_stage in stages, f'Valid stages for fusion: {stages}'
        self.fuse_stage = fuse_stage 
    
        if self.fuse_stage == '1_2':
            self.scale_factor = 2
        elif self.fuse_stage == '1_4':
            self.scale_factor = 4
        elif self.fuse_stage == '1_8':
            self.scale_factor = 8 
        elif self.fuse_stage == '1_16':
            self.scale_factor = 16
        elif self.fuse_stage == '1_32':
            self.scale_factor = 32
        else:
            raise ValueError(f'Valid stages for fusion: {stages}')

        self.conv_cnn1 = nn.Conv2d(
            in_channels=in_chan_CNN, 
            out_channels=1, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            dilation=1, 
            bias=True
        )
        self.conv_output = nn.Conv2d(
            in_channels=2,
            out_channels=1, 
            kernel_size=1, 
            stride=1,
            padding=0,
            dilation=1,
            bias=True
        )

    def forward(self, x_cnn, x_trans):
        # get CNN to single channel dimension via convolution 
        x_cnn = self.conv_cnn1(x_cnn) # output: [N, 1, H', W']
        x = torch.cat([x_cnn, x_trans], dim=1) # output: [N, 2, H', W']
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear') # output: [N, 2, H, W]
        x = self.conv_output(x) # output: [N, 1, H, W]
        return x
        
        

class MiniEncoderFuse(nn.Module):
    def __init__(
        self, 
        in_chan_CNN, 
        in_chan_TRANS, 
        intermediate_chan,
        out_chan=1,
        stage=None,
        drop_rate = 0.5
        ):
        super(MiniEncoderFuse, self).__init__()

        stages = ['1_2', '1_4', '1_8', '1_16', '1_32']
        self.fuse_stage = stage
        assert self.fuse_stage in stages

        if self.fuse_stage == '1_2':
            self.scale_factor = 2
        elif self.fuse_stage == '1_4':
            self.scale_factor = 4
        elif self.fuse_stage == '1_8':
            self.scale_factor = 8 
        elif self.fuse_stage == '1_16':
            self.scale_factor = 16
        elif self.fuse_stage == '1_32':
            self.scale_factor = 32
        else:
            raise ValueError(f'Valid stages for fusion: {stages}')

        self.down1 = Down(in_chan_CNN + in_chan_TRANS, intermediate_chan)
        # self.super1 = SuperficialModule(nIn=intermediate_chan)
        self.down2 = Down(intermediate_chan, intermediate_chan)
        self.up1 = Up(intermediate_chan, intermediate_chan)
        # self.super2 = SuperficialModule(nIn=intermediate_chan)
        self.up2 = Up(intermediate_chan, out_chan)

    def forward(self, x_CNN, x_TRANS):
        assert(x_CNN.shape[0] == x_TRANS.shape[0] 
            and x_CNN.shape[2] == x_TRANS.shape[2] 
            and x_CNN.shape[3] == x_TRANS.shape[3])
            
        x = torch.cat([x_CNN, x_TRANS], dim=1) #; print(f'\tcat output {x.shape}')
        x = self.down1(x) #; print(f'\tdown1 output {x.shape}')
        # x = self.super1(x)
        x = self.down2(x) #; print(f'\tdown2 output {x.shape}')
        x = self.up1(x) #; print(f'\tup1 output {x.shape}')
        # x = self.super2(x)
        x = self.up2(x) #; print(f'\tup2 output {x.shape}')
        seg_map = F.interpolate(
            x, 
            scale_factor = self.scale_factor, 
            mode='bilinear') 
        return seg_map

class MiniEncoderFuseDWSep(nn.Module):
    def __init__(
        self, 
        in_chan_CNN, 
        in_chan_TRANS, 
        intermediate_chan,
        out_chan=1,
        stage=None,
        drop_rate = 0.5
        ):
        super(MiniEncoderFuseDWSep, self).__init__()

        stages = ['1_2', '1_4', '1_8', '1_16', '1_32']
        self.fuse_stage = stage
        assert self.fuse_stage in stages

        if self.fuse_stage == '1_2':
            self.scale_factor = 2
        elif self.fuse_stage == '1_4':
            self.scale_factor = 4
        elif self.fuse_stage == '1_8':
            self.scale_factor = 8 
        elif self.fuse_stage == '1_16':
            self.scale_factor = 16
        elif self.fuse_stage == '1_32':
            self.scale_factor = 32
        else:
            raise ValueError(f'Valid stages for fusion: {stages}')

        self.down1 = DownDWSep(in_chan_CNN + in_chan_TRANS, intermediate_chan)
        # self.super1 = SuperficialModule(nIn=intermediate_chan)
        self.down2 = DownDWSep(intermediate_chan, intermediate_chan)
        self.up1 = UpDWSep(intermediate_chan, intermediate_chan)
        # self.super2 = SuperficialModule(nIn=intermediate_chan)
        self.up2 = UpDWSep(intermediate_chan, out_chan)

    def forward(self, x_CNN, x_TRANS):
        assert(x_CNN.shape[0] == x_TRANS.shape[0] 
            and x_CNN.shape[2] == x_TRANS.shape[2] 
            and x_CNN.shape[3] == x_TRANS.shape[3])
            
        x = torch.cat([x_CNN, x_TRANS], dim=1) #; print(f'\tcat output {x.shape}')
        x = self.down1(x) #; print(f'\tdown1 output {x.shape}')
        # x = self.super1(x)
        x = self.down2(x) #; print(f'\tdown2 output {x.shape}')
        x = self.up1(x) #; print(f'\tup1 output {x.shape}')
        # x = self.super2(x)
        x = self.up2(x) #; print(f'\tup2 output {x.shape}')
        seg_map = F.interpolate(
            x, 
            scale_factor = self.scale_factor, 
            mode='bilinear') 
        return seg_map

if __name__ == '__main__':
    fuse_1_2 = MiniEncoderFuse(256, 64, 64, 1, stage='1_2')
    # fuse_1_4 = MiniEncoderFuse(256, 64, 64, 1, fraction=0.25)
    # fuse_1_8 = MiniEncoderFuse(512, 128, 64, 1, fraction=0.125)

    from torchsummary import summary 
    summary(fuse_1_2, [(256, 96, 128), (64, 96, 128)], batch_size=2, device='cpu')
    # summary(fuse_1_4, [(256, 48, 64), (64, 48, 64)], batch_size=2, device='cpu')
    # summary(fuse_1_8, [(512, 24, 32), (128, 24, 32)], batch_size=2, device='cpu')
