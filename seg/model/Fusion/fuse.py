from multiprocessing.sharedctypes import Value
import numpy as np
from sklearn.preprocessing import scale
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from seg.model.CNN.CNN_parts import Down, Up
from seg.model.siddnet.parts import SuperficialModule

from seg.model.zed.parts import DownDWSep, UpDWSep, UpDWSepRFB
from seg.utils.check_parameters import count_parameters

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
        assert x_CNN.shape[0] == x_TRANS.shape[0], \
            f'N, batch_size different. Given: {x_CNN.shape[0], x_TRANS.shape[0]}'
        assert x_CNN.shape[2] == x_TRANS.shape[2], \
            f'H, height different. Given: {x_CNN.shape[2], x_TRANS.shape[2]}'
        assert x_CNN.shape[3] == x_TRANS.shape[3], \
            f'W, width different. Given: {x_CNN.shape[3], x_TRANS.shape[3]}'
            
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

# for usage with AblationStudies (where we have NoViT) could have just implemented 
# the change of if x_TRANS is not None in MiniEncoderFuseDWSep but I don't want to mess 
# with anything 
class MiniEncoderFuseDWSepNoTrans(nn.Module):
    def __init__(
        self, 
        in_chan_CNN, 
        in_chan_TRANS, 
        intermediate_chan,
        out_chan=1,
        stage=None,
        drop_rate = 0.5
        ):
        super(MiniEncoderFuseDWSepNoTrans, self).__init__()

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
        
        if x_TRANS is not None:
            x = torch.cat([x_CNN, x_TRANS], dim=1) #; print(f'\tcat output {x.shape}')
        else:
            x = x_CNN

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

class MiniEncoderFuseDWSepRFB(nn.Module):
    def __init__(
        self, 
        in_chan_CNN, 
        in_chan_TRANS, 
        intermediate_chan,
        out_chan=1,
        stage=None,
        drop_rate = 0.5
        ):
        super(MiniEncoderFuseDWSepRFB, self).__init__()

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
        self.up2 = UpDWSepRFB(intermediate_chan, out_chan)

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

class CCMSubBlock(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d

        combine_kernel = 2 * d - 1

        self.conv = nn.Sequential(
            nn.Conv2d(nIn, nIn, kernel_size=(combine_kernel, 1), stride=stride, padding=(padding - 1, 0),
                        groups=nIn, bias=False),
            nn.BatchNorm2d(nIn),
            nn.PReLU(nIn),
            nn.Conv2d(nIn, nIn, kernel_size=(1, combine_kernel), stride=stride, padding=(0, padding - 1),
                        groups=nIn, bias=False),
            nn.BatchNorm2d(nIn),
            nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False,
                        dilation=d),
            nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False))

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class dualCCM(nn.Module):
    def __init__(
        self,
        nIn,
        nOut,
        kSize=3,
        d=[2, 3],
    ):
        super().__init__()
        print(f'CCM initialized.')
        self.CCM1 = CCMSubBlock(nIn, nOut, kSize, stride=1, d=d[0])
        self.CCM2 = CCMSubBlock(nIn, nOut, kSize, stride=1, d=d[1])
    def forward(self, x):
        x = self.CCM1(x)
        x = self.CCM2(x)
        x = F.relu6(x, inplace=True)
        return x 

from seg.model.general.DW_sep import SeparableConv2D
class CCMFusionModule(nn.Module):
    def __init__(
        self,
        in_channels_trans,
        in_channels_cnn,
        inter_channels,
        out_channels=1,
        stage=None, 
    ):
        super().__init__()

        stages = ['1_2', '1_4', '1_8', '1_16', '1_32']
        self.fuse_stage = stage
        assert self.fuse_stage in stages

        if self.fuse_stage == '1_2':
            self.scale_factor = 2
            dilations = [5, 4]
        elif self.fuse_stage == '1_4':
            self.scale_factor = 4
            dilations = [4, 3]
        elif self.fuse_stage == '1_8':
            self.scale_factor = 8 
            dilations = [3, 2]
        elif self.fuse_stage == '1_16':
            self.scale_factor = 16
            dilations = [2, 2]
        else:
            raise ValueError(f'Valid stages for fusion: {stages}')

        self.down1 = DownDWSep(in_channels_trans + in_channels_cnn, inter_channels)
        self.down2 = DownDWSep(inter_channels, inter_channels)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.CCM1 = dualCCM(inter_channels, inter_channels, kSize=3, d=dilations)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.CCM2 = dualCCM(inter_channels, inter_channels, kSize=3, d=dilations)
        self.DWSep = SeparableConv2D(inter_channels, out_channels, kernel_size=1)
    
    def forward(self, x_cnn, x_trans):
        x = torch.cat([x_cnn, x_trans], dim=1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.up1(x)
        x = self.CCM1(x)
        x = self.up2(x)
        x = self.CCM2(x)
        x = self.DWSep(x)
        return x


        

if __name__ == '__main__':
    fuse_1_2_reg = MiniEncoderFuse(256, 64, 64, 1, stage='1_2').cuda()
    fuse_1_2_DW = MiniEncoderFuseDWSep(256, 64, 64, 1, stage='1_2').cuda()
    count_parameters(fuse_1_2_DW)

    input_size = (512, 512)
    # 1 / 2 size input
    print(f'\n\n1 / 2  Input')
    input_cnn = torch.randn((2, 256, input_size[0] // 2, input_size[1] // 2), device='cuda')
    input_trans = torch.randn((2, 64, input_size[0] // 2, input_size[1] // 2), device='cuda')
    CCM_1_2_model = CCMFusionModule(input_trans.shape[1], input_cnn.shape[1], inter_channels=64, out_channels=1, stage='1_2').cuda()
    output = CCM_1_2_model(input_cnn, input_trans)
    print(f'output.shape: {output.shape}')
    count_parameters(CCM_1_2_model)


    # 1 / 4 size input
    print(f'\n\n1 / 4  Input')
    input_cnn = torch.randn((2, 256, input_size[0] // 4, input_size[1] // 4), device='cuda')
    input_trans = torch.randn((2, 64, input_size[0] // 4, input_size[1] // 4), device='cuda')
    CCM_1_2_model = CCMFusionModule(input_trans.shape[1], input_cnn.shape[1], inter_channels=64, out_channels=1, stage='1_4').cuda()
    output = CCM_1_2_model(input_cnn, input_trans)
    print(f'output.shape: {output.shape}')

    # 1 / 8 size input
    print(f'\n\n1 / 8  Input')
    input_cnn = torch.randn((2, 256, input_size[0] // 8, input_size[1] // 8), device='cuda')
    input_trans = torch.randn((2, 64, input_size[0] // 8, input_size[1] // 8), device='cuda')
    CCM_1_2_model = CCMFusionModule(input_trans.shape[1], input_cnn.shape[1], inter_channels=64, out_channels=1, stage='1_8').cuda()
    output = CCM_1_2_model(input_cnn, input_trans)
    print(f'output.shape: {output.shape}')

    # 1 / 16 size input
    print(f'\n\n1 / 16  Input')
    input_cnn = torch.randn((2, 256, input_size[0] // 16, input_size[1] // 16), device='cuda')
    input_trans = torch.randn((2, 64, input_size[0] // 16, input_size[1] // 16), device='cuda')
    CCM_1_2_model = CCMFusionModule(input_trans.shape[1], input_cnn.shape[1], inter_channels=64, out_channels=1, stage='1_16').cuda()
    output = CCM_1_2_model(input_cnn, input_trans)
    print(f'output.shape: {output.shape}')

    # # 1 / 2 size input
    # test_CCM_input = torch.randn((2, 256, 128, 128), device='cuda')
    # CCM_sub = CCMSubBlock(256, 64, kSize=3, stride=1, d=1).cuda()
    # output = CCM_sub(test_CCM_input)
    # print(f'output of CCM using k = 3, d = 1: {output.shape}')

    # test_CCM_input = torch.randn((2, 256, 64, 64), device='cuda')
    # CCM_sub = CCMSubBlock(256, 64, kSize=4, stride=1, d=1).cuda()
    # output = CCM_sub(test_CCM_input)
    # print(f'output of CCM using k = 4, d = 1: {output.shape}')

    # test_CCM_input = torch.randn((2, 256, 64, 64), device='cuda')
    # CCM_sub = CCMSubBlock(256, 64, kSize=3, stride=1, d=2).cuda()
    # output = CCM_sub(test_CCM_input)
    # print(f'output of CCM using k = 3, d = 2: {output.shape}')