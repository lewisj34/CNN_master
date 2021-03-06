import torch
import torch.nn as nn 
import torch.nn.functional as F
import yaml
from pathlib import Path

from seg.model.CNN.CNN import CNN_BRANCH, modUNet
from seg.model.CNN.CNN_parts import Up

from seg.model.transformer.create_modelV2 import create_transformerV2
from seg.utils.check_parameters import count_parameters
from ..fuse import MiniEncoderFuse
from .parts import RFB_modified, BNRconv3x3

from timm.models.layers import trunc_normal_

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class RFBFusionModule1_16(nn.Module):
    def __init__(
        self,
        in_chan_t,
        in_chan_c,
        out_chan=512,
        inter_chan=None,
        dilation=1
    ):
        super().__init__()
        """
        Similar to MiniEncoderFusion however, this module does not 
        provide upsampling. This will have to be done separately. 
        """
        print(f'RFBFusionModule initiated.')

        if inter_chan == None:
            inter_chan = out_chan

        self.rfb_t = RFB_modified(in_chan_t, inter_chan)
        self.rfb_c = RFB_modified(in_chan_c, inter_chan)

        self.conv = BNRconv3x3(inter_chan * 2, out_chan, dilation=dilation)

        print(f'Initializing weights...')
        self.apply(init_weights)

    
    def forward(self, x_c, x_t, x_extra=None):
        x_t = self.rfb_t(x_t)
        x_c = self.rfb_c(x_c)

        if x_extra == None:
            x_out = torch.cat([x_t, x_c], dim=1)
        else:
            x_out = torch.cat([x_t, x_c, x_extra], dim=1)

        x_out = self.conv(x_out)

        return x_out
        
class RFBFusionModule1_8(nn.Module):
    def __init__(
        self,
        in_chan_t,
        in_chan_c,
        out_chan=256,
        in_chan_x_16=1,
        inter_chan=None,
        dilation1=1,
        dilation2=3,
    ):
        super().__init__()

        """
        Similar to MiniEncoderFusion however, this module does not 
        provide upsampling. This will have to be done separately. 
        """
        print(f'RFBFusionModule initiated.')

        if inter_chan == None:
            inter_chan = out_chan

        self.rfb_t = RFB_modified(in_chan_t, inter_chan)
        self.rfb_c = RFB_modified(in_chan_c, inter_chan)

        self.conv1 = BNRconv3x3(inter_chan * 2 + in_chan_x_16, out_chan, dilation=dilation1)
        self.conv2 = BNRconv3x3(out_chan, out_chan, dilation=dilation2)
        
        print(f'Initializing weights...')
        self.apply(init_weights)
    
    def forward(self, x_c, x_t, x_extra=None):
        x_t = self.rfb_t(x_t)
        x_c = self.rfb_c(x_c)

        if x_extra == None:
            x_out = torch.cat([x_t, x_c], dim=1)
        else:
            x_out = torch.cat([x_t, x_c, x_extra], dim=1)

        x_out = self.conv1(x_out)
        x_out = self.conv2(x_out)

        return x_out

class RFBFusionModule1_4(nn.Module):
    def __init__(
        self,
        in_chan_t,
        in_chan_c,
        out_chan=256,
        in_chan_x_8=1,
        inter_chan=None,
        dilation1=1,
        dilation2=3,
        dilation3=6,
    ):
        super().__init__()

        """
        Similar to MiniEncoderFusion however, this module does not 
        provide upsampling. This will have to be done separately. 
        """
        print(f'RFBFusionModule initiated.')

        if inter_chan == None:
            inter_chan = out_chan

        self.rfb_t = RFB_modified(in_chan_t, inter_chan)
        self.rfb_c = RFB_modified(in_chan_c, inter_chan)

        self.conv1 = BNRconv3x3(inter_chan * 2 + in_chan_x_8, out_chan, dilation=dilation1)
        self.conv2 = BNRconv3x3(out_chan, out_chan, dilation=dilation2)
        self.conv3 = BNRconv3x3(out_chan, out_chan, dilation=dilation3)

        print(f'Initializing weights...')
        self.apply(init_weights)
    
    def forward(self, x_c, x_t, x_extra=None):
        x_t = self.rfb_t(x_t)
        x_c = self.rfb_c(x_c)

        if x_extra == None:
            x_out = torch.cat([x_t, x_c], dim=1)
        else:
            x_out = torch.cat([x_t, x_c, x_extra], dim=1)

        x_out = self.conv1(x_out)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)

        return x_out

class FusionNetworkRFB(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        out_chans = [64, 32, 16, 8],
        dilation1=1,
        dilation2=2,
        dilation3=3,
    ):
        super(FusionNetworkRFB, self).__init__()

        self.cnn_branch = modUNet(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            bilinear=True,
            channels=[64 // 2, 128 // 2, 256 // 2, 512 // 2, 1024 // 2, 512 // 2, 256 // 2, 128 // 2, 64 // 2],
        )
        # now populate dimensions 
        self.cnn_branch.get_dimensions(
            N_in = cnn_model_cfg['batch_size'],
            C_in = cnn_model_cfg['in_channels'],
            H_in = cnn_model_cfg['image_size'][0], 
            W_in = cnn_model_cfg['image_size'][1]
        )
        self.trans_branch = create_transformerV2(trans_model_cfg, 
            decoder='linear')
        num_output_trans = trans_model_cfg['num_output_trans']
        print(f'num_output_trans: {num_output_trans}')
        # num_output_trans = 64

        # fusion pipeline
        # out_chans = [256, 128, 64]

        self.fuse_1_16 = RFBFusionModule1_16(
            in_chan_t=num_output_trans, 
            in_chan_c=self.cnn_branch.x_1_16.shape[1],
            out_chan=out_chans[0], # 128
            dilation=dilation1, # 1
        )
        self.fuse_1_8 = RFBFusionModule1_8(
            in_chan_t=num_output_trans, 
            in_chan_c=self.cnn_branch.x_1_8.shape[1],
            in_chan_x_16=out_chans[0], # 128
            out_chan=out_chans[1], # 64
            dilation1=dilation1, # 1 
            dilation2=dilation2, # 2
        )
        self.fuse_1_4 = RFBFusionModule1_4(
            in_chan_t=num_output_trans, 
            in_chan_c=self.cnn_branch.x_1_4.shape[1],
            in_chan_x_8=out_chans[1], # 64
            out_chan=out_chans[2], # 32
            dilation1=dilation1,
            dilation2=dilation2,
            dilation3=dilation3,
        )
        self.up_fuse1 = Up(
            in_channels=out_chans[2],
            out_channels=out_chans[1], 
            bilinear=True,
        )

        self.up_fuse2 = Up(
            in_channels=out_chans[1],
            out_channels=1,
            bilinear=True,
        )
        
        print(f'Initializing weights...')
        self.apply(init_weights)

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)
        x_final_trans = self.trans_branch(images) # 5 x 1 x 156 x 156

        # fusion pipeline
        x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans_branch.x_1_16)
        x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans_branch.x_1_8, F.upsample_bilinear(x_1_16, scale_factor=2))
        x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans_branch.x_1_4, F.upsample_bilinear(x_1_8, scale_factor=2))

        x_fuse = self.up_fuse1(x_1_4)
        x_fuse = self.up_fuse2(x_fuse)

        tensor_list = [x_final_cnn, x_final_trans, x_fuse]
        mean = torch.mean(torch.stack(tensor_list), dim=0) 
        return mean

if __name__ == '__main__':
    out_chans = [128, 64, 32]

    model = RFBFusionModule1_16(
        in_chan_t=64, 
        in_chan_c=512,
        out_chan=out_chans[0],
        dilation=1,
    )

    p1 = count_parameters(model)

    model = RFBFusionModule1_8(
        in_chan_t=64, 
        in_chan_c=512,
        out_chan=out_chans[1],
        in_chan_x_16=out_chans[0],
        dilation1=1,
        dilation2=3,
    )
    
    p2 = count_parameters(model)

    model = RFBFusionModule1_4(
        in_chan_t=64, 
        in_chan_c=256,
        in_chan_x_8=out_chans[1],
        out_chan=out_chans[2],
        dilation1=1,
        dilation2=3,
        dilation3=6,
    )

    p3 = count_parameters(model)

    print(f'Total params: {(p1 + p2 + p3) / 10**6}M')