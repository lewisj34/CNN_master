import torch
import torch.nn as nn
import torch.nn.functional as F

from seg.model.zed.zedNet import zedNetMod
from seg.model.transformer.transformerV3 import create_transformerV3
from .fuse import MiniEncoderFuse



class NewZedFusionNetworkMOD(nn.Module):
    """
    Same as NewZedFusionNetwork just with modifiable number of channels for CNN 
    """
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        with_fusion=True,
        num_output_channels_cnn = [64, 128, 256, 512, 512, 256, 128, 64, 64],
        trans_decoder_inter_chans = [512, 64, 32, 16],
        ):
        super(NewZedFusionNetworkMOD, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16 or self.patch_size == 32, \
            'patch_size must be {16, 32}'

        self.cnn_branch = zedNetMod(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            num_output_channels = num_output_channels_cnn,
            bilinear=True,
        )
        # now populate dimensions 
        self.cnn_branch.get_dimensions(
            N_in = cnn_model_cfg['batch_size'],
            C_in = cnn_model_cfg['in_channels'],
            H_in = cnn_model_cfg['image_size'][0], 
            W_in = cnn_model_cfg['image_size'][1]
        )

        print(f'Warning in file: {__file__}, we are manually assigning the \
decoder to have a `linear` value in create_transformer when creating the \
fusion network and thus not using the decoder value input to main() in \
train.py, but im too tired to try and figure out how to work that and were \
running the terminal right now so...') # SEE BELOW.... decoder = 'linear'
        # need to do something or pull information from the trans_model_cfg and
        #  pull that info. but yeah. wahtever rn lol 
        self.trans_branch = create_transformerV3(
            trans_model_cfg, 
            decoder='linear', 
            num_chans_CNN=num_output_channels_cnn[1:4], 
            inter_chans=trans_decoder_inter_chans,
        )
        
        num_output_trans = trans_model_cfg['num_output_trans']
        print(f'num_output_trans: {num_output_trans}')
        # num_output_trans = 64

        self.with_fusion = with_fusion
        if self.with_fusion:
            self.fuse_1_2 = MiniEncoderFuse( # NOTE: 64 classes trans output manually input here 
                self.cnn_branch.x_1_2.shape[1], num_output_trans, 64, 1, stage = '1_2')
            self.fuse_1_4 = MiniEncoderFuse(
                self.cnn_branch.x_1_4.shape[1], num_output_trans, 64, 1, stage='1_4')
            self.fuse_1_8 = MiniEncoderFuse(
                self.cnn_branch.x_1_8.shape[1], num_output_trans, 64, 1, stage='1_8')
            self.fuse_1_16 = MiniEncoderFuse(
                self.cnn_branch.x_1_16.shape[1], num_output_trans, 64, 1, stage='1_16')
            if self.patch_size == 32:
                self.fuse_1_32 = MiniEncoderFuse(
                    self.cnn_branch.x_1_32.shape[1], num_output_trans, 64, 1, stage='1_32')

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)
        x_final_trans = self.trans_branch(images, self.cnn_branch.x_1_8, self.cnn_branch.x_1_4, self.cnn_branch.x_1_2) # 5 x 1 x 156 x 156

        '''
        self.CNN_BRANCH and self.TRANSFORMER_BRANCH should have same members:
                { output_1_4, output_1_2 }
        '''
        if self.with_fusion:
            # 1 / 2 - note (kind of wack given that you have to interploate from 1/4)
            self.x_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, self.trans_branch.x_1_2)
            self.x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans_branch.x_1_4)
            self.x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans_branch.x_1_8)
            self.x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans_branch.x_1_16)

            if self.patch_size == 16:
                tensor_list = [x_final_cnn, x_final_trans, self.x_1_2, self.x_1_4, self.x_1_8, self.x_1_16]
                mean = torch.mean(torch.stack(tensor_list), dim=0) 
                return mean
            elif self.patch_size == 32:
                x_1_32 = self.fuse_1_32(self.cnn_branch.x_1_32, self.trans_branch.x_1_32)
                tensor_list = [x_final_cnn, x_final_trans, self.x_1_2, self.x_1_4, self.x_1_8, self.x_1_16, self.x_1_32]
                mean = torch.mean(torch.stack(tensor_list), dim=0) 
                return mean

class NewZedFusionNetworkMODPWOut(nn.Module):
    """
    Same as NewZedFusionNetwork just with modifiable number of channels for CNN 
    """
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        with_fusion=True,
        num_output_channels_cnn = [64, 128, 256, 512, 512, 256, 128, 64, 64],
        trans_decoder_inter_chans = [512, 64, 32, 16],
        ):
        super(NewZedFusionNetworkMODPWOut, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16 or self.patch_size == 32, \
            'patch_size must be {16, 32}'

        self.cnn_branch = zedNetMod(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            num_output_channels = num_output_channels_cnn,
            bilinear=True,
        )
        # now populate dimensions 
        self.cnn_branch.get_dimensions(
            N_in = cnn_model_cfg['batch_size'],
            C_in = cnn_model_cfg['in_channels'],
            H_in = cnn_model_cfg['image_size'][0], 
            W_in = cnn_model_cfg['image_size'][1]
        )

        print(f'Warning in file: {__file__}, we are manually assigning the \
decoder to have a `linear` value in create_transformer when creating the \
fusion network and thus not using the decoder value input to main() in \
train.py, but im too tired to try and figure out how to work that and were \
running the terminal right now so...') # SEE BELOW.... decoder = 'linear'
        # need to do something or pull information from the trans_model_cfg and
        #  pull that info. but yeah. wahtever rn lol 
        self.trans_branch = create_transformerV3(
            trans_model_cfg, 
            decoder='linear', 
            num_chans_CNN=num_output_channels_cnn[1:4], 
            inter_chans=trans_decoder_inter_chans,
        )
        
        num_output_trans = trans_model_cfg['num_output_trans']
        print(f'num_output_trans: {num_output_trans}')
        # num_output_trans = 64

        self.with_fusion = with_fusion
        if self.with_fusion:
            self.fuse_1_2 = MiniEncoderFuse( # NOTE: 64 classes trans output manually input here 
                self.cnn_branch.x_1_2.shape[1], num_output_trans, 16, 1, stage = '1_2')
            self.fuse_1_4 = MiniEncoderFuse(
                self.cnn_branch.x_1_4.shape[1], num_output_trans, 32, 1, stage='1_4')
            self.fuse_1_8 = MiniEncoderFuse(
                self.cnn_branch.x_1_8.shape[1], num_output_trans, 32, 1, stage='1_8')
            self.fuse_1_16 = MiniEncoderFuse(
                self.cnn_branch.x_1_16.shape[1], num_output_trans, 64, 1, stage='1_16')
            if self.patch_size == 32:
                self.fuse_1_32 = MiniEncoderFuse(
                    self.cnn_branch.x_1_32.shape[1], num_output_trans, 64, 1, stage='1_32')
        
        self.conv_out = nn.Conv2d(6, 1, 1)

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)
        x_final_trans = self.trans_branch(images, self.cnn_branch.x_1_8, self.cnn_branch.x_1_4, self.cnn_branch.x_1_2) # 5 x 1 x 156 x 156

        '''
        self.CNN_BRANCH and self.TRANSFORMER_BRANCH should have same members:
                { output_1_4, output_1_2 }
        '''
        # 1 / 2 - note (kind of wack given that you have to interploate from 1/4)
        self.x_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, self.trans_branch.x_1_2)
        self.x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans_branch.x_1_4)
        self.x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans_branch.x_1_8)
        self.x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans_branch.x_1_16)

        tensor_list = [x_final_cnn, x_final_trans, self.x_1_2, self.x_1_4, self.x_1_8, self.x_1_16]
        x_out = self.conv_out(torch.cat(tensor_list, dim=1))
        return x_out

class BNRconv3x3(nn.Module):
    def __init__(
        self,
        in_planes, 
        out_planes, 
        stride=1, 
        groups=1, 
        dilation=1,
    ):
        super(BNRconv3x3, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x 

class BNRdilatedconv3x3(nn.Module):
    def __init__(
        self,
        in_planes, 
        out_planes, 
        stride=1, 
        groups=1, 
        dilation=1,
    ):
        super(BNRdilatedconv3x3, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x 

class Fuse(nn.Module):
    def __init__(
        self,
        in_chan_c,
        in_chan_t,
        inter_chan,
        out_chan,
        stage, # to be taken out 
    ):
        """
        For now I want to design this to be as similar as possible to the "miniencoderFuse"
        so that we can do a true check of the parameters, i dont want to actually include an upsampling operation in this one 
        but we'll keep it there for now so that again we can check the number of parameters 
        """
        super().__init__()

        # again this section of code with the if else structure is to be taken out 
        self.fuse_stage = stage
        if self.fuse_stage == '1_8':
            
        
            self.c1 = BNRconv3x3(in_chan_c + in_chan_t)


class NewZedFusionNetworkModifiedFusion(nn.Module):
    def __init__(
        self,
        cnn_model_cfg,
        trans_model_cfg,
        num_output_channels_cnn = [64, 128, 256, 512, 512, 256, 128, 64, 64],
        trans_decoder_inter_chans = [512, 64, 32, 16],
    ):
        """
        NewZedFusion with modifiable zed CNN channels, but also modified fusion 
        """
        super().__init__()
        print(f'NewZedFusionNetworkModified initiated.')


class PWConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_out = nn.Conv2d(6, 1, 1)
    
    def forward(self, tensor_list: list):
        x_out = torch.cat(tensor_list, dim=1)
        x_out = self.conv_out(x_out)
        return x_out


if __name__ == '__main__':
    tensor_list = list()

    for i in range(6):
        tensor_list.append(torch.randn((3, 1, 256, 256), device='cuda'))
    
    model = PWConv().cuda()

    out = model(tensor_list); print(out.shape)