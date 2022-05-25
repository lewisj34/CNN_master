"""
I DONT THINK THIS IS ACTUALLY ANY DIFFERENT FROM OLDFUSIONNETWORK, I MADE THIS FILE on APRIL 19TH 
AND DOESNT REFLECT THE ADDITION OF CREATE_TRANSFORMER_V2
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F 

import yaml
from pathlib import Path
from seg.model.Fusion.CondensedFusion import BNRconv3x3
from seg.model.alt_cnns.pranetSimple import RFB_modified
from seg.model.transformer.decoder_new import DecoderMultiClassDilationAndSCSEFusion
from seg.model.transformer.transformerV3 import create_transformerV3

from seg.model.zed.zedNet import zedNet, zedNetDWSep, zedNetMod
from seg.model.transformer.create_model import create_transformer
from seg.model.transformer.create_modelV2 import create_transformerV2
from seg.model.Fusion.fuse import SimpleFusion
from .fuse import CCMFusionModule, MiniEncoderFuse, MiniEncoderFuseDWSep, MiniEncoderFuseDWSepRFB

# NOTE THIS IS THE VERSION WITH ONE CHANNEL INPUT FROM THE TRANSFORMER !!!!!
class ZedFusionNetwork(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        with_fusion=True,
        ):
        super(ZedFusionNetwork, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16 or self.patch_size == 32, \
            'patch_size must be {16, 32}'

        self.cnn_branch = zedNet(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            bilinear=True,
            attention=True,
        )

        # now populate dimensions 
        self.cnn_branch.get_dimensions(
            N_in = cnn_model_cfg['batch_size'],
            C_in = cnn_model_cfg['in_channels'],
            H_in = cnn_model_cfg['image_size'][0], 
            W_in = cnn_model_cfg['image_size'][1]
        )
        print(f'Warning: manually assigning linear decoder in NewFusionNetwork')
        self.trans_branch = create_transformer(trans_model_cfg, 
            decoder='linear')

        print(f'self.cnn_branch.x_1_2.shape[1]: {self.cnn_branch.x_1_2.shape[1]}')
        print(f'self.cnn_branch.x_1_4.shape[1]: {self.cnn_branch.x_1_4.shape[1]}')
        print(f'self.cnn_branch.x_1_8.shape[1]: {self.cnn_branch.x_1_8.shape[1]}')
        print(f'self.cnn_branch.x_1_16.shape[1]: {self.cnn_branch.x_1_16.shape[1]}')
        self.with_fusion = with_fusion
        if self.with_fusion:
            self.fuse_1_2 = MiniEncoderFuse(
                self.cnn_branch.x_1_2.shape[1], 1, 64, 1, stage = '1_2')
            self.fuse_1_4 = MiniEncoderFuse(
                self.cnn_branch.x_1_4.shape[1], 1, 64, 1, stage='1_4')
            self.fuse_1_8 = MiniEncoderFuse(
                self.cnn_branch.x_1_8.shape[1], 1, 64, 1, stage='1_8')
            self.fuse_1_16 = MiniEncoderFuse(
                self.cnn_branch.x_1_16.shape[1], 1, 64, 1, stage='1_16')
            if self.patch_size == 32:
                self.fuse_1_32 = MiniEncoderFuse(
                    self.cnn_branch.x_1_32.shape[1], 1, 64, 1, stage='1_32')

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)
        x_final_trans = self.trans_branch(images) # 5 x 1 x 156 x 156

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


class NewZedFusionNetwork(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        with_fusion=True,
        ):
        super(NewZedFusionNetwork, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16 or self.patch_size == 32, \
            'patch_size must be {16, 32}'

        self.cnn_branch = zedNet(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            bilinear=True,
            attention=True,
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
        self.trans_branch = create_transformerV2(trans_model_cfg, 
            decoder='linear')
        
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
        x_final_trans = self.trans_branch(images) # 5 x 1 x 156 x 156

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


class NewZedFusionNetworkDWSep(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        with_fusion=True,
        ):
        super(NewZedFusionNetworkDWSep, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16 or self.patch_size == 32, \
            'patch_size must be {16, 32}'

        self.cnn_branch = zedNetDWSep(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            bilinear=True,
            attention=True,
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
        self.trans_branch = create_transformerV2(trans_model_cfg, 
            decoder='linear')
        
        num_output_trans = trans_model_cfg['num_output_trans']
        print(f'num_output_trans: {num_output_trans}')
        # num_output_trans = 64

        self.with_fusion = with_fusion
        if self.with_fusion:
            self.fuse_1_2 = MiniEncoderFuseDWSep( # NOTE: 64 classes trans output manually input here 
                self.cnn_branch.x_1_2.shape[1], num_output_trans, 64, 1, stage = '1_2')
            self.fuse_1_4 = MiniEncoderFuseDWSep(
                self.cnn_branch.x_1_4.shape[1], num_output_trans, 64, 1, stage='1_4')
            self.fuse_1_8 = MiniEncoderFuseDWSep(
                self.cnn_branch.x_1_8.shape[1], num_output_trans, 64, 1, stage='1_8')
            self.fuse_1_16 = MiniEncoderFuseDWSep(
                self.cnn_branch.x_1_16.shape[1], num_output_trans, 64, 1, stage='1_16')
            if self.patch_size == 32:
                self.fuse_1_32 = MiniEncoderFuseDWSep(
                    self.cnn_branch.x_1_32.shape[1], num_output_trans, 64, 1, stage='1_32')

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)
        x_final_trans = self.trans_branch(images) # 5 x 1 x 156 x 156

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

class NewZedFusionNetworkDWSepRFB(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        with_fusion=True,
        ):
        super(NewZedFusionNetworkDWSepRFB, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16 or self.patch_size == 32, \
            'patch_size must be {16, 32}'

        self.cnn_branch = zedNetDWSep(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            bilinear=True,
            attention=True,
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
        self.trans_branch = create_transformerV2(trans_model_cfg, 
            decoder='linear')
        
        num_output_trans = trans_model_cfg['num_output_trans']
        print(f'num_output_trans: {num_output_trans}')
        # num_output_trans = 64

        self.with_fusion = with_fusion
        if self.with_fusion:
            self.fuse_1_2 = MiniEncoderFuseDWSepRFB( # NOTE: 64 classes trans output manually input here 
                self.cnn_branch.x_1_2.shape[1], num_output_trans, 64, 1, stage = '1_2')
            self.fuse_1_4 = MiniEncoderFuseDWSepRFB(
                self.cnn_branch.x_1_4.shape[1], num_output_trans, 64, 1, stage='1_4')
            self.fuse_1_8 = MiniEncoderFuseDWSepRFB(
                self.cnn_branch.x_1_8.shape[1], num_output_trans, 64, 1, stage='1_8')
            self.fuse_1_16 = MiniEncoderFuseDWSepRFB(
                self.cnn_branch.x_1_16.shape[1], num_output_trans, 64, 1, stage='1_16')
            if self.patch_size == 32:
                self.fuse_1_32 = MiniEncoderFuseDWSepRFB(
                    self.cnn_branch.x_1_32.shape[1], num_output_trans, 64, 1, stage='1_32')

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)
        x_final_trans = self.trans_branch(images) # 5 x 1 x 156 x 156

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

from seg.model.transformer.transformerNoDecoder import create_transformerV4
from seg.model.zed.parts import DoubleConvDWSep

class NewZedFusionAttentionTransDecoderDWSepCNN(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        with_fusion=True,
        ):
        super(NewZedFusionAttentionTransDecoderDWSepCNN, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16 or self.patch_size == 32, \
            'patch_size must be {16, 32}'

        self.cnn_branch = zedNetDWSep(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            bilinear=True,
            attention=True,
        )
        # now populate dimensions 
        self.cnn_branch.get_dimensions(
            N_in = cnn_model_cfg['batch_size'],
            C_in = cnn_model_cfg['in_channels'],
            H_in = cnn_model_cfg['image_size'][0], 
            W_in = cnn_model_cfg['image_size'][1]
        )

        self.trans_branch = create_transformerV4(trans_model_cfg, 
            decoder='linear') # output should now be [N, num_output_trans, 16, 16]

        num_output_trans = trans_model_cfg['num_output_trans']
        print(f'num_output_trans: {num_output_trans}')

        in_chans_fuse_1=512
        in_chans_fuse_2=256
        inter_chans=32

        self.conv_fuse_1 = BNRconv3x3(in_chans_fuse_1, num_output_trans)
        self.conv_fuse_2 = BNRconv3x3(in_chans_fuse_2, inter_chans)

        self.decoder_trans = DecoderMultiClassDilationAndSCSEFusion(
            input_size=(16,16),
            in_chans=num_output_trans,
            in_chans_fuse_1=num_output_trans,
            in_chans_fuse_2=inter_chans, 
            inter_chans=inter_chans, 
            out_chans=1,
            dilation1=1,
            dilation2=3,
        )
        

        # num_output_trans = 64

        self.with_fusion = with_fusion
        if self.with_fusion:
            self.fuse_1_2 = MiniEncoderFuseDWSep( # NOTE: 64 classes trans output manually input here 
                self.cnn_branch.x_1_2.shape[1], num_output_trans, 64, 1, stage = '1_2')
            self.fuse_1_4 = MiniEncoderFuseDWSep(
                self.cnn_branch.x_1_4.shape[1], num_output_trans, 64, 1, stage='1_4')
            self.fuse_1_8 = MiniEncoderFuseDWSep(
                self.cnn_branch.x_1_8.shape[1], num_output_trans, 64, 1, stage='1_8')
            self.fuse_1_16 = MiniEncoderFuseDWSep(
                self.cnn_branch.x_1_16.shape[1], num_output_trans, 64, 1, stage='1_16')
            if self.patch_size == 32:
                self.fuse_1_32 = MiniEncoderFuseDWSep(
                    self.cnn_branch.x_1_32.shape[1], num_output_trans, 64, 1, stage='1_32')

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)
        x_final_trans = self.trans_branch(images) # 5 x 1 x 156 x 156
        dec_fuse_1_16 = self.conv_fuse_1(self.cnn_branch.x_1_16)
        dec_fuse_1_4 = self.conv_fuse_2(self.cnn_branch.x_1_4)
        x_final_trans = self.decoder_trans(x_final_trans, dec_fuse_1_16, dec_fuse_1_4)

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

class NPZedFusion(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        ):
        super(NPZedFusion, self).__init__()

        tbackbone = trans_model_cfg['backbone']
        assert trans_model_cfg['backbone'] == 'vit_small_patch16_384', \
            f'backbone chosen: {tbackbone}Trying to keep this light: use vit_small_patch16_384'

        self.cnn_branch = zedNet(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            bilinear=True,
            attention=True,
        )
        
        self.cnn_branch.get_dimensions(
            N_in = cnn_model_cfg['batch_size'],
            C_in = cnn_model_cfg['in_channels'],
            H_in = cnn_model_cfg['image_size'][0], 
            W_in = cnn_model_cfg['image_size'][1]
        )

        num_output_trans = trans_model_cfg['num_output_trans']
        print(f'num_output_trans: {num_output_trans}')

        self.output_size = (trans_model_cfg['image_size'][0], trans_model_cfg['image_size'][1])
        trans_model_cfg['image_size'] = (trans_model_cfg['image_size'][0] // 2, trans_model_cfg['image_size'][1] // 2)
        self.trans_entrance_size = trans_model_cfg['image_size']

        self.trans_branch_0_0 = RFB_modified(
            in_channel = self.cnn_branch.x_1_2.shape[1], 
            out_channel = num_output_trans,
        )
        self.trans_branch_DWSep_conv_BNS_1 = nn.Sequential(
            nn.Conv2d(num_output_trans, num_output_trans, kernel_size=3, padding=1, groups=num_output_trans),
            nn.Conv2d(num_output_trans, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.SiLU(inplace=True)
        )
        self.trans_branch_1 = create_transformerV4(trans_model_cfg)

        self.trans_branch_DWSep_conv_BNS_2 = nn.Sequential(
            nn.Conv2d(num_output_trans, num_output_trans, kernel_size=3, padding=1, groups=num_output_trans),
            nn.Conv2d(num_output_trans, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.SiLU(inplace=True)
        )
        self.trans_branch_2 = create_transformerV4(trans_model_cfg)

        # trans decoder stuff 
        in_chans_fuse_1=512
        in_chans_fuse_2=256
        inter_chans=32

        self.conv_fuse_1 = DoubleConvDWSep(in_chans_fuse_1, num_output_trans)
        self.conv_fuse_2 = DoubleConvDWSep(in_chans_fuse_2, inter_chans) 

        self.decoder_trans = DecoderMultiClassDilationAndSCSEFusion(
            input_size=(16,16),
            in_chans=num_output_trans,
            in_chans_fuse_1=num_output_trans,
            in_chans_fuse_2=inter_chans, 
            inter_chans=inter_chans, 
            out_chans=1,
            dilation1=1,
            dilation2=3,
        )

        self.fuse_1_2 = MiniEncoderFuseDWSep( # NOTE: 64 classes trans output manually input here 
            self.cnn_branch.x_1_2.shape[1], num_output_trans, 64, 1, stage = '1_2')
        self.fuse_1_4 = MiniEncoderFuseDWSep(
            self.cnn_branch.x_1_4.shape[1], num_output_trans, 64, 1, stage='1_4')
        self.fuse_1_8 = MiniEncoderFuseDWSep(
            self.cnn_branch.x_1_8.shape[1], num_output_trans, 64, 1, stage='1_8')
        self.fuse_1_16 = MiniEncoderFuseDWSep(
            self.cnn_branch.x_1_16.shape[1], num_output_trans, 64, 1, stage='1_16')

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)

        x_trans = self.trans_branch_0_0(self.cnn_branch.x_1_2)
        x_trans = F.upsample_bilinear(x_trans, size=self.trans_entrance_size)
        x_trans = self.trans_branch_DWSep_conv_BNS_1(x_trans)

        x_trans = self.trans_branch_1(x_trans)
        x_trans = F.upsample_bilinear(x_trans, size=self.trans_entrance_size)
        x_trans = self.trans_branch_DWSep_conv_BNS_2(x_trans)
        x_trans = self.trans_branch_2(x_trans)

        dec_fuse_1_16 = self.conv_fuse_1(self.cnn_branch.x_1_16)
        dec_fuse_1_4 = self.conv_fuse_2(self.cnn_branch.x_1_4)
        x_final_trans = self.decoder_trans(x_trans, dec_fuse_1_16, dec_fuse_1_4)

        self.x_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, F.upsample_bilinear(self.trans_branch_2.x_1_2, scale_factor=2))
        self.x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, F.upsample_bilinear(self.trans_branch_2.x_1_4, scale_factor=2))
        self.x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, F.upsample_bilinear(self.trans_branch_2.x_1_8, scale_factor=2))
        self.x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, F.upsample_bilinear(self.trans_branch_2.x_1_16, scale_factor=2))

        tensor_list = [x_final_cnn, x_final_trans, self.x_1_2, self.x_1_4, self.x_1_8, self.x_1_16]
        mean = torch.mean(torch.stack(tensor_list), dim=0) 
        return mean

class NPZedFusionNoRFB(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        ):
        super(NPZedFusionNoRFB, self).__init__()

        tbackbone = trans_model_cfg['backbone']
        assert trans_model_cfg['backbone'] == 'vit_small_patch16_384', \
            f'backbone chosen: {tbackbone}Trying to keep this light: use vit_small_patch16_384'

        self.cnn_branch = zedNet(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            bilinear=True,
            attention=True,
        )
        
        self.cnn_branch.get_dimensions(
            N_in = cnn_model_cfg['batch_size'],
            C_in = cnn_model_cfg['in_channels'],
            H_in = cnn_model_cfg['image_size'][0], 
            W_in = cnn_model_cfg['image_size'][1]
        )

        num_output_trans = trans_model_cfg['num_output_trans']
        print(f'num_output_trans: {num_output_trans}')

        self.output_size = (trans_model_cfg['image_size'][0], trans_model_cfg['image_size'][1])
        trans_model_cfg['image_size'] = (trans_model_cfg['image_size'][0] // 2, trans_model_cfg['image_size'][1] // 2)
        self.trans_entrance_size = trans_model_cfg['image_size']

        self.trans_branch_0_0 = nn.Sequential(
            nn.Conv2d(self.cnn_branch.x_1_2.shape[1], self.cnn_branch.x_1_2.shape[1], kernel_size=3, padding=1, groups=self.cnn_branch.x_1_2.shape[1]),
            nn.Conv2d(self.cnn_branch.x_1_2.shape[1], num_output_trans, kernel_size=1),
            nn.BatchNorm2d(num_output_trans),
            nn.SiLU(inplace=True)
        )
        self.trans_branch_DWSep_conv_BNS_1 = nn.Sequential(
            nn.Conv2d(num_output_trans, num_output_trans, kernel_size=3, padding=1, groups=num_output_trans),
            nn.Conv2d(num_output_trans, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.SiLU(inplace=True)
        )
        self.trans_branch_1 = create_transformerV4(trans_model_cfg)

        self.trans_branch_DWSep_conv_BNS_2 = nn.Sequential(
            nn.Conv2d(num_output_trans, num_output_trans, kernel_size=3, padding=1, groups=num_output_trans),
            nn.Conv2d(num_output_trans, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.SiLU(inplace=True)
        )
        self.trans_branch_2 = create_transformerV4(trans_model_cfg)

        # trans decoder stuff 
        in_chans_fuse_1=512
        in_chans_fuse_2=256
        inter_chans=32

        self.conv_fuse_1 = DoubleConvDWSep(in_chans_fuse_1, num_output_trans)
        self.conv_fuse_2 = DoubleConvDWSep(in_chans_fuse_2, inter_chans) 

        self.decoder_trans = DecoderMultiClassDilationAndSCSEFusion(
            input_size=(16,16),
            in_chans=num_output_trans,
            in_chans_fuse_1=num_output_trans,
            in_chans_fuse_2=inter_chans, 
            inter_chans=inter_chans, 
            out_chans=1,
            dilation1=1,
            dilation2=3,
        )

        self.fuse_1_2 = MiniEncoderFuseDWSep( # NOTE: 64 classes trans output manually input here 
            self.cnn_branch.x_1_2.shape[1], num_output_trans, 64, 1, stage = '1_2')
        self.fuse_1_4 = MiniEncoderFuseDWSep(
            self.cnn_branch.x_1_4.shape[1], num_output_trans, 64, 1, stage='1_4')
        self.fuse_1_8 = MiniEncoderFuseDWSep(
            self.cnn_branch.x_1_8.shape[1], num_output_trans, 64, 1, stage='1_8')
        self.fuse_1_16 = MiniEncoderFuseDWSep(
            self.cnn_branch.x_1_16.shape[1], num_output_trans, 64, 1, stage='1_16')

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)

        x_trans = self.trans_branch_0_0(self.cnn_branch.x_1_2)
        x_trans = F.upsample_bilinear(x_trans, size=self.trans_entrance_size)
        x_trans = self.trans_branch_DWSep_conv_BNS_1(x_trans)

        x_trans = self.trans_branch_1(x_trans)
        x_trans = F.upsample_bilinear(x_trans, size=self.trans_entrance_size)
        x_trans = self.trans_branch_DWSep_conv_BNS_2(x_trans)
        x_trans = self.trans_branch_2(x_trans)

        dec_fuse_1_16 = self.conv_fuse_1(self.cnn_branch.x_1_16)
        dec_fuse_1_4 = self.conv_fuse_2(self.cnn_branch.x_1_4)
        x_final_trans = self.decoder_trans(x_trans, dec_fuse_1_16, dec_fuse_1_4)

        self.x_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, F.upsample_bilinear(self.trans_branch_2.x_1_2, scale_factor=2))
        self.x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, F.upsample_bilinear(self.trans_branch_2.x_1_4, scale_factor=2))
        self.x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, F.upsample_bilinear(self.trans_branch_2.x_1_8, scale_factor=2))
        self.x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, F.upsample_bilinear(self.trans_branch_2.x_1_16, scale_factor=2))

        tensor_list = [x_final_cnn, x_final_trans, self.x_1_2, self.x_1_4, self.x_1_8, self.x_1_16]
        mean = torch.mean(torch.stack(tensor_list), dim=0) 
        return mean

class SingleTransformerZedFusion(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        ):
        super(SingleTransformerZedFusion, self).__init__()

        assert trans_model_cfg['backbone'] == 'vit_base_patch16_384', \
            f'were only use the big one for now'
        
        self.cnn_branch = zedNet(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            bilinear=True,
            attention=True,
        )
        
        self.cnn_branch.get_dimensions(
            N_in = cnn_model_cfg['batch_size'],
            C_in = cnn_model_cfg['in_channels'],
            H_in = cnn_model_cfg['image_size'][0], 
            W_in = cnn_model_cfg['image_size'][1]
        )

        num_output_trans = trans_model_cfg['num_output_trans']
        print(f'num_output_trans: {num_output_trans}')

        self.output_size = (trans_model_cfg['image_size'][0], trans_model_cfg['image_size'][1])

        self.trans_branch_DWSep_conv_BNS_0 = nn.Sequential(
            nn.Conv2d(self.cnn_branch.x_1_2.shape[1], self.cnn_branch.x_1_2.shape[1], kernel_size=3, padding=1, groups=self.cnn_branch.x_1_2.shape[1]),
            nn.Conv2d(self.cnn_branch.x_1_2.shape[1], num_output_trans, kernel_size=1),
            nn.BatchNorm2d(num_output_trans),
            nn.SiLU(inplace=True)
        )
        self.trans_branch_DWSep_conv_BNS_1 = nn.Sequential(
            nn.Conv2d(num_output_trans, num_output_trans, kernel_size=3, padding=1, groups=num_output_trans),
            nn.Conv2d(num_output_trans, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.SiLU(inplace=True)
        )
        self.trans_branch_1 = create_transformerV4(trans_model_cfg)

        # trans decoder stuff 
        in_chans_fuse_1=512
        in_chans_fuse_2=256
        inter_chans=32

        self.conv_fuse_1 = DoubleConvDWSep(in_chans_fuse_1, num_output_trans)
        self.conv_fuse_2 = DoubleConvDWSep(in_chans_fuse_2, inter_chans) 

        self.decoder_trans = DecoderMultiClassDilationAndSCSEFusion(
            input_size=(16,16),
            in_chans=num_output_trans,
            in_chans_fuse_1=num_output_trans,
            in_chans_fuse_2=inter_chans, 
            inter_chans=inter_chans, 
            out_chans=1,
            dilation1=1,
            dilation2=3,
        )

        self.fuse_1_2 = MiniEncoderFuseDWSep( # NOTE: 64 classes trans output manually input here 
            self.cnn_branch.x_1_2.shape[1], num_output_trans, 64, 1, stage = '1_2')
        self.fuse_1_4 = MiniEncoderFuseDWSep(
            self.cnn_branch.x_1_4.shape[1], num_output_trans, 64, 1, stage='1_4')
        self.fuse_1_8 = MiniEncoderFuseDWSep(
            self.cnn_branch.x_1_8.shape[1], num_output_trans, 64, 1, stage='1_8')
        self.fuse_1_16 = MiniEncoderFuseDWSep(
            self.cnn_branch.x_1_16.shape[1], num_output_trans, 64, 1, stage='1_16')

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)

        x_trans = self.trans_branch_DWSep_conv_BNS_0(self.cnn_branch.x_1_2)
        x_trans = F.upsample_bilinear(x_trans, size=self.output_size)
        x_trans = self.trans_branch_DWSep_conv_BNS_1(x_trans)
        x_trans = self.trans_branch_1(x_trans)

        dec_fuse_1_16 = self.conv_fuse_1(self.cnn_branch.x_1_16)
        dec_fuse_1_4 = self.conv_fuse_2(self.cnn_branch.x_1_4)
        x_final_trans = self.decoder_trans(x_trans, dec_fuse_1_16, dec_fuse_1_4)

        self.x_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, self.trans_branch_1.x_1_2)
        self.x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans_branch_1.x_1_4)
        self.x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans_branch_1.x_1_8)
        self.x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans_branch_1.x_1_16)

        tensor_list = [x_final_cnn, x_final_trans, self.x_1_2, self.x_1_4, self.x_1_8, self.x_1_16]
        mean = torch.mean(torch.stack(tensor_list), dim=0) 
        return mean

class NewZedFusionAttentionTransDecoderDWSepCNNWithCCMFuse(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        with_fusion=True,
        ):
        super(NewZedFusionAttentionTransDecoderDWSepCNNWithCCMFuse, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16 or self.patch_size == 32, \
            'patch_size must be {16, 32}'

        self.cnn_branch = zedNetDWSep(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            bilinear=True,
            attention=True,
        )
        # now populate dimensions 
        self.cnn_branch.get_dimensions(
            N_in = cnn_model_cfg['batch_size'],
            C_in = cnn_model_cfg['in_channels'],
            H_in = cnn_model_cfg['image_size'][0], 
            W_in = cnn_model_cfg['image_size'][1]
        )

        self.trans_branch = create_transformerV4(trans_model_cfg, 
            decoder='linear') # output should now be [N, num_output_trans, 16, 16]

        num_output_trans = trans_model_cfg['num_output_trans']
        print(f'num_output_trans: {num_output_trans}')

        in_chans_fuse_1=512
        in_chans_fuse_2=256
        inter_chans=32

        self.conv_fuse_1 = BNRconv3x3(in_chans_fuse_1, num_output_trans)
        self.conv_fuse_2 = BNRconv3x3(in_chans_fuse_2, inter_chans)

        self.decoder_trans = DecoderMultiClassDilationAndSCSEFusion(
            input_size=(16,16),
            in_chans=num_output_trans,
            in_chans_fuse_1=num_output_trans,
            in_chans_fuse_2=inter_chans, 
            inter_chans=inter_chans, 
            out_chans=1,
            dilation1=1,
            dilation2=3,
        )
        

        # num_output_trans = 64

        self.with_fusion = with_fusion
        if self.with_fusion:
            self.fuse_1_2 = CCMFusionModule( # NOTE: 64 classes trans output manually input here 
                self.cnn_branch.x_1_2.shape[1], num_output_trans, 64, 1, stage = '1_2')
            self.fuse_1_4 = CCMFusionModule(
                self.cnn_branch.x_1_4.shape[1], num_output_trans, 64, 1, stage='1_4')
            self.fuse_1_8 = CCMFusionModule(
                self.cnn_branch.x_1_8.shape[1], num_output_trans, 64, 1, stage='1_8')
            self.fuse_1_16 = CCMFusionModule(
                self.cnn_branch.x_1_16.shape[1], num_output_trans, 64, 1, stage='1_16')
            if self.patch_size == 32:
                self.fuse_1_32 = CCMFusionModule(
                    self.cnn_branch.x_1_32.shape[1], num_output_trans, 64, 1, stage='1_32')

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)
        x_final_trans = self.trans_branch(images) # 5 x 1 x 156 x 156
        dec_fuse_1_16 = self.conv_fuse_1(self.cnn_branch.x_1_16)
        dec_fuse_1_4 = self.conv_fuse_2(self.cnn_branch.x_1_4)
        x_final_trans = self.decoder_trans(x_final_trans, dec_fuse_1_16, dec_fuse_1_4)

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