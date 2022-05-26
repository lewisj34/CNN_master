"""
I DONT THINK THIS IS ACTUALLY ANY DIFFERENT FROM OLDFUSIONNETWORK, I MADE THIS FILE on APRIL 19TH 
AND DOESNT REFLECT THE ADDITION OF CREATE_TRANSFORMER_V2
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F 

import yaml
from pathlib import Path
from seg.model.CNN.CNN import CNN_BRANCH
from seg.model.Fusion.CondensedFusion import BNRconv3x3
from seg.model.alt_cnns.pranetSimple import RFB_modified
from seg.model.general.DW_sep import SeparableConv2D
from seg.model.transformer.decoder_new import DecoderMultiClassDilationAndSCSE, DecoderMultiClassDilationAndSCSEFusion, DecoderMultiClassDilationAndSCSEFusionJustOne, DecoderMultiClassDilationAndSCSEReduced, UpModDilatedDWSep, UpModDilated
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
from seg.model.zed.parts import DoubleConvDWSep, UpDWSep

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
                self.cnn_branch.x_1_2.shape[1], num_output_trans, 64, 4, stage = '1_2')
            self.up_1_2 = UpModDilated(4, 1, True, scale_factor=2, dilation=4)

            self.fuse_1_4 = CCMFusionModule(
                self.cnn_branch.x_1_4.shape[1], num_output_trans, 64, 8, stage='1_4')
            self.up_1_4_0 = UpModDilated(8, 4, True, scale_factor=2, dilation=2)
            self.up_1_4_1 = UpModDilated(4, 1, True, scale_factor=2, dilation=3)

            self.fuse_1_8 = CCMFusionModule(
                self.cnn_branch.x_1_8.shape[1], num_output_trans, 64, 16, stage='1_8')
            self.up_1_8_0 = UpModDilated(16, 8, True, scale_factor=2, dilation=1)
            self.up_1_8_1 = UpModDilated(8, 1, True, scale_factor=4, dilation=2)

            self.fuse_1_16 = CCMFusionModule(
                self.cnn_branch.x_1_16.shape[1], num_output_trans, 64, 32, stage='1_16')
            self.up_1_16_0 = UpModDilated(32, 16, True, scale_factor=4, dilation=1)
            self.up_1_16_1 = UpModDilated(16, 1, True, scale_factor=4, dilation=1)

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
            self.x_1_2 = self.up_1_2(self.x_1_2)
            self.x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans_branch.x_1_4)
            self.x_1_4 = self.up_1_4_0(self.x_1_4)
            self.x_1_4 = self.up_1_4_1(self.x_1_4)
            self.x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans_branch.x_1_8)
            self.x_1_8 = self.up_1_8_0(self.x_1_8)
            self.x_1_8 = self.up_1_8_1(self.x_1_8)
            self.x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans_branch.x_1_16)
            self.x_1_16 = self.up_1_16_0(self.x_1_16)
            self.x_1_16 = self.up_1_16_1(self.x_1_16)


            if self.patch_size == 16:
                tensor_list = [x_final_cnn, x_final_trans, self.x_1_2, self.x_1_4, self.x_1_8, self.x_1_16]
                mean = torch.mean(torch.stack(tensor_list), dim=0) 
                return mean

class SingleTransformerZedFusionV2(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        ):
        super(SingleTransformerZedFusionV2, self).__init__()

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

        # entrance flow 
        self.enFlow1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3),
            nn.Conv2d(3, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.SiLU(True),
        )
        self.enFlow2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.SiLU(True),
        )

        self.trans_branch_1 = create_transformerV4(trans_model_cfg)

        # trans decoder stuff 
        in_chans_fuse_1=512
        in_chans_fuse_2=512
        inter_chans=32
        fuse_chans = num_output_trans // 2

        self.conv_fuse_1 = DoubleConvDWSep(in_chans_fuse_1, fuse_chans)

        self.decoder_trans = DecoderMultiClassDilationAndSCSEFusionJustOne(
            input_size=(16,16),
            in_chans=num_output_trans,
            in_chans_fuse_1=fuse_chans,
            inter_chans=inter_chans,
            out_chans=1,
            dilation1=1,
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
        images = self.enFlow1(images)
        images = self.enFlow2(images)

        x_final_cnn = self.cnn_branch(images)

        x_trans = self.trans_branch_1(images)

        dec_fuse_1_16 = self.conv_fuse_1(self.cnn_branch.x_1_16)
        x_final_trans = self.decoder_trans(x_trans, dec_fuse_1_16)

        self.x_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, self.trans_branch_1.x_1_2)
        self.x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans_branch_1.x_1_4)
        self.x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans_branch_1.x_1_8)
        self.x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans_branch_1.x_1_16)

        tensor_list = [x_final_cnn, x_final_trans, self.x_1_2, self.x_1_4, self.x_1_8, self.x_1_16]
        mean = torch.mean(torch.stack(tensor_list), dim=0) 
        return mean

class SingleTransformerZedFusionV2NoAttentionDecoder(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        ):
        super(SingleTransformerZedFusionV2NoAttentionDecoder, self).__init__()

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

        # entrance flow 
        self.enFlow1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3),
            nn.Conv2d(3, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.SiLU(True),
        )
        self.enFlow2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.SiLU(True),
        )

        self.trans_branch_1 = create_transformerV4(trans_model_cfg)

        self.decoder_trans = DecoderMultiClassDilationAndSCSEReduced(
            in_chans=num_output_trans,
            inter_chans=32,
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
        images = self.enFlow1(images)
        images = self.enFlow2(images)

        x_final_cnn = self.cnn_branch(images)

        x_trans = self.trans_branch_1(images)

        x_final_trans = self.decoder_trans(x_trans)

        self.x_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, self.trans_branch_1.x_1_2)
        self.x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans_branch_1.x_1_4)
        self.x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans_branch_1.x_1_8)
        self.x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans_branch_1.x_1_16)

        tensor_list = [x_final_cnn, x_final_trans, self.x_1_2, self.x_1_4, self.x_1_8, self.x_1_16]
        mean = torch.mean(torch.stack(tensor_list), dim=0) 
        return mean

from seg.model.general.invert import NegativeInvert
class ReverseAttentionCNNandlittleandBigTransformer(nn.Module):
    def __init__(
        self,
        cnn_model_cfg,
        big_trans_model_cfg,
        sml_trans_model_cfg,
        decoder_cfg,
        trans_model_cfg_copy=None, # not actually used for anything outside of copying other params
        num_output_trans_big=64,
        num_output_trans_sml=1,
    ):
        super().__init__()
        # import model details for transformer 

        assert trans_model_cfg_copy is not None, \
            f'This is just the main trans_model_cfg value given in train.py'
        
        big_trans_model_cfg['image_size'] = trans_model_cfg_copy['image_size']
        big_trans_model_cfg["dropout"] = trans_model_cfg_copy['dropout']
        big_trans_model_cfg["drop_path_rate"] = trans_model_cfg_copy['drop_path_rate']
        big_trans_model_cfg['n_cls'] = trans_model_cfg_copy['n_cls']
        big_trans_model_cfg['decoder'] = decoder_cfg
        big_trans_model_cfg['num_output_trans'] = num_output_trans_big
        decoder_cfg['name'] = 'linear' 

        sml_trans_model_cfg['image_size'] = trans_model_cfg_copy['image_size']
        sml_trans_model_cfg["dropout"] = trans_model_cfg_copy['dropout']
        sml_trans_model_cfg["drop_path_rate"] = trans_model_cfg_copy['drop_path_rate']
        sml_trans_model_cfg['n_cls'] = trans_model_cfg_copy['n_cls']
        sml_trans_model_cfg['decoder'] = decoder_cfg
        sml_trans_model_cfg['num_output_trans'] = num_output_trans_sml

        self.big_trans = create_transformerV4(big_trans_model_cfg, decoder='linear')
        self.sml_trans = create_transformerV4(sml_trans_model_cfg, decoder='linear')
        
        self.cnn_branch = CNN_BRANCH(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            bilinear=True,
        )
        self.cnn_branch.get_dimensions(
            N_in = cnn_model_cfg['batch_size'],
            C_in = cnn_model_cfg['in_channels'],
            H_in = cnn_model_cfg['image_size'][0], 
            W_in = cnn_model_cfg['image_size'][1]
        )
        

        negative_inv_expand_factor = 2
        self.inv_x_1_2_cnn = NegativeInvert(
            self.cnn_branch.x_1_2.shape[1], # changing this expansion to either lower or hgiher than the transformer is a hyperparam   
            out_chans = 64,
            scale_factor = 8,
        )
        self.inv_x_1_4_cnn = NegativeInvert(
            self.cnn_branch.x_1_4.shape[1], # changing this expansion to either lower or hgiher than the transformer is a hyperparam 
            out_chans = 128,
            scale_factor = 4
        )
        self.inv_x_1_8_cnn = NegativeInvert(
            self.cnn_branch.x_1_8.shape[1], # changing this expansion to either lower or hgiher than the transformer is a hyperparam 
            out_chans = 256,
            scale_factor = 2,
        )

        # # THIS HASNT BEEN PUIT INTO P{LACCE YET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1}
        # self.inv_x_1_2_trans = NegativeInvert(
        #     64,    
        #     out_chans = 64,
        #     scale_factor = 8,
        # )
        # self.inv_x_1_4_trans = NegativeInvert(
        #     64, 
        #     out_chans = 128,
        #     scale_factor = 4
        # )
        # self.inv_x_1_8_trans = NegativeInvert(
        #     64, 
        #     out_chans = 256,
        #     scale_factor = 2,
        # )

        self.fuse_1_2 = MiniEncoderFuseDWSep(
            64, # this is just the same as the expand channels from self.inv_x_1_2 above
            big_trans_model_cfg['num_output_trans'], 
            64, 
            1,
            stage = '1_2'
        )
        self.fuse_1_4 = MiniEncoderFuseDWSep(
            128, # this is just the same as the expand channels from self.inv_x_1_4 above
            big_trans_model_cfg['num_output_trans'], 
            64, 
            1, 
            stage='1_4'
        )
        self.fuse_1_8 = MiniEncoderFuseDWSep(
            256, # this is just the same as the expand channels from self.inv_x_1_8 above
            big_trans_model_cfg['num_output_trans'], 
            64, 
            1, 
            stage='1_8'
        )
        self.conv16x16 = nn.Sequential(
            SeparableConv2D(self.cnn_branch.x_1_16.shape[1], big_trans_model_cfg['num_output_trans'] * 2, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(big_trans_model_cfg['num_output_trans'] * 2),
            nn.ReLU6(True),
        )
        self.fuse_1_16 = MiniEncoderFuseDWSep(
            big_trans_model_cfg['num_output_trans'] * 2, # this is just the same as the expand channels from self.inv_x_1_16 above
            big_trans_model_cfg['num_output_trans'], 
            64, 
            1, 
            stage='1_16'
        )

        self.decoder_trans = DecoderMultiClassDilationAndSCSEReduced(
            in_chans=big_trans_model_cfg['num_output_trans'],
            inter_chans=32,
            out_chans=1,
            dilation1=1,
            dilation2=3,
        )
        

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)
        x_trans_sml = self.sml_trans(images)

        x_1_2 = self.inv_x_1_2_cnn(x_trans_sml, self.cnn_branch.x_1_2) # ;print(f'[x_1_2]: \t{x_1_2.shape}')
        x_1_4 = self.inv_x_1_4_cnn(x_trans_sml, self.cnn_branch.x_1_4) # ;print(f'[x_1_4]: \t{x_1_4.shape}')
        x_1_8 = self.inv_x_1_8_cnn(x_trans_sml, self.cnn_branch.x_1_8) # ;print(f'[x_1_8]: \t{x_1_8.shape}')

        x_trans_big = self.big_trans(images)

        x_final_trans = self.decoder_trans(x_trans_big)

        # print(f'[cnn_branch.x_1_2     ]:\t {self.cnn_branch.x_1_2.shape}')      # torch.Size([5, 128, 128, 128])
        # print(f'[x_1_2                ]:\t {x_1_2.shape}')                      # torch.Size([5, 64, 128, 128])
        # print(f'[big_trans.x_1_2      ]:\t {self.big_trans.x_1_2.shape}')       # torch.Size([5, 64, 128, 128])
        # print('\n\n')
        # print(f'[cnn_branch.x_1_4     ]:\t {self.cnn_branch.x_1_4.shape}')      # torch.Size([5, 256, 64, 64])
        # print(f'[x_1_4                ]:\t {x_1_4.shape}')                      # torch.Size([5, 128, 64, 64])
        # print(f'[big_trans.x_1_4      ]:\t {self.big_trans.x_1_4.shape}')       # torch.Size([5, 64, 64, 64])
        # print('\n\n')
        # print(f'[cnn_branch.x_1_8     ]:\t {self.cnn_branch.x_1_8.shape}')      # torch.Size([5, 512, 32, 32])
        # print(f'[x_1_8                ]:\t {x_1_8.shape}')                      # torch.Size([5, 256, 32, 32])
        # print(f'[big_trans.x_1_8      ]:\t {self.big_trans.x_1_8.shape}')       # torch.Size([5, 64, 32, 32])

        self.x_1_2 = self.fuse_1_2(x_1_2, self.big_trans.x_1_2)
        self.x_1_4 = self.fuse_1_4(x_1_4, self.big_trans.x_1_4)
        self.x_1_8 = self.fuse_1_8(x_1_8, self.big_trans.x_1_8)
        self.x_1_16 = self.fuse_1_16(self.conv16x16(self.cnn_branch.x_1_16), self.big_trans.x_1_16)

        tensor_list = [x_final_cnn, x_final_trans, self.x_1_2, self.x_1_4, self.x_1_8, self.x_1_16]
        mean = torch.mean(torch.stack(tensor_list), dim=0) 
        return mean
