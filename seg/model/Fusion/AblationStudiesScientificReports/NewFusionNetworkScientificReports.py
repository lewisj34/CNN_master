"""
For the ViT ablation studies requested by the Scientific Reports editorial staff
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F 

import yaml
from pathlib import Path
from seg.model.CNN.CNN import CNN_BRANCH
from seg.model.Fusion.CondensedFusion import BNRconv3x3
from seg.model.Fusion.newCNN import xCNN_v2
from seg.model.alt_cnns.pranetSimple import RFB_modified
from seg.model.general.DW_sep import SeparableConv2D
from seg.model.transformer.decoder_new import DecoderMultiClassDilationAndSCSE, DecoderMultiClassDilationAndSCSEFusion, DecoderMultiClassDilationAndSCSEFusionJustOne, DecoderMultiClassDilationAndSCSEReduced, DecoderMultiClassDilationWithSingleSeparableSqueezeandExitationBlock, UpModDilatedDWSep, UpModDilated
from seg.model.transformer.transformerV3 import create_transformerV3

from seg.model.zed.zedNet import zedNet, zedNetDWSep, zedNetDWSepWithCCMAndRFB, zedNetMod, zedNetDWSepWithCCM, zedNetDWSepWithCCMinAllOfIt, zedNetDWSepWithCCMmodeddedFromBest
from seg.model.transformer.create_model import create_transformer
from seg.model.transformer.create_modelV2 import create_transformerV2
from seg.model.Fusion.fuse import SimpleFusion
from ..fuse import CCMFusionModule, MiniEncoderFuse, MiniEncoderFuseDWSep, MiniEncoderFuseDWSepRFB

from .unet_model import UNet

class AblationStudyScientificReportsNoViTUNetReplace(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        with_fusion=True,
        ):
        super(AblationStudyScientificReportsNoViTUNetReplace, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16 or self.patch_size == 32, \
            'patch_size must be {16, 32}'

        self.cnn_branch = zedNetDWSepWithCCMinAllOfIt(
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
        self.trans_branch = UNet(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            bilinear=True,
        )
        self.trans_branch.get_dimensions(
            N_in = cnn_model_cfg['batch_size'],
            C_in = cnn_model_cfg['in_channels'],
            H_in = cnn_model_cfg['image_size'][0], 
            W_in = cnn_model_cfg['image_size'][1]
        )
        
        num_output_trans = trans_model_cfg['num_output_trans']
        print(f'num_output_trans: {num_output_trans}')
        # num_output_trans = 64

        self.with_fusion = with_fusion
        if self.with_fusion:
            self.fuse_1_2 = MiniEncoderFuseDWSep( # NOTE: 64 classes trans output manually input here 
                self.cnn_branch.x_1_2.shape[1], self.trans_branch.x_1_2.shape[1], 64, 1, stage = '1_2')
            self.fuse_1_4 = MiniEncoderFuseDWSep(
                self.cnn_branch.x_1_4.shape[1], self.trans_branch.x_1_4.shape[1], 64, 1, stage='1_4')
            self.fuse_1_8 = MiniEncoderFuseDWSep(
                self.cnn_branch.x_1_8.shape[1], self.trans_branch.x_1_8.shape[1], 64, 1, stage='1_8')
            self.fuse_1_16 = MiniEncoderFuseDWSep(
                self.cnn_branch.x_1_16.shape[1], self.trans_branch.x_1_16.shape[1], 64, 1, stage='1_16')
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