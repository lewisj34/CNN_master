import torch
import torch.nn as nn 
import yaml
from pathlib import Path

from seg.model.CNN.CNN import CNN_BRANCH
from seg.model.CNN.CNN_backboned import CNN_BRANCH_WITH_BACKBONE
from seg.model.transformer.create_model import create_transformer
from seg.model.transformer.create_modelV2 import create_transformerV2
from seg.model.Fusion.fuse import SimpleFusion
from .fuse import MiniEncoderFuse

class NewFusionNetworkWeight(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        cnn_pretrained=False,
        with_fusion=True,
        with_aspp=False,
        ):
        super(NewFusionNetworkWeight, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16 or self.patch_size == 32, \
            'patch_size must be {16, 32}'
        self.cnn_pretrained = cnn_pretrained
        if self.cnn_pretrained:
            # just a couple checks 
            assert cnn_model_cfg['image_size'][0] == cnn_model_cfg['image_size'][1], \
                'image_height and width must be the same' 
            assert cnn_model_cfg['image_size'][0] == 256 or \
                cnn_model_cfg['image_size'][0] == 512, 'self explanatory'

            self.cnn_branch = CNN_BRANCH_WITH_BACKBONE(
                n_channels=cnn_model_cfg['in_channels'],
                n_classes=cnn_model_cfg['num_classes'],
                patch_size=cnn_model_cfg['patch_size'],
                backbone_name=cnn_model_cfg['backbone'],
                bilinear=True,
                pretrained=self.cnn_pretrained,
                with_fusion=True,
                input_size=cnn_model_cfg['image_size'][0],
            )
        else:
            self.cnn_branch = CNN_BRANCH(
                n_channels=cnn_model_cfg['in_channels'],
                n_classes=cnn_model_cfg['num_classes'],
                patch_size=cnn_model_cfg['patch_size'],
                use_ASPP=with_aspp,
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
        self.x_final_cnn = self.cnn_branch(images)
        self.x_final_trans = self.trans_branch(images) # 5 x 1 x 156 x 156

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
                tensor_list = [self.x_final_cnn, self.x_final_trans, self.x_1_2, self.x_1_4, self.x_1_8, self.x_1_16]
                mean = torch.mean(torch.stack(tensor_list), dim=0) 
                return mean
            elif self.patch_size == 32:
                raise ValueError(f'PS: 32 not supported.')