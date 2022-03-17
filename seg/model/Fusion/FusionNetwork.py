import torch
import torch.nn as nn 
import yaml
from pathlib import Path

from seg.model.CNN.CNN import CNN_BRANCH
from seg.model.CNN.CNN_backboned import CNN_BRANCH_WITH_BACKBONE
from seg.model.segmenter.create_model import create_transformer
from seg.model.Fusion.fuse import SimpleFusion
from seg.model.siddnet.siddnet import CBR, SuperficialModule_subblock, CCMSubBlock
from .fuse import MiniEncoderFuse

class OldFusionNetwork(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        cnn_pretrained=False,
        with_fusion=True,
        ):
        super(OldFusionNetwork, self).__init__()

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
                pretrained=True,
                with_fusion=True,
                input_size=cnn_model_cfg['image_size'][0],
            )
        else:
            self.cnn_branch = CNN_BRANCH(
                n_channels=cnn_model_cfg['in_channels'],
                n_classes=cnn_model_cfg['num_classes'],
                patch_size=cnn_model_cfg['patch_size'],
                bilinear=True
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
        self.trans_branch = create_transformer(trans_model_cfg, 
            decoder='linear')

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
        x_final_trans = self.trans_branch(images)
        '''
        self.CNN_BRANCH and self.TRANSFORMER_BRANCH should have same members:
                { output_1_4, output_1_2 }
        '''
        if self.with_fusion:
            # 1 / 2 - note (kind of wack given that you have to interploate from 1/4)
            x_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, self.trans_branch.x_1_2)
            x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans_branch.x_1_4)
            x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans_branch.x_1_8)
            x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans_branch.x_1_16)

            if self.patch_size == 16:
                tensor_list = [x_final_cnn, x_final_trans, x_1_2, x_1_4, x_1_8, x_1_16]
                mean = torch.mean(torch.stack(tensor_list), dim=0) 
                return mean
            elif self.patch_size == 32:
                x_1_32 = self.fuse_1_32(self.cnn_branch.x_1_32, self.trans_branch.x_1_32)
                tensor_list = [x_final_cnn, x_final_trans, x_1_2, x_1_4, x_1_8, x_1_16, x_1_32]
                mean = torch.mean(torch.stack(tensor_list), dim=0) 
                return mean

class SimpleFusionNetwork(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
    ):
        super(SimpleFusionNetwork, self).__init__()
        if cnn_model_cfg['backbone'] is None:
            print('Excellent, the cnn_model_cfg has been set up correctly now.')
        else:
            raise ValueError(f'UNet with backbone not supported yet.')

        self.cnn_branch = CNN_BRANCH(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
            bilinear=True
        )

        print(f'Warning in file: {__file__}, we are manually assigning the \
decoder to have a `linear` value in create_transformer when creating the \
fusion network and thus not using the decoder value input to main() in \
train.py, but im too tired to try and figure out how to work that and were \
running the terminal right now so...') # SEE BELOW.... decoder = 'linear'
        # need to do something or pull information from the trans_model_cfg and
        #  pull that info. but yeah. wahtever rn lol 
        self.trans_branch = create_transformer(trans_model_cfg, 
            decoder='linear')

        assert trans_model_cfg['patch_size'] == cnn_model_cfg['patch_size']
        self.patch_size = trans_model_cfg['patch_size']
        self.fuse_1_2 = SimpleFusion(128, 1, out_chan=1, fuse_stage='1_2')
        self.fuse_1_4 = SimpleFusion(256, 1, out_chan=1, fuse_stage='1_4')
        self.fuse_1_8 = SimpleFusion(512, 1, out_chan=1, fuse_stage='1_8')
        self.fuse_1_16 = SimpleFusion(512, 1, out_chan=1, fuse_stage='1_16')
        if self.patch_size == 32:
            self.fuse_1_32 = SimpleFusion(512, 1, out_chan=1, fuse_stage='1_32')
    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)
        x_final_trans = self.trans_branch(images)
        x_fused_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, self.trans_branch.x_1_2)
        x_fused_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans_branch.x_1_4)
        x_fused_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans_branch.x_1_8)
        x_fused_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans_branch.x_1_16)
        if self.patch_size == 32:
            x_fused_1_32 = self.fuse_1_32(self.cnn_branch.x_1_32, self.trans_branch.x_1_32)
            tensor_list = [x_final_cnn, x_final_trans, x_fused_1_2, x_fused_1_4, x_fused_1_8, x_fused_1_16, x_fused_1_32]
        else:
            tensor_list = [x_final_cnn, x_final_trans, x_fused_1_2, x_fused_1_4, x_fused_1_8, x_fused_1_16]
        mean = torch.mean(torch.stack(tensor_list), dim=0) 
        return mean 


class SimplestFusionNetwork(nn.Module):
    def __init__(
        self,
        cnn_model_cfg,
        trans_model_cfg,
        with_weights=False,
    ):
        super(SimplestFusionNetwork, self).__init__()

        # if cnn_model_cfg['backbone'] is None:
        #     print('Excellent, the cnn_model_cfg has been set up correctly now.')
        # else:
        #     raise ValueError(f'UNet with backbone not supported yet.')

        self.cnn_branch = CNN_BRANCH_WITH_BACKBONE(
            n_channels = cnn_model_cfg['in_channels'],
            n_classes = cnn_model_cfg['num_classes'],
            patch_size = cnn_model_cfg['patch_size'],
            backbone_name=cnn_model_cfg['backbone'],
            bilinear=True, 
            pretrained=True,
        )

        print(f'Warning in file: {__file__}, we are manually assigning the \
decoder to have a `linear` value in create_transformer when creating the \
fusion network and thus not using the decoder value input to main() in \
train.py, but im too tired to try and figure out how to work that and were \
running the terminal right now so...') # SEE BELOW.... decoder = 'linear'
        # need to do something or pull information from the trans_model_cfg and
        #  pull that info. but yeah. wahtever rn lol 
        print(f'\nLoading transformer, pretrained in {__file__}')
        self.trans_branch = create_transformer(trans_model_cfg, 
            decoder='linear')


        # learnable parameters - after trying with just dim 1, try modifying it so the weights are the size of the whole seg map
        self.with_weights = with_weights
        if self.with_weights:
            self.w_cnn = nn.Parameter(torch.rand(1))
            self.w_trans = nn.Parameter(torch.rand(1))
            self.w_cnn.requires_grad = True
            self.w_trans.requires_grad = True

    def forward(self, x):
        x_cnn = self.cnn_branch(x)
        x_trans = self.trans_branch(x)
        
        # model weights multiplication
        if self.with_weights:
            x_cnn = x_cnn * self.w_cnn
            x_trans = x_trans * self.w_trans

        mean = torch.mean(torch.stack([x_cnn, x_trans]), dim=0) 
        return mean 

# class SimplestFusionNetworkSiddNet(nn.Module):
#     def __init__(
#         self,
#         cnn_model_cfg,
#         trans_model_cfg,
#     ):
#         super(SimplestFusionNetworkSiddNet, self).__init__()

#         # self.cnn_branch = 

if __name__ == '__main__':

    cfg = yaml.load(open(Path(__file__).parent / "cnn_config.yml", "r"), 
        Loader=yaml.FullLoader)
    cnn_model_name = 'unet'
    model_cfg = cfg['model'][cnn_model_name]
    print(f'model_cfg:\n {model_cfg}')

