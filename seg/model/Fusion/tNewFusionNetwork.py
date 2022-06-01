import torch
import torch.nn as nn 
import torch.nn.functional as F 

import yaml
from pathlib import Path
from seg.model.CNN.CNN import CNN_BRANCH
from seg.model.alt_cnns.pranetSimple import RFB_modified
from seg.model.general.DW_sep import SeparableConv2D
from seg.model.transformer.transformerV3 import create_transformerV3

from seg.model.zed.zedNet import zedNet, zedNetDWSep, zedNetDWSepWithCCMAndRFB, zedNetMod, zedNetDWSepWithCCM, zedNetDWSepWithCCMinAllOfIt, zedNetDWSepWithCCMmodeddedFromBest
from seg.model.transformer.create_modelV2 import create_transformerV2
from seg.model.Fusion.fuse import SimpleFusion
from seg.utils.check_parameters import count_parameters
from .fuse import CCMFusionModule, MiniEncoderFuse, MiniEncoderFuseDWSep, MiniEncoderFuseDWSepRFB

class PatchAggregation(nn.Module):
    def __init__(
        self,
        image_size = (512, 512),
        PS=16,
        num_tensors=5,
        divide_by_sum=True,
    ):
        super(PatchAggregation, self).__init__()
        self.image_size = image_size
        self.PS = PS
        self.num_tensors = num_tensors
        self.divide_by_sum = divide_by_sum

        self.grid_size = image_size[0] // PS, image_size[1] // PS
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_weights = self.num_patches * self.num_tensors

        self.weights = nn.Parameter(
            torch.randn(
                1,
                self.num_tensors, 
                self.grid_size[0], 
                self.grid_size[1]
                ),
                requires_grad=True
            )
        self.upsample = nn.Upsample(size=self.image_size, mode='nearest')
    
    def forward(self, tensors):
        upsampled_weights = self.upsample(self.weights)
        output = tensors * upsampled_weights
        output = torch.sum(output, dim=1).unsqueeze(dim=1)
        if self.divide_by_sum:
            weighted_sum = upsampled_weights.sum(dim=1)
            output = torch.div(output, weighted_sum)
            return output
        return output

class NewZedFusionNetworkDWSepWithCCMinDWModuleWithPatchAggregation(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        aggregate_patch_size=16,
        divide_by_sum=True,
        ):
        super(NewZedFusionNetworkDWSepWithCCMinDWModuleWithPatchAggregation, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16 or self.patch_size == 32, \
            'patch_size must be {16, 32}'

        self.cnn_branch = zedNetDWSepWithCCM(
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

        self.fuse_1_2 = MiniEncoderFuseDWSep( # NOTE: 64 classes trans output manually input here 
            self.cnn_branch.x_1_2.shape[1], num_output_trans, 64, 1, stage = '1_2')
        self.fuse_1_4 = MiniEncoderFuseDWSep(
            self.cnn_branch.x_1_4.shape[1], num_output_trans, 64, 1, stage='1_4')
        self.fuse_1_8 = MiniEncoderFuseDWSep(
            self.cnn_branch.x_1_8.shape[1], num_output_trans, 64, 1, stage='1_8')
        self.fuse_1_16 = MiniEncoderFuseDWSep(
            self.cnn_branch.x_1_16.shape[1], num_output_trans, 64, 1, stage='1_16')
    
        self.agg = PatchAggregation(
            image_size=cnn_model_cfg['image_size'], 
            PS=aggregate_patch_size, 
            num_tensors=6, 
            divide_by_sum=divide_by_sum,
        )

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)
        x_final_trans = self.trans_branch(images) # 5 x 1 x 156 x 156

        self.x_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, self.trans_branch.x_1_2)
        self.x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans_branch.x_1_4)
        self.x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans_branch.x_1_8)
        self.x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans_branch.x_1_16)

        tensor_list = [x_final_cnn, x_final_trans, self.x_1_2, self.x_1_4, self.x_1_8, self.x_1_16]
        output = self.agg(torch.cat(tensor_list, dim=1))
        return output


if __name__ == '__main__':
    batch_size = 5
    image_size = (512, 512)
    patch_size = 16
    num_tensors = 4
    tlist = list()
    for i in range(num_tensors):
        tlist.append(torch.randn((batch_size, 1, image_size[0], image_size[1]), device='cuda'))
    
    x = torch.cat(tlist, dim=1)

    mod = PatchAggregation(
        image_size = image_size,
        PS = patch_size,
        num_tensors=num_tensors
    ).cuda()

    y = mod(x)
    print(f'y.shape: {y.shape}')

    weights = torch.randn((1, 2, 2, 2))
    print(f'weights:\n{weights}')
    weight_sum = weights.sum(dim=1)
    print(f'weighted_sum:\n{weight_sum}')

    count_parameters(mod)