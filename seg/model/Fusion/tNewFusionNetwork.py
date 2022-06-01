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

class WeightedAvg(nn.Module):
    def __init__(
        self,
        num_tensors,
    ):
        super().__init__()
        # using einsum
        self.w = nn.Parameter(torch.rand(num_tensors))

        # using linear 
        # self.wAvgLayer = nn.Linear(num_tensors, 1)     

    def forward(self, tensor_list: list) -> torch.Tensor:
        # using the linear layer version 
        # print(f'self.w: {self.wAvgLayer.weight}')
        # res = self.wAvgLayer(torch.stack(tensor_list, dim=-1))
        # return res

        # using the einsum verison 
        print(f'w: {self.w}')
        res = torch.einsum('ik,kj->ij', torch.stack(tensor_list), self.w)
        return res

class Crazy(nn.Module):
    def __init__(
        self,
        image_size=(512, 512),
        PS=16,
        num_tensors=5, # number of tensors to be considered
    ):
        super().__init__()
        assert image_size[0] % PS == 0 and image_size[1] % PS == 0, \
            f'H, W must be divisible by PS. Given: {image_size[0], image_size[1], PS}'
        
        self.num_tensors = num_tensors
        self.grid_size = image_size[0] // PS, image_size[1] // PS
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_weights = self.num_patches * num_tensors

        print(f'GS: {self.grid_size}')
        print(f'num_patches: {self.num_patches}')
        print(f'num_weights: {self.num_weights}')
        
        self.proj = nn.Conv2d(in_channels=num_tensors, out_channels=num_tensors, kernel_size=PS, stride=PS)

        self.w = nn.Parameter(torch.randn(self.num_tensors, self.grid_size[0], self.grid_size[1]))

    def forward(self, x):
        print(f'w: {self.w.shape}')

        # x = torch.einsum('ik,kj->ij', [self.w])
        return self.proj(x)
        
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        print(f'proj weights: {self.proj.weight.shape}')
        return x


if __name__ == '__main__':
    # num_tensors = 5 
    # tList = list()
    # for i in range(num_tensors):
    #     tList.append(torch.randn((5, 1, 256, 256), device='cuda'))

    # wAvg = WeightedAvg(num_tensors).cuda()

    # res = wAvg(tList)

    # print(f'res: {res.shape}')

    image_size = (256, 256)
    num_tensors=5
    tlist = list()
    for i in range(num_tensors):
        tlist.append(torch.randn((1, 1, image_size[0], image_size[1]), device='cuda'))
    
    x = torch.cat(tlist, dim=1)

    c = Crazy(
        image_size=image_size,
        PS = 128, 
        num_tensors=5,
    ).cuda()

    y = c(x)

    print(f'y.shape:{y.shape}')

    tensor_shape = (1, 224, 224)  # shape of each tensor
    five_tensors = torch.randn(5, *tensor_shape, requires_grad=True)
    weights = torch.rand(5, requires_grad=True)
    weighted_avg = (weights.view(5, 1, 1, 1) * five_tensors).sum(dim=0) / weights.sum()
    print(f'five_tensors.shape: {five_tensors.shape}')
    print((five_tensors[:, 0, 100, 100] * weights).sum() / weights.sum())
    print(weighted_avg[0, 100, 100])
    print(f'weighted_avg.shape: {weighted_avg.shape}')
