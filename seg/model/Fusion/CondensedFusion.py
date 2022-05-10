from logging import Logger
import torch
import torch.nn as nn 
import torch.nn.functional as F

from seg.model.CNN.CNN import CNN_BRANCH
from seg.model.transformer.create_modelV2 import create_transformerV2
from seg.model.zed.parts import SCSEModule
from .fuse import MiniEncoderFuse

class CondensedFusionNetwork(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        with_fusion=True,
        ):
        super(CondensedFusionNetwork, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16 or self.patch_size == 32, \
            'patch_size must be {16, 32}'

        self.cnn_branch = CNN_BRANCH(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Merger(nn.Module):
    def __init__(
        self,
        num_seg_maps: int, # the number of [N, 1, 256, 256] maps to be merged
        use_weights: bool = False,
    ):
        super(Merger, self).__init__()

        # not sure if this is a good place to do the weights - investigate
        self.use_weights = use_weights
        if self.use_weights:
            print(f'Using weights in Merger.')
            self.weights = nn.Parameter(data=torch.ones([num_seg_maps]), requires_grad=True)
            print(f'ERROR: Weights not initialized yet.')
            exit(1)


        self.num_seg_maps = num_seg_maps
        self.convs_1st = nn.ModuleList()
        self.convs_2nd = nn.ModuleList()
        self.convs_3rd = nn.ModuleList()

        for i in range(num_seg_maps):
            self.convs_1st.append(conv3x3(1, 64))
            self.convs_2nd.append(conv3x3(64, 128))
            self.convs_3rd.append(conv3x3(128, 256))
        
        self.scse = SCSEModule(256, reduction=16)

    def forward(self, tList: list):
        assert len(tList) == self.num_seg_maps, \
            f'len(tList), {len(tList)} != num_seg_maps: {self.num_seg_maps}'
        tens_list = list()
        assert len(tens_list) == 0
        
        for i in range(self.num_seg_maps):
            x_i = self.convs_1st[i](tList[i])
            x_i = self.convs_2nd[i](x_i)
            x_i = self.convs_3rd[i](x_i)
            tens_list.append(x_i)
        
        # GLOBAL AVG POOLING 
        output = torch.mean(torch.stack(tens_list), dim=0)

        # squeeze and excitation 
        output = self.scse(output)

        # average along final dim
        output = torch.mean(output, dim=1, keepdim=True)
        return output


class MergerNoSqueezeAndExitation(nn.Module):
    def __init__(
        self,
        num_seg_maps: int, # the number of [N, 1, 256, 256] maps to be merged
        use_weights: bool = False,
    ):
        super(MergerNoSqueezeAndExitation, self).__init__()

        # not sure if this is a good place to do the weights - investigate
        self.use_weights = use_weights
        if self.use_weights:
            print(f'Using weights in Merger.')
            self.weights = nn.Parameter(data=torch.ones([num_seg_maps]), requires_grad=True)
            print(f'ERROR: Weights not initialized yet.')
            exit(1)


        self.num_seg_maps = num_seg_maps
        self.convs_1st = nn.ModuleList()
        self.convs_2nd = nn.ModuleList()
        self.convs_3rd = nn.ModuleList()

        for i in range(num_seg_maps):
            self.convs_1st.append(conv3x3(1, 64))
            self.convs_2nd.append(conv3x3(64, 128))
            self.convs_3rd.append(conv3x3(128, 256))
        
    def forward(self, tList: list):
        assert len(tList) == self.num_seg_maps, \
            f'len(tList), {len(tList)} != num_seg_maps: {self.num_seg_maps}'
        tens_list = list()
        assert len(tens_list) == 0
        
        for i in range(self.num_seg_maps):
            x_i = self.convs_1st[i](tList[i])
            x_i = self.convs_2nd[i](x_i)
            x_i = self.convs_3rd[i](x_i)
            tens_list.append(x_i)
        
        # GLOBAL AVG POOLING 
        output = torch.mean(torch.stack(tens_list), dim=0)

        # average along final dim
        output = torch.mean(output, dim=1, keepdim=True)

        return output   


class NewFusionNetworkWithMergingNo1_2NoWeightsForSegMaps(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        ):
        super(NewFusionNetworkWithMergingNo1_2NoWeightsForSegMaps, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16, 'patch_size must be 16'

        self.cnn_branch = CNN_BRANCH(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
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
        
        self.merger = Merger(
            num_seg_maps=5,
            use_weights=False,
        )

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)
        x_final_trans = self.trans_branch(images) # 5 x 1 x 156 x 156

        '''
        self.CNN_BRANCH and self.TRANSFORMER_BRANCH should have same members:
                { output_1_4, output_1_2 }
        '''
        # NOT including x.1_2 for this 
        # self.x_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, self.trans_branch.x_1_2)
        self.x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans_branch.x_1_4)
        self.x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans_branch.x_1_8)
        self.x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans_branch.x_1_16)

        tensor_list = [x_final_cnn, x_final_trans, self.x_1_4, self.x_1_8, self.x_1_16]
        output = self.merger(tensor_list)
        return output


class NewFusionNetworkWithMergingNo1_2NoWeightsForSegMapsNoSqueeze(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        ):
        super(NewFusionNetworkWithMergingNo1_2NoWeightsForSegMapsNoSqueeze, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16, 'patch_size must be 16'

        self.cnn_branch = CNN_BRANCH(
            n_channels=cnn_model_cfg['in_channels'],
            n_classes=cnn_model_cfg['num_classes'],
            patch_size=cnn_model_cfg['patch_size'],
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
        
        self.merger = MergerNoSqueezeAndExitation(
            num_seg_maps=5,
            use_weights=False,
        )

    def forward(self, images):
        x_final_cnn = self.cnn_branch(images)
        x_final_trans = self.trans_branch(images) # 5 x 1 x 156 x 156

        '''
        self.CNN_BRANCH and self.TRANSFORMER_BRANCH should have same members:
                { output_1_4, output_1_2 }
        '''
        # NOT including x.1_2 for this 
        # self.x_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, self.trans_branch.x_1_2)
        self.x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans_branch.x_1_4)
        self.x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans_branch.x_1_8)
        self.x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans_branch.x_1_16)

        tensor_list = [x_final_cnn, x_final_trans, self.x_1_4, self.x_1_8, self.x_1_16]
        output = self.merger(tensor_list)
        return output

if __name__ == '__main__':

    num_seg_maps = 5
    seg_maps = list()
    for i in range(num_seg_maps):
        x = torch.randn((2, 1, 256, 256), device='cuda')
        seg_maps.append(x)
        print(seg_maps[i].shape)
    
    model = MergerNoSqueezeAndExitation(
        num_seg_maps = len(seg_maps),
        use_weights=False,
    ).cuda()

    
    for i in range(20):
        out = model(seg_maps)
    # output = torch.mean(torch.stack(seg_maps), dim=0)
    # print(f'output.shape: {output.shape}')