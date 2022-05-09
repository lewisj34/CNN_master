import torch
import torch.nn as nn 

from seg.model.CNN.CNN import CNN_BRANCH
from seg.model.transformer.create_modelV2 import create_transformerV2
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
    ):
        super(Merger, self).__init__()
        self.num_seg_maps = num_seg_maps
        convs_1st = nn.ModuleList()
        convs_2nd = nn.ModuleList()
        convs_3rd = nn.ModuleList()

        for i in range(num_seg_maps):
            convs_1st.append(conv3x3(1, 64))
            convs_2nd.append(conv3x3(64, 128))
            convs_3rd.append(conv3x3(128, 256))
    def forward(self):
        for i in range()



class NewFusionModule(nn.Module):
    def __init__(
        self,
        in_channels_CNN,
        in_channels_Trans,
    ):
        """
        Assumes that the input dimensions of the transformer are going to be 
        just 1/16, whereas the CNN we will have a varying number of skip 
        connections.
        
        Uses attention and global average pooling. 
        """
        super(NewFusionModule, self).__init__()
        self.c1_c = conv3x3(in_channels_CNN, 64)
        self.c1_t = conv3x3(in_channels_Trans, 64) 

        self.c2_c = conv3x3(64, 128)
        self.c2_t = conv3x3(64, 128)
        
        self.c3_c = conv3x3(128, 256)
        self.c3_t = conv3x3(128, 256)

        self.gap = nn.AvgPool2d(kernel_size = )

        x = torch.randn(1, 256, 100, 100)

        x = torch.nn.AvgPool2d(kernel_size = 100, stride = 0, padding = 0, ceil_mode=False, count_include_pad=True)(x)

        



    
    def forward(self, x_trans, x_cnn):
        x = torch.cat([x_cnn, x_trans], dim=1)
        return x
        

if __name__ == '__main__':
    print(f'Hello world!')
    x_trans = torch.rand((10, 64, 16, 16))
    x_cnn = torch.rand((10, 512, 16, 16))
    fusion_mod = NewFusionModule(
        in_channels_CNN=x_cnn.shape[1],
        in_channels_Trans=x_trans.shape[1],
    ).cuda()

    x = fusion_mod.forward(x_trans, x_cnn)
    print(f'x.shape: {x.shape}')
    
