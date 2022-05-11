from logging import Logger
import torch
import torch.nn as nn 
import torch.nn.functional as F

from seg.model.CNN.CNN import CNN_BRANCH
from seg.model.transformer.create_modelV2 import create_transformerV2
from seg.model.zed.parts import NouveauAttention, SCSEModule
from .fuse import MiniEncoderFuse

class Merger3x3BNR(nn.Module):
    def __init__(
        self,
        num_seg_maps: int, # the number of [N, 1, 256, 256] maps to be merged
        use_weights: bool = False,
    ):
        super(Merger3x3BNR, self).__init__()

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
            self.convs_1st.append(BNRconv3x3(1, 64))
            self.convs_2nd.append(BNRconv3x3(64, 128))
            self.convs_3rd.append(BNRconv3x3(128, 256))
        
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

class CondensedFusionNetwork(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
    ):
        """
        Uses a fusion network style similar to before, but Merger has batch norm
        and relu at the end. Also includes a squeeze and excitation module at 
        the end. -> has not been tested yet. So most similar to model in this .py
        file:
            `NewFusionNetworkWithMergingNo1_2NoWeightsForSegMaps`
        """
        super(CondensedFusionNetwork, self).__init__()

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
        
        self.merger = Merger3x3BNR(
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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

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

class dTripleConv(nn.Module):
    def __init__(
        self,
        in_planes, 
        out_planes, 
        stride=1, 
        groups=1, 
    ):
        """
        Dilated Triple Convolution with BatchNorm and ReLU. 
        1st conv3x3 is dilation=1
        2nd conv3x3 is dilation=2
        3rd conv3x3 is dilation=3
        Visual found here: 
        https://www.researchgate.net/figure/3-3-convolution-kernels-with-different-dilation-rate-as-1-2-and-3_fig9_323444534
        """
        super(dTripleConv, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, groups=groups, bias=False, dilation=1)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride,
                    padding=2, groups=groups, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride,
                    padding=3, groups=groups, bias=False, dilation=3)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x 

class TripleConv(nn.Module):
    def __init__(
        self,
        in_planes, 
        out_planes, 
        mid_planes,
        stride=1, 
        groups=1, 
        dilation=1
    ):
        """
        TripleConv - no dilations though (unless specified for all of them in 
        args)
        Visual found here: 
        https://www.researchgate.net/figure/3-3-convolution-kernels-with-different-dilation-rate-as-1-2-and-3_fig9_323444534
        """
        super(dTripleConv, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x 

class tUp(nn.Module):
    """
    Upsampling - followed by triple conv. 
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = TripleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)

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

class NewFusion(nn.Module):
    def __init__(
        self,
        in_chans_t,
        in_chans_c,
        dec_chans_t=(256, 256, 512),
        dec_chans_c=(256, 256, 512),
    ):
        super(NewFusion, self).__init__()
        print(f'Hyperfocus decoder channel progression, Trans: {dec_chans_t}')
        print(f'Hyperfocus decoder channel progression, CNN: {dec_chans_c}')

        self.c1_t = BNRconv3x3(in_chans_t, dec_chans_t[0])
        self.c1_c = BNRconv3x3(in_chans_c, dec_chans_c[0])

        self.c2_t = BNRconv3x3(dec_chans_t[0], dec_chans_t[1])
        self.c2_c = BNRconv3x3(dec_chans_c[0], dec_chans_c[1])

        self.c3_t = BNRconv3x3(dec_chans_t[1], dec_chans_t[2])
        self.c3_c = BNRconv3x3(dec_chans_c[1], dec_chans_c[2])

        self.att1 = NouveauAttention(dec_chans_t[2] + dec_chans_c[2], reduction=16)

    def forward(self, x_t, x_c):
        # feature extraction and concatenation at [16, 16] scale 
        x_out_t = self.c1_t(x_t)
        x_out_t = self.c2_t(x_out_t)
        x_out_t = self.c3_t(x_out_t)

        x_out_c = self.c1_c(x_c)
        x_out_c = self.c2_c(x_out_c)
        x_out_c = self.c3_c(x_out_c)

        x_out = torch.cat([x_out_t, x_out_c], dim=1) # output: x_out: torch.Size([2, 1024, 16, 16]) (as expected)

        # now incorporate actual decoder and upsampling from other feature scales 
        x_out = self.att1(x_out)


        print(f'x_out: {x_out.shape}')


        return x_out 
        

        


if __name__ == '__main__':
    x_t = torch.randn((2, 64, 16, 16), device='cuda')
    x_c = torch.randn((2, 512, 16, 16), device='cuda')

    fuse = NewFusion(
        in_chans_t=x_t.shape[1],
        in_chans_c=x_c.shape[1],
        dec_chans_t=(256, 256, 512),
        dec_chans_c=(256, 256, 512),
    ).cuda()

    out = fuse(x_t, x_c)
