import torch 
import torch.nn as nn
import torch.nn.functional as F

from seg.model.CNN.CNN_parts import Down, Up
from seg.model.CNN.CNN import CNN_BRANCH
from seg.model.transformer.create_modelV2 import create_transformerV2
from seg.model.zed.parts import NouveauAttention, SCSEModule
from .CondensedFusion import Merger3x3BNR

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=padding, dilation=dilation,groups=planes, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.pointwise(x)
        return x
 
 
class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes=1, output_stride=16):
        super(ASPP, self).__init__()
 
        if output_stride == 16:
            dilations = [1, 2, 3, 4]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
 
        self.aspp1 = _ASPPModule(inplanes, 64, 1, padding=1, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 64, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 64, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 64, 3, padding=dilations[3], dilation=dilations[3])
 
        self.conv1 = nn.Conv2d(256, outplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(0.5)
 
    def forward(self, x):
        x1 = self.aspp1(x) #64,62,30
        x2 = self.aspp2(x) #64,64,32
        x3 = self.aspp3(x) #64,64,32
        x4 = self.aspp4(x) #64,64,32
        x = torch.cat((x1, x2, x3, x4), dim=1)
 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class ASPP_fuse(nn.Module):
    def __init__(
        self, 
        in_chan_CNN, 
        in_chan_TRANS, 
        intermediate_chan,
        out_chan=1,
        stage=None,
        drop_rate = 0.5
        ):
        super(ASPP_fuse, self).__init__()

        stages = ['1_2', '1_4', '1_8', '1_16', '1_32']
        self.fuse_stage = stage
        assert self.fuse_stage in stages

        if self.fuse_stage == '1_2':
            self.scale_factor = 2
        elif self.fuse_stage == '1_4':
            self.scale_factor = 4
        elif self.fuse_stage == '1_8':
            self.scale_factor = 8 
        elif self.fuse_stage == '1_16':
            self.scale_factor = 16
        elif self.fuse_stage == '1_32':
            self.scale_factor = 32
        else:
            raise ValueError(f'Valid stages for fusion: {stages}')

        self.down1 = Down(in_chan_CNN + in_chan_TRANS, intermediate_chan)
        # self.super1 = SuperficialModule(nIn=intermediate_chan)
        self.down2 = Down(intermediate_chan, intermediate_chan)
        self.up1 = Up(intermediate_chan, intermediate_chan)
        # self.super2 = SuperficialModule(nIn=intermediate_chan)
        self.up2 = Up(intermediate_chan, intermediate_chan)
        self.aspp = ASPP(inplanes=intermediate_chan, outplanes=out_chan)

    def forward(self, x_CNN, x_TRANS):
        assert(x_CNN.shape[0] == x_TRANS.shape[0] 
            and x_CNN.shape[2] == x_TRANS.shape[2] 
            and x_CNN.shape[3] == x_TRANS.shape[3])
            
        x = torch.cat([x_CNN, x_TRANS], dim=1) #; print(f'\tcat output {x.shape}')
        x = self.down1(x) #; print(f'\tdown1 output {x.shape}')
        # x = self.super1(x)
        x = self.down2(x) #; print(f'\tdown2 output {x.shape}')
        x = self.up1(x) #; print(f'\tup1 output {x.shape}')
        # x = self.super2(x)
        x = self.up2(x) #; print(f'\tup2 output {x.shape}')
        x = self.aspp(x)
        seg_map = F.interpolate(
            x, 
            scale_factor = self.scale_factor, 
            mode='bilinear') 
        return seg_map

class ASPPFusionNetwork(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
    ):
        """
        Replaces fusion modules with ASPP modules. 
        """
        super(ASPPFusionNetwork, self).__init__()

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

        # self.fuse_1_2 = ASPP_fuse( # NOTE: 64 classes trans output manually input here 
        #     self.cnn_branch.x_1_2.shape[1], num_output_trans, 64, 1, stage = '1_2')
        # self.fuse_1_4 = ASPP_fuse(
        #     self.cnn_branch.x_1_4.shape[1], num_output_trans, 64, 1, stage='1_4')
        self.fuse_1_8 = ASPP_fuse(
            self.cnn_branch.x_1_8.shape[1], num_output_trans, 64, 1, stage='1_8')
        self.fuse_1_16 = ASPP_fuse(
            self.cnn_branch.x_1_16.shape[1], num_output_trans, 64, 1, stage='1_16')
        if self.patch_size == 32:
            self.fuse_1_32 = ASPP_fuse(
                self.cnn_branch.x_1_32.shape[1], num_output_trans, 64, 1, stage='1_32')
        
        self.merger = Merger3x3BNR(
            num_seg_maps=4,
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
        # self.x_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, self.trans_branch.x_1_2); print(f'WARNING: self.x_1_2 NOT included in forward(x) in tensor_list FusionNetwork')
        # self.x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans_branch.x_1_4); print(f'WARNING: self.x_1_4 NOT included in forward(x) in tensor_list FusionNetwork')
        self.x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans_branch.x_1_8)
        self.x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans_branch.x_1_16)

        # print(f'x_final_cnn.shape: {x_final_cnn.shape}')
        # print(f'x_final_trans.shape: {x_final_trans.shape}')
        # print(f'x_1_8.shape: {self.x_1_8.shape}')
        # print(f'x_1_16.shape: {self.x_1_16.shape}')


        tensor_list = [x_final_cnn, x_final_trans, self.x_1_8, self.x_1_16]
        output = self.merger(tensor_list)
        return output

if __name__ == '__main__':
    x = torch.randn((10, 128, 256, 256), device='cuda')
    x_trans = torch.randn((10, 64, 16, 16), device='cuda')
    x_cnn = torch.randn((10, 512, 16, 16), device='cuda')
    model = ASPP_fuse(
        in_chan_CNN=512,
        in_chan_TRANS=64,
        intermediate_chan=64,
        out_chan=1,
        stage='1_2'
    ).cuda()

    x_out = model(x_cnn,x_trans)
    print(f'x_out.shape: {x_out.shape}')

    # from torchsummary import summary 
    # summary(model=model, input_size=[x.shape[1:], x.shape[1:]], batch_size=x.shape[0])