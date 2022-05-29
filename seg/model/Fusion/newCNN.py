import torch 
import torch.nn as nn 
import torch.nn.functional as F
from seg.model.Fusion.fuse import MiniEncoderFuseDWSep 

from seg.model.general.input_project import InputProjectionA
from seg.model.general.count_params_master import count_parameters

from pathlib import Path
from seg.model.CNN.CNN import CNN_BRANCH
from seg.model.transformer.create_modelV2 import create_transformerV2
from seg.model.zed.parts import NouveauAttention, SCSEModule, UpDWSep
from seg.model.zed.zedNet import zedNet, zedNetDWSep, zedNetMod

from seg.model.Fusion.CondensedFusion import BNRconv3x3
from seg.model.general.RFB_sep import RFB_separable
from seg.model.general.DW_sep import SeparableConv2D

from .siddnet_parts import CCMModule, RCCMModule, BR


class xCNN(nn.Module):
    def __init__(
        self,
        cnn_model_cfg,
        init_block_convs=24,    # can be made into list w below
        sec_block_convs=48,     # can be made into list w above 
        p=5,
    ):
        """
        As we were kind of thinking about yesterday. Becuase this has an initial
        convolutional structure and downsampling at the beginning we have some 
        options. 
            1.  Keep input downsampled. Feed into CNN and trans at increased num
                chans. Use create_transformerV5. 
            2.  Keep input downsampled. However reduce to input = 3 channels. 
                will default to old code, can use create_transformerV4 or whatever. 
            3.  Upsample input. Take larger number of channels. Same solution as P1 
                in terms of the transformer. 
            4.  Upsample input. Reduce to 3 channels. Use create_transformerV4. 
        """
        super(xCNN, self).__init__()
        print(f'Model: {self._get_name()} initialized.')


        # --- constant size convolution block ---
        self.init_block = nn.Sequential(
            BNRconv3x3(in_planes=3, out_planes=init_block_convs, stride=1),
            BNRconv3x3(in_planes=init_block_convs, out_planes=init_block_convs, stride=1),
            BNRconv3x3(in_planes=init_block_convs, out_planes=init_block_convs, stride=1),
        )

        rfb_out_chans=64
        self.init_rfb_block = nn.Sequential(
            RFB_separable(in_channel=3, out_channel=32),
            nn.MaxPool2d(2),
            RFB_separable(in_channel=32, out_channel=rfb_out_chans),
            nn.MaxPool2d(2)
        )


        # --- downsampling convolution ---
        self.level1 = BNRconv3x3(init_block_convs, init_block_convs, stride=2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)
        self.BR_1 = nn.Sequential(
            nn.BatchNorm2d(init_block_convs + init_block_convs),
            nn.PReLU(init_block_convs + init_block_convs)
        )
        self.level2_0 = CCMModule(
            init_block_convs + init_block_convs, 
            sec_block_convs, 
            ratio=[1, 2, 3]
        )
        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(
                RCCMModule(sec_block_convs, sec_block_convs, ratio=[1, 3, 4]))  # , ratio=[1,3,4]
        self.b2 = BR(sec_block_convs * 2 + init_block_convs)

        self.BNS_RFBs = nn.ModuleList()
        planes=[sec_block_convs * 2 + init_block_convs + rfb_out_chans, 256, 512, 1024]
        for i in range(0, len(planes) - 1):
            self.BNS_RFBs.append(
                nn.Sequential(
                    SeparableConv2D(planes[i], planes[i+1], kernel_size=3, 
                    stride=1, padding=1, dilation=1),
                    nn.BatchNorm2d(planes[i+1]), 
                    nn.SiLU(True),
                    RFB_separable(planes[i+1], planes[i+1]),
                    nn.MaxPool2d(2),
                )
            )
        
        # decoder branch

        out_chans = [512, 256, 128, 64, 32]
        self.up1 = UpDWSep(planes[3] + planes[2], out_chans[0], bilinear=True)
        self.up2 = UpDWSep(out_chans[0] + planes[1], out_chans[1], bilinear=True)
        self.att1 = SCSEModule(in_channels=out_chans[1], reduction=16)
        self.up3 = UpDWSep(out_chans[1] + planes[0], out_chans[2], bilinear=True)
        self.up4 = UpDWSep(out_chans[2] + init_block_convs, out_chans[3], bilinear=True)
        self.att3 = NouveauAttention(out_chans[3], reduction=2, AvgPoolKernelSize=31, AvgPoolPadding=15)
        self.up5 = UpDWSep(out_chans[3], out_chans[3], bilinear=True)
        self.final_conv = SeparableConv2D(out_chans[3], out_channels=1, kernel_size=1)


    def forward(self, input):
        img=input
        input = self.init_block(input);                                         # print(f'[input]: \t {input.shape}')
        rfb_input = self.init_rfb_block(img);                                   # print(f'[rfb_input]: \t {rfb_input.shape}')
        output0 = self.level1(input);                                           # print(f'[output0]:\t {output0.shape}')
        inp1 = self.sample1(input);                                             # print(f'[inp1]:\t\t {inp1.shape}')
        inp2 = self.sample2(input);                                             # print(f'[inp2]:\t\t {inp2.shape}')
        output0_cat = self.BR_1(torch.cat([output0, inp1], dim=1));             # print(f'[output0_cat]:\t {output0_cat.shape}')
        output1_0 = self.level2_0(output0_cat);                                 # print(f'[output1_0]:\t {output1_0.shape}')

        for i, layer in enumerate(self.level2):   # RCCM-1 X 5
            if i == 0:
                output1 = layer(output1_0)    #,  Ci =48, Co=48
            else:
                output1 = layer(output1) #,  Ci =48, Co=48
        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1));        # print(f'[output1_cat]:\t {output1_cat.shape}')
        output128x128 = torch.cat([output1_cat, rfb_input], 1);                 # print(f'[output1_cat]:\t {output1_cat.shape}')
        
        # run through self.BNS_RFBs
        output64x64 = self.BNS_RFBs[0](output128x128);                          # print(f'[output64x64]:\t {output64x64.shape}')                         
        output32x32 = self.BNS_RFBs[1](output64x64);                            # print(f'[output32x32]:\t {output32x32.shape}')  
        output16x16 = self.BNS_RFBs[2](output32x32);                            # print(f'[output16x16]:\t {output16x16.shape}')  

        output = self.up1(output16x16, output32x32);                            # print(f'[output]:\t {output.shape}')
        output = self.up2(output, output64x64);                                 # print(f'[output]:\t {output.shape}')
        output = self.up3(output, output128x128);                               # print(f'[output]:\t {output.shape}')
        output = self.up4(output, inp1);                                        # print(f'[output]:\t {output.shape}')
        output = self.up5(output);                                              # print(f'[output]:\t {output.shape}')
        seg_map = self.final_conv(output);                                      # print(f'[seg_map]:\t {seg_map.shape}')

        return seg_map 

class xCNN_v2(nn.Module):
    def __init__(
        self,
        in_channels=3,
        init_block_convs=32,    # can be made into list w below
        sec_block_convs=128,     # can be made into list w above 
        p=5,
    ):
        """
        As we were kind of thinking about yesterday. Becuase this has an initial
        convolutional structure and downsampling at the beginning we have some 
        options. 
            1.  Keep input downsampled. Feed into CNN and trans at increased num
                chans. Use create_transformerV5. 
            2.  Keep input downsampled. However reduce to input = 3 channels. 
                will default to old code, can use create_transformerV4 or whatever. 
            3.  Upsample input. Take larger number of channels. Same solution as P1 
                in terms of the transformer. 
            4.  Upsample input. Reduce to 3 channels. Use create_transformerV4. 
        """
        super(xCNN_v2, self).__init__()
        print(f'Model: {self._get_name()} initialized.')


        # --- constant size convolution block ---
        self.init_block = nn.Sequential(
            BNRconv3x3(in_planes=in_channels, out_planes=init_block_convs, stride=1),
            BNRconv3x3(in_planes=init_block_convs, out_planes=init_block_convs, stride=1),
            BNRconv3x3(in_planes=init_block_convs, out_planes=init_block_convs, stride=1),
        )

        rfb_out_chans=64
        self.init_rfb_block = nn.Sequential(
            RFB_separable(in_channel=in_channels, out_channel=32),
            nn.MaxPool2d(2),
            RFB_separable(in_channel=32, out_channel=rfb_out_chans),
            nn.MaxPool2d(2)
        )


        # --- downsampling convolution ---
        self.level1 = BNRconv3x3(init_block_convs, init_block_convs, stride=2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)
        self.BR_1 = nn.Sequential(
            nn.BatchNorm2d(init_block_convs + init_block_convs),
            nn.PReLU(init_block_convs + init_block_convs)
        )
        self.level2_0 = CCMModule(
            init_block_convs + init_block_convs, 
            sec_block_convs, 
            ratio=[1, 2, 3]
        )
        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(
                RCCMModule(sec_block_convs, sec_block_convs, ratio=[1, 3, 4]))  # , ratio=[1,3,4]
        self.b2 = BR(sec_block_convs * 2 + init_block_convs)

        self.BNS_RFBs = nn.ModuleList()
        planes=[sec_block_convs * 2 + init_block_convs + rfb_out_chans, 256, 512, 1024]
        for i in range(0, len(planes) - 1):
            self.BNS_RFBs.append(
                nn.Sequential(
                    SeparableConv2D(planes[i], planes[i+1], kernel_size=3, 
                    stride=1, padding=1, dilation=1),
                    nn.BatchNorm2d(planes[i+1]), 
                    nn.SiLU(True),
                    RFB_separable(planes[i+1], planes[i+1]),
                    nn.MaxPool2d(2),
                )
            )
        
        # decoder branch

        out_chans = [512, 64, 128, 64, 32]
        self.up1 = UpDWSep(planes[3] + planes[2], out_chans[0], bilinear=True)
        self.up2 = UpDWSep(out_chans[0] + planes[1], out_chans[1], bilinear=True)
        self.up3 = UpDWSep(out_chans[1] + planes[0], out_chans[2], bilinear=True)
        self.up4 = UpDWSep(out_chans[2] + init_block_convs, out_chans[3], bilinear=True)
        self.up5 = UpDWSep(out_chans[3], out_chans[3], bilinear=True)
        self.final_conv = SeparableConv2D(out_chans[3], out_channels=1, kernel_size=1)

    def get_dimensions(self, N_in, C_in, H_in, W_in, printXDimensions=True):
        dummy_tensor = torch.zeros(N_in, C_in, H_in, W_in)
        x = self.forward(dummy_tensor)
        if printXDimensions:
            print(f'Running a forward pass of {self._get_name()}')
            print(f'self.x_1_2.shape: {self.x_1_2.shape}')
            print(f'self.x_1_4.shape: {self.x_1_4.shape}')
            print(f'self.x_1_8.shape: {self.x_1_8.shape}')
            print(f'self.x_1_16.shape: {self.x_1_16.shape}')
        del dummy_tensor

    def forward(self, input):
        img=input
        input = self.init_block(input);                                         # print(f'[input]: \t {input.shape}')
        rfb_input = self.init_rfb_block(img);                                   # print(f'[rfb_input]: \t {rfb_input.shape}')
        output0 = self.level1(input);                                           # print(f'[output0]:\t {output0.shape}')
        inp1 = self.sample1(input);                                             # print(f'[inp1]:\t\t {inp1.shape}')
        inp2 = self.sample2(input);                                             # print(f'[inp2]:\t\t {inp2.shape}')
        output0_cat = self.BR_1(torch.cat([output0, inp1], dim=1));             # print(f'[output0_cat]:\t {output0_cat.shape}')
        output1_0 = self.level2_0(output0_cat);                                 # print(f'[output1_0]:\t {output1_0.shape}')

        for i, layer in enumerate(self.level2):   # RCCM-1 X 5
            if i == 0:
                output1 = layer(output1_0)    #,  Ci =48, Co=48
            else:
                output1 = layer(output1) #,  Ci =48, Co=48
        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1));        # print(f'[output1_cat]:\t {output1_cat.shape}')
        output128x128 = torch.cat([output1_cat, rfb_input], 1);                 # print(f'[output1_cat]:\t {output1_cat.shape}')
        self.x_1_2 = output128x128

        # run through self.BNS_RFBs
        output64x64 = self.BNS_RFBs[0](output128x128);                          # print(f'[output64x64]:\t {output64x64.shape}')                         
        output32x32 = self.BNS_RFBs[1](output64x64);                            # print(f'[output32x32]:\t {output32x32.shape}')  
        output16x16 = self.BNS_RFBs[2](output32x32);                            # print(f'[output16x16]:\t {output16x16.shape}')  

        self.x_1_4 = output64x64
        self.x_1_8 = output32x32

        self.x_1_16 = output16x16
        output = self.up1(output16x16, output32x32);                            # print(f'[output]:\t {output.shape}')
        output = self.up2(output, output64x64);                                 # print(f'[output]:\t {output.shape}')
        output = self.up3(output, output128x128);                               # print(f'[output]:\t {output.shape}')
        output = self.up4(output, inp1);                                        # print(f'[output]:\t {output.shape}')
        output = self.up5(output);                                              # print(f'[output]:\t {output.shape}')
        seg_map = self.final_conv(output);                                      # print(f'[seg_map]:\t {seg_map.shape}')

        return seg_map 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class SeparableSEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, r=16):
        super(SeparableSEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SeparableConv2D(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SeparableConv2D(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # add SE block
        self.se = SE_Block(planes, r)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # add SE operation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.cat([out, identity], dim=1)
        # out += identity
        out = self.relu(out)

        return out

class xFusion(nn.Module):
    def __init__(
        self, 
        cnn_model_cfg,
        trans_model_cfg,
        with_fusion=True,
        ):
        super(xFusion, self).__init__()

        self.patch_size = cnn_model_cfg['patch_size']
        assert cnn_model_cfg['patch_size'] == trans_model_cfg['patch_size'], \
            'patch_size not configd properly, model_cfgs have different values'
        assert self.patch_size == 16 or self.patch_size == 32, \
            'patch_size must be {16, 32}'

        print(f'cnn_model_cfg[in_channels]:', cnn_model_cfg['in_channels']); assert cnn_model_cfg['in_channels'] == 3
        print(f'cnn_model_cfg[init_block_convs]:', cnn_model_cfg['init_block_convs'])
        print(f'cnn_model_cfg[sec_block_convs]:', cnn_model_cfg['sec_block_convs'])

        self.cnn_branch = xCNN_v2(
            in_channels=cnn_model_cfg['in_channels'], 
            init_block_convs=cnn_model_cfg['init_block_convs'], 
            sec_block_convs=cnn_model_cfg['sec_block_convs'], 
            p=5
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


if __name__ == '__main__':
    scse = SE_Block(64, r=16).cuda()
    x = torch.randn((1, 64, 128, 128), device='cuda')
    output = scse(x)
    print(f'[input, SE_Block]: \t {x.shape}')
    print(f'[output, SE_Block]: \t {output.shape}')

    separable_scse = SeparableSEBasicBlock(64, 32, stride=1).cuda()
    x = torch.randn((1, 64, 128, 128), device='cuda')
    output = separable_scse(x)
    print(f'[in, SepSCSEBlock]: \t {x.shape}')
    print(f'[out, SepSCSEBlock]: \t {output.shape}')
    count_parameters(separable_scse)