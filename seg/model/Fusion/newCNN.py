import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from seg.model.general.input_project import InputProjectionA
from seg.model.general.count_params_master import count_parameters

from pathlib import Path
from seg.model.CNN.CNN import CNN_BRANCH
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
        planes=[sec_block_convs * 2 + init_block_convs + rfb_out_chans, 256 // 2 , 512 // 2, 1024 // 2]
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

        out_chans = [512 // 2, 256 // 2, 128 // 2, 64 // 2, 32 // 2]
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

