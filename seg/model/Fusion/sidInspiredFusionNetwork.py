import torch 
import torch.nn as nn
import torch.nn.functional as F 

from seg.model.general.input_project import InputProjectionA
from seg.model.general.count_params_master import count_parameters

from pathlib import Path
from seg.model.CNN.CNN import CNN_BRANCH
from seg.model.zed.zedNet import zedNet, zedNetDWSep, zedNetMod

from seg.model.Fusion.CondensedFusion import BNRconv3x3
from seg.model.alt_cnns.pranetSimple import RFB_modified
from seg.model.general.DW_sep import SeparableConv2D
from seg.model.transformer.decoder_new import DecoderMultiClassDilationAndSCSE, DecoderMultiClassDilationAndSCSEFusion, DecoderMultiClassDilationAndSCSEFusionJustOne, DecoderMultiClassDilationAndSCSEReduced, UpModDilatedDWSep, UpModDilated
from seg.model.transformer.transformerNoDecoderAdjustableChannels import create_transformerV5

from seg.model.transformer.create_model import create_transformer, create_vit
from seg.model.transformer.create_modelV2 import create_transformerV2
from seg.model.Fusion.fuse import SimpleFusion
from .fuse import CCMFusionModule, MiniEncoderFuse, MiniEncoderFuseDWSep, MiniEncoderFuseDWSepRFB
from .siddnet_parts import CCMModule, RCCMModule, BR

class MultiLevelInputFusionNetworkSingleTransformer(nn.Module):
    def __init__(
        self,
        cnn_model_cfg,
        trans_model_cfg,
        init_block_convs=24,    # can be made into list w below
        sec_block_convs=48,     # can be made into list w above 
        option=None,
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
        super(MultiLevelInputFusionNetworkSingleTransformer, self).__init__()
        print(f'Model: {self._get_name()} initialized.')


        # --- constant size convolution block ---
        self.init_block = nn.Sequential(
            BNRconv3x3(in_planes=3, out_planes=init_block_convs, stride=1),
            BNRconv3x3(in_planes=init_block_convs, out_planes=init_block_convs, stride=1),
            BNRconv3x3(in_planes=init_block_convs, out_planes=init_block_convs, stride=1),
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

        self.option = option
        if self.option == 1:
            print(f'Keeping input downsampled from initial convolution.', \
            f'Feeding input at increased number of channels:', \
            f'{sec_block_convs * 2 + init_block_convs}')

            trans_model_cfg['in_channels'] = sec_block_convs * 2 + init_block_convs
            cnn_model_cfg['in_channels'] = sec_block_convs * 2 + init_block_convs

            self.trans = create_transformerV5(trans_model_cfg, decoder='linear')
            
            self.cnn_branch = zedNetDWSep(
                n_channels=cnn_model_cfg['in_channels'],
                n_classes=cnn_model_cfg['num_classes'],
                patch_size=cnn_model_cfg['patch_size'],
                bilinear=True,
                attention=True,
            )
        elif self.option == 2:
            print(f'Keeping input downsampled. Input chans to CNN and Trans: 3')
            self.DWSepconv1 = SeparableConv2D(
                sec_block_convs * 2 + init_block_convs,
                3,
                kernel_size=3, 
                padding=1,
            )
            trans_model_cfg['in_channels'] = 3
            cnn_model_cfg['in_channels'] = 3

            self.trans = create_transformerV5(trans_model_cfg, decoder='linear')

        elif self.option == 3:
            print(f'Upsampling to input size. Keeping increased number of channels.',\
            f'Feeding input at increased number of channels:', \
            f'{sec_block_convs * 2 + init_block_convs}')
            self.upsample1 = nn.Upsample(
                scale_factor=4, 
                mode='bilinear', 
                align_corners=False
            )
            trans_model_cfg['in_channels'] = sec_block_convs * 2 + init_block_convs
            cnn_model_cfg['in_channels'] = sec_block_convs * 2 + init_block_convs

            self.trans = create_transformerV5(trans_model_cfg, decoder='linear')

        elif self.option == 4:
            print(f'Upsampling to input size. Input chans to CNN and Trans: 3.')
            self.upsample1 = nn.Upsample(
                scale_factor=4, 
                mode='bilinear', 
                align_corners=False
            )
            self.DWSepconv1 = SeparableConv2D(
                sec_block_convs * 2 + init_block_convs,
                3,
                kernel_size=3, 
                padding=1,
            )
            trans_model_cfg['in_channels'] = 3
            cnn_model_cfg['in_channels'] = 3

            self.trans = create_transformerV5(trans_model_cfg, decoder='linear')
        else:
            raise ValueError(f'option: {option} invalid.')

        # generate CNN
        self.cnn_branch = zedNetDWSep(
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

        # generate trans decopder
        num_output_trans = trans_model_cfg['num_output_trans']
        print(f'num_output_trans: {num_output_trans}')

        self.decoder_trans = DecoderMultiClassDilationAndSCSEReduced(
            in_chans=num_output_trans,
            inter_chans=32,
            out_chans=1,
            dilation1=1,
            dilation2=3,
        )

        self.fuse_1_2 = CCMFusionModule( # NOTE: 64 classes trans output manually input here 
            self.cnn_branch.x_1_2.shape[1], num_output_trans, 64, 4, stage = '1_2')
        self.up_1_2 = UpModDilated(4, 1, True, scale_factor=2, dilation=4)

        self.fuse_1_4 = CCMFusionModule(
            self.cnn_branch.x_1_4.shape[1], num_output_trans, 64, 8, stage='1_4')
        self.up_1_4_0 = UpModDilated(8, 4, True, scale_factor=2, dilation=2)
        self.up_1_4_1 = UpModDilated(4, 1, True, scale_factor=2, dilation=3)

        self.fuse_1_8 = CCMFusionModule(
            self.cnn_branch.x_1_8.shape[1], num_output_trans, 64, 16, stage='1_8')
        self.up_1_8_0 = UpModDilated(16, 8, True, scale_factor=2, dilation=1)
        self.up_1_8_1 = UpModDilated(8, 1, True, scale_factor=4, dilation=2)

        self.fuse_1_16 = CCMFusionModule(
            self.cnn_branch.x_1_16.shape[1], num_output_trans, 64, 32, stage='1_16')
        self.up_1_16_0 = UpModDilated(32, 16, True, scale_factor=4, dilation=1)
        self.up_1_16_1 = UpModDilated(16, 1, True, scale_factor=4, dilation=1)


    def forward(self, input):
        input = self.init_block(input);                                         print(f'[input]: \t {input.shape}')
        output0 = self.level1(input);                                           print(f'[output0]:\t {output0.shape}')
        inp1 = self.sample1(input);                                             print(f'[inp1]:\t\t {inp1.shape}')
        inp2 = self.sample2(input);                                             print(f'[inp2]:\t\t {inp2.shape}')
        output0_cat = self.BR_1(torch.cat([output0, inp1], dim=1));             print(f'[output0_cat]:\t {output0_cat.shape}')
        output1_0 = self.level2_0(output0_cat);                                 print(f'[output1_0]:\t {output1_0.shape}')

        for i, layer in enumerate(self.level2):   # RCCM-1 X 5
            if i == 0:
                output1 = layer(output1_0)    #,  Ci =48, Co=48
            else:
                output1 = layer(output1) #,  Ci =48, Co=48
        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1));        print(f'[output1_cat]:\t {output1_cat.shape}')
        

        if self.option == 1:
            output = F.upsample(output1_cat, scale_factor=2, 
            align_corners=False, mode='bilinear');                              print(f'[output]: \t {output.shape}')
        elif self.option == 2:
            output = self.DWSepconv1(output1_cat);                              print(f'[output]: \t {output.shape}')
        elif self.option == 3:
            output = self.upsample1(output1_cat);                               print(f'[output]: \t {output.shape}')
        elif self.option == 4: 
            output = self.upsample1(output1_cat);                               
            output = self.DWSepconv1(output);                                   print(f'[output]: \t {output.shape}')

        x_final_cnn = self.cnn_branch(output);                                  print(f'[x_final_cnn]:\t {x_final_cnn.shape}')
        x_final_trans = self.trans(output);                                     print(f'[x_final_trans]:\t {x_final_trans.shape}')
        x_final_trans = self.decoder_trans(x_final_trans);                      print(f'[x_final_trans]:\t {x_final_trans.shape}')

        self.x_1_2 = self.fuse_1_2(self.cnn_branch.x_1_2, self.trans.x_1_2);    print(f'[x_1_2]:\t\t {self.x_1_2.shape}')
        self.x_1_2 = self.up_1_2(self.x_1_2);                                   print(f'[x_1_2]:\t\t {self.x_1_2.shape}')
        self.x_1_4 = self.fuse_1_4(self.cnn_branch.x_1_4, self.trans.x_1_4);    print(f'[x_1_4]:\t\t {self.x_1_4.shape}')
        self.x_1_4 = self.up_1_4_0(self.x_1_4);                                 print(f'[x_1_4]:\t\t {self.x_1_4.shape}')
        self.x_1_4 = self.up_1_4_1(self.x_1_4);                                 print(f'[x_1_4]:\t\t {self.x_1_4.shape}')
        self.x_1_8 = self.fuse_1_8(self.cnn_branch.x_1_8, self.trans.x_1_8);    print(f'[x_1_8]:\t\t {self.x_1_8.shape}')
        self.x_1_8 = self.up_1_8_0(self.x_1_8);                                 print(f'[x_1_8]:\t\t {self.x_1_8.shape}')
        self.x_1_8 = self.up_1_8_1(self.x_1_8);                                 print(f'[x_1_8]:\t\t {self.x_1_8.shape}')
        self.x_1_16 = self.fuse_1_16(self.cnn_branch.x_1_16, self.trans.x_1_16);print(f'[x_1_16]:\t\t {self.x_1_16.shape}')
        self.x_1_16 = self.up_1_16_0(self.x_1_16);                              print(f'[x_1_16]:\t\t {self.x_1_16.shape}')
        self.x_1_16 = self.up_1_16_1(self.x_1_16);                              print(f'[x_1_16]:\t\t {self.x_1_16.shape}')

        tensor_list = [x_final_cnn, x_final_trans, self.x_1_2, self.x_1_4, self.x_1_8, self.x_1_16]
        return torch.mean(torch.stack(tensor_list), dim=0) 

# class MultiLevelInputFusionNetworkDualTransformer(nn.Module):
#     def __init__(
#         self,
#         cnn_model_cfg,
#         big_trans_model_cfg,
#         sml_trans_model_cfg,
#         decoder_cfg,
#         trans_model_cfg_copy,
#         num_output_trans_big=64,
#         num_output_trans_sml=1,
#         basic_0=24,
#     ):
#         super(MultiLevelInputFusionNetworkDualTransformer, self).__init__()
#         cnn_model_cfg['basic_0'] = basic_0

#         big_trans_model_cfg['image_size'] = trans_model_cfg_copy['image_size']
#         big_trans_model_cfg["dropout"] = trans_model_cfg_copy['dropout']
#         big_trans_model_cfg["drop_path_rate"] = trans_model_cfg_copy['drop_path_rate']
#         big_trans_model_cfg['n_cls'] = trans_model_cfg_copy['n_cls']
#         big_trans_model_cfg['decoder'] = decoder_cfg
#         big_trans_model_cfg['num_output_trans'] = num_output_trans_big
#         decoder_cfg['name'] = 'linear' 

#         sml_trans_model_cfg['image_size'] = trans_model_cfg_copy['image_size']
#         sml_trans_model_cfg["dropout"] = trans_model_cfg_copy['dropout']
#         sml_trans_model_cfg["drop_path_rate"] = trans_model_cfg_copy['drop_path_rate']
#         sml_trans_model_cfg['n_cls'] = trans_model_cfg_copy['n_cls']
#         sml_trans_model_cfg['decoder'] = decoder_cfg
#         sml_trans_model_cfg['num_output_trans'] = num_output_trans_sml

#         # ---- entrance flow to network ----

#         self.init_block = nn.Sequential(
#             BNRconv3x3(in_planes=3, out_planes=basic_0, stride=1),
#             BNRconv3x3(in_planes=basic_0, out_planes=basic_0, stride=1),
#             BNRconv3x3(in_planes=basic_0, out_planes=basic_0, stride=1),
#         )
#         self.level1 = BNRconv3x3(in_planes=basic_0, out_planes=basic_0, stride=2)
#         self.sample1 = InputProjectionA(1)
#         self.sample2 = InputProjectionA(2)

#         self.BR_1 = nn.Sequential(nn.BatchNorm2d(basic_0 + basic_0), nn.PReLU(basic_0 + basic_0))


#         big_trans_model_cfg['in_channels'] = basic_0 + basic_0 # this would have been 3 otherwise. but its not even set
#         sml_trans_model_cfg['in_channels'] = basic_0 + basic_0 # this would have been 3 otherwise. but its not even set
#         self.big_trans = create_transformerV5(big_trans_model_cfg, decoder='linear')
#         self.sml_trans = create_transformerV5(sml_trans_model_cfg, decoder='linear')
        

#         exit(1)


#     def forward(self, input):
#         input = self.init_block(input);                                         print(f'[input]: \t {input.shape}')
#         output0 = self.level1(input);                                           print(f'[output0]:\t {output0.shape}')
#         inp1 = self.sample1(input);                                             print(f'[inp1]:\t\t {inp1.shape}')
#         output0_cat = self.BR_1(torch.cat([output0, inp1], dim=1));             print(f'[output0_cat]:\t {output0_cat.shape}')
#         return output0_cat

# if __name__ == '__main__':
#     model = MultiLevelInputFusionNetwork()
#     input = torch.randn((1, 3, 512, 512))
#     output = model(input)

#     count_parameters(model)    


#     from seg.model.transformer.ViT import VisionTransformer

#     trans = VisionTransformer(
#         image_size = (256, 256),
#         patch_size = 16, 
#         n_layers = 12,
#         d_model = 384,
#         d_ff = 4 * 384,
#         n_heads = 6,
#         n_cls=64,
#         channels=64
#     )

#     count_parameters(trans)