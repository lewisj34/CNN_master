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

class MultiLevelInputFusionNetwork(nn.Module):
    def __init__(
        self,
        cnn_model_cfg,
        big_trans_model_cfg,
        sml_trans_model_cfg,
        decoder_cfg,
        trans_model_cfg_copy,
        num_output_trans_big=64,
        num_output_trans_sml=1,
        basic_0=24,
    ):
        super(MultiLevelInputFusionNetwork, self).__init__()
        cnn_model_cfg['basic_0'] = basic_0

        big_trans_model_cfg['image_size'] = trans_model_cfg_copy['image_size']
        big_trans_model_cfg["dropout"] = trans_model_cfg_copy['dropout']
        big_trans_model_cfg["drop_path_rate"] = trans_model_cfg_copy['drop_path_rate']
        big_trans_model_cfg['n_cls'] = trans_model_cfg_copy['n_cls']
        big_trans_model_cfg['decoder'] = decoder_cfg
        big_trans_model_cfg['num_output_trans'] = num_output_trans_big
        decoder_cfg['name'] = 'linear' 

        sml_trans_model_cfg['image_size'] = trans_model_cfg_copy['image_size']
        sml_trans_model_cfg["dropout"] = trans_model_cfg_copy['dropout']
        sml_trans_model_cfg["drop_path_rate"] = trans_model_cfg_copy['drop_path_rate']
        sml_trans_model_cfg['n_cls'] = trans_model_cfg_copy['n_cls']
        sml_trans_model_cfg['decoder'] = decoder_cfg
        sml_trans_model_cfg['num_output_trans'] = num_output_trans_sml

        # ---- entrance flow to network ----

        self.init_block = nn.Sequential(
            BNRconv3x3(in_planes=3, out_planes=basic_0, stride=1),
            BNRconv3x3(in_planes=basic_0, out_planes=basic_0, stride=1),
            BNRconv3x3(in_planes=basic_0, out_planes=basic_0, stride=1),
        )
        self.level1 = BNRconv3x3(in_planes=basic_0, out_planes=basic_0, stride=2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.BR_1 = nn.Sequential(nn.BatchNorm2d(basic_0 + basic_0), nn.PReLU(basic_0 + basic_0))


        big_trans_model_cfg['in_channels'] = basic_0 + basic_0 # this would have been 3 otherwise. but its not even set
        sml_trans_model_cfg['in_channels'] = basic_0 + basic_0 # this would have been 3 otherwise. but its not even set
        self.big_trans = create_transformerV5(big_trans_model_cfg, decoder='linear')
        self.sml_trans = create_transformerV5(sml_trans_model_cfg, decoder='linear')
        

        exit(1)


    def forward(self, input):
        input = self.init_block(input);                                         print(f'[input]: \t {input.shape}')
        output0 = self.level1(input);                                           print(f'[output0]:\t {output0.shape}')
        inp1 = self.sample1(input);                                             print(f'[inp1]:\t\t {inp1.shape}')
        output0_cat = self.BR_1(torch.cat([output0, inp1], dim=1));             print(f'[output0_cat]:\t {output0_cat.shape}')
        return output0_cat

if __name__ == '__main__':
    model = MultiLevelInputFusionNetwork()
    input = torch.randn((1, 3, 512, 512))
    output = model(input)

    count_parameters(model)    


    from seg.model.transformer.ViT import VisionTransformer

    trans = VisionTransformer(
        image_size = (256, 256),
        patch_size = 16, 
        n_layers = 12,
        d_model = 384,
        d_ff = 4 * 384,
        n_heads = 6,
        n_cls=64,
        channels=64
    )

    count_parameters(trans)