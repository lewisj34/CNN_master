from logging import Logger
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torchsummary import summary 
from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs

from .ViT import VisionTransformer
from .decoder import DecoderLinear, MaskTransformer
from .decoder_new import DecoderMultiClassDilatioaAndRFB, DecoderMultiClassDilation, DecoderMultiClassDilationAndSCSE, DecoderPlus, DecoderMultiClass
from .decoder_new_new import DecoderMultiClassRFB
from .utils import checkpoint_filter_fn, padding, unpadding

def create_decoder(d_model, patch_size, decoder_cfg, branch=''):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = d_model
    decoder_cfg["patch_size"] = patch_size

    if "linear" in name:
        if branch == 'fusion':
            print("fusion decoder")
            decoder = DecoderLinear(
                n_cls = decoder_cfg['n_cls'] + 31, 
                patch_size = decoder_cfg['patch_size'], 
                d_encoder = decoder_cfg['d_encoder']
            )
        else:
            decoder = DecoderLinear(
                n_cls = decoder_cfg['n_cls'], 
                patch_size = decoder_cfg['patch_size'], 
                d_encoder = decoder_cfg['d_encoder']
            )
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder

class TransformerNoEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size,
        decoder,
        n_cls,
        num_outputs_trans=32,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.decoder = decoder
        
        self.t_1_2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(False),
        )
        self.t_1_4 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(False),
        )
        self.t_1_8 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(False),
        )
        self.t_1_16 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(False),
        )
        # manually put in params for this, would need modification for diff
        # input sizes other than a patch size of 16x16, and an output size of 
        # 256x256
        self.use_decoderPlus = True
        self.useDilatedDecoderPlus = True
        if self.use_decoderPlus:
            if self.useDilatedDecoderPlus:
                print(f'\nWARNING in  file: {__file__}: Using DecoderMultiClassDilation in create_transformerV2: ', self.use_decoderPlus)
                self.decoderPlus = DecoderMultiClassDilation(
                    input_size=(16,16), 
                    in_chans=num_outputs_trans,
                    output_size=(256,256),
                    inter_chans=32,
                    out_chans=n_cls,
                    dilation1=1,
                    dilation2=3,
                )
            else:
                print(f'Just using decoderPlus (no dilation / no RFB): ', self.use_decoderPlus)
                self.decoderPlus = DecoderMultiClass(
                    input_size=(16,16), 
                    in_chans=num_outputs_trans,
                    output_size=(256,256),
                    inter_chans=32,
                    out_chans=n_cls,
                )

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, x):
        H = 512
        W = 512

        # Output should be: [N, input_chans, 32, 32]
        x_1_2 = self.t_1_2(x)
        x_1_4 = self.t_1_4(x)
        x_1_8 = self.t_1_8(x)
        x_1_16 = self.t_1_16(x)

        # print(f'output of decoder: {masks.shape}')
        # interpolate up to partial sizes and assert they'll be successful  
        if self.patch_size == 16:
            self.x_1_2 = F.interpolate(x_1_2, size=(H // 2, W // 2), 
                mode='bilinear') # output: torch.Size([N, 1, 128, 128])
            self.x_1_4 = F.interpolate(x_1_4, size=(H // 4, W // 4), 
                mode='bilinear') # output: torch.Size([N, 1, 64, 64])
            self.x_1_8 = F.interpolate(x_1_8, size=(H // 8, W // 8), 
                mode='bilinear') # output: torch.Size([N, 1, 32, 32])
            self.x_1_16 = x_1_16

        if self.use_decoderPlus:
            masks = self.decoderPlus(x_1_16)
        return masks

def create_transformerNoTransEncoder(model_cfg, decoder='linear', in_channels_cnn=512):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]
    decoder_cfg['d_model'] = model_cfg['d_model']


    num_output_trans = model_cfg['num_output_trans']
    print(f'num_output_trans in create_transformer: {num_output_trans}')
    # num_output_trans = 64

    model_cfg['n_cls'] = num_output_trans
    decoder_cfg['n_cls'] = num_output_trans
    decoder = create_decoder(model_cfg['d_model'], model_cfg['patch_size'], decoder_cfg)
    model = TransformerNoEncoder(in_channels_cnn, 16, decoder, n_cls=model_cfg["n_cls"], num_outputs_trans=num_output_trans)

    return model