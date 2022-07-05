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

def create_vit(model_cfg):
    model_cfg['d_ff'] = 4 * model_cfg['d_model']

    default_cfg = default_cfgs[model_cfg['backbone']]

    model = VisionTransformer(
        image_size = model_cfg['image_size'],
        patch_size = model_cfg['patch_size'],
        n_layers = model_cfg['n_layers'],
        d_model = model_cfg['d_model'],
        d_ff = model_cfg['d_ff'],
        n_heads = model_cfg['n_heads'],
        n_cls = model_cfg['n_cls'],
        dropout = model_cfg['dropout'], 
        drop_path_rate = model_cfg['drop_path_rate'],
        distilled = model_cfg['distilled'],
    )

    if 'deit' in model_cfg['backbone']:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    else:
        load_custom_pretrained(model, default_cfg)

    return model 

def create_decoder(encoder, decoder_cfg, branch=''):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

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

class TransformerNoDecoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        num_outputs_trans=32,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=encoder.n_cls, 
                out_channels=1, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.BatchNorm2d(1),
            nn.ReLU6(),
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]


        # input to decoder: torch.Size([16, 256, 768])
        # output to decoder is always going to be H_orig (or W_orig) // 16 
        masks = self.decoder(x, (H, W)) # output: torch.Size([N, 1, 16, 16]) 

        # print(f'output of decoder: {masks.shape}')
        # interpolate up to partial sizes and assert they'll be successful  
        if self.patch_size == 16:
            self.x_1_2 = F.interpolate(masks, size=(H // 2, W // 2), 
                mode='bilinear') # output: torch.Size([N, 1, 128, 128])
            self.x_1_4 = F.interpolate(masks, size=(H // 4, W // 4), 
                mode='bilinear') # output: torch.Size([N, 1, 64, 64])
            self.x_1_8 = F.interpolate(masks, size=(H // 8, W // 8), 
                mode='bilinear') # output: torch.Size([N, 1, 32, 32])
            self.x_1_16 = masks

            printXDimensions = False
            if printXDimensions:
                print(f'self.x_1_2.shape: {self.x_1_2.shape}')
                print(f'self.x_1_4.shape: {self.x_1_4.shape}')
                print(f'self.x_1_8.shape: {self.x_1_8.shape}')
                print(f'self.x_1_16.shape: {self.x_1_16.shape}')
                print(f'patch_size: {self.patch_size}')
        masks = F.interpolate(masks, size=(H, W), mode="bilinear") # output: torch.Size([16, 1, 256, 256])
        masks = unpadding(masks, (H_ori, W_ori)) # output: torch.Size([16, 1, 256, 256])
        masks = self.out_conv(masks)
        return masks


def create_transformerNoTransDecoder(model_cfg, decoder='linear'):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]
    decoder_cfg['d_model'] = model_cfg['d_model']


    num_output_trans = model_cfg['num_output_trans']
    print(f'num_output_trans in create_transformer: {num_output_trans}')
    # num_output_trans = 64

    model_cfg['n_cls'] = num_output_trans
    decoder_cfg['n_cls'] = num_output_trans
    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = TransformerNoDecoder(encoder, decoder, n_cls=model_cfg["n_cls"], num_outputs_trans=num_output_trans)

    return model
