import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torchsummary import summary 
from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs

from .ViT import VisionTransformer
from .decoder import DecoderLinear, MaskTransformer
from .decoder_new import DecoderPlus, DecoderMultiClass
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

class TransformerV2(nn.Module):
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
        
        # manually put in params for this, would need modification for diff
        # input sizes other than a patch size of 16x16, and an output size of 
        # 256x256
        self.use_decoderPlus = True
        print(f'Using decoderPlus: ', self.use_decoderPlus)
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
        elif self.patch_size == 32:
            self.x_1_2 = F.interpolate(masks, size=(H // 2, W // 2), 
                mode='bilinear') # output: torch.Size([N, 1, 128, 128])
            self.x_1_4 = F.interpolate(masks, size=(H // 4, W // 4), 
                mode='bilinear') # output: torch.Size([N, 1, 64, 64])
            self.x_1_8 = F.interpolate(masks, size=(H // 8, W // 8), 
                mode='bilinear') # output: torch.Size([N, 1, 32, 32])
            self.x_1_16 = F.interpolate(masks, size=(H // 16, W // 16), 
                mode='bilinear')
            self.x_1_32 = masks

            printXDimensions = False
            if printXDimensions:
                print(f'self.x_1_2.shape: {self.x_1_2.shape}')
                print(f'self.x_1_4.shape: {self.x_1_4.shape}')
                print(f'self.x_1_8.shape: {self.x_1_8.shape}')
                print(f'self.x_1_16.shape: {self.x_1_16.shape}')
                print(f'self.x_1_32.shape: {self.x_1_32.shape}')
                print(f'patch_size: {self.patch_size}')

        if self.use_decoderPlus:
            masks = self.decoderPlus(masks)
        else:
            masks = F.interpolate(masks, size=(H, W), mode="bilinear") # output: torch.Size([16, 1, 256, 256])
        masks = unpadding(masks, (H_ori, W_ori)) # output: torch.Size([16, 1, 256, 256])
        return masks


def create_transformerV2(model_cfg, decoder='linear'):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]
    decoder_cfg['d_model'] = model_cfg['d_model']

    num_output_trans = 64
    model_cfg['n_cls'] = num_output_trans
    decoder_cfg['n_cls'] = num_output_trans
    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = TransformerV2(encoder, decoder, n_cls=model_cfg["n_cls"], num_outputs_trans=num_output_trans)

    return model
