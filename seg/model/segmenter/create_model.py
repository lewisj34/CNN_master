import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torchsummary import summary 
from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs

from seg.model.segmenter.ViT import VisionTransformer
from seg.model.segmenter.decoder import DecoderLinear, MaskTransformer
from seg.model.segmenter.decoder_new import DecoderNew
from seg.model.segmenter.utils import checkpoint_filter_fn, padding, unpadding

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

def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(
            n_cls = decoder_cfg['n_cls'], 
            patch_size = decoder_cfg['patch_size'], 
            d_encoder = decoder_cfg['d_encoder']
        )
    elif name == "progressive":
        print(f'Decoder used: DecoderNew')
        decoder = DecoderNew(
            num_classes = decoder_cfg['n_cls'],
            patch_size = decoder_cfg['patch_size'],
            d_model = decoder_cfg['d_model'],
        )
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder

class Transformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder

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

        # print(f'output of encoder: {x.shape}')

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

        masks = F.interpolate(masks, size=(H, W), mode="bilinear") # output: torch.Size([16, 1, 256, 256])
        masks = unpadding(masks, (H_ori, W_ori)) # output: torch.Size([16, 1, 256, 256])
        return masks


def create_transformer(model_cfg, decoder='linear'):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]
    decoder_cfg['d_model'] = model_cfg['d_model']

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = Transformer(encoder, decoder, n_cls=model_cfg["n_cls"])

    return model

if __name__ == '__main__':
    import yaml 
    from pathlib import Path 

    backbone = 'vit_base_patch16_384' # vit_base_patch16_384, vit_base_patch32_384
    decoder = 'progressive'
    image_height = 256
    image_width = 256
    batch_size = 2 
    dropout = 0.
    drop_path_rate = 0.1
    n_cls = 1 

    trans_cfg = yaml.load(open("/home/john/Documents/Dev_Linux/segmentation/trans_isolated/vit_config.yml", "r"), 
        Loader=yaml.FullLoader)
    trans_model_cfg = trans_cfg['model'][backbone]

    if "mask_transformer" in decoder: # always wack, hence default is linear - REWRITE THIS AND TAKE MASK TRANSFORMER OUT 
        decoder_cfg = trans_cfg["decoder"]["mask_transformer"]
        # raise ValueError(f'Were not doing anything but linear decoder.')
    else:
        decoder_cfg = trans_cfg["decoder"][decoder]

    # images
    trans_model_cfg['image_size'] = (image_height, image_width)
    trans_model_cfg["dropout"] = dropout
    trans_model_cfg["drop_path_rate"] = drop_path_rate
    trans_model_cfg['backbone'] = backbone 
    trans_model_cfg['n_cls'] = n_cls
    decoder_cfg['name'] = decoder 
    trans_model_cfg['decoder'] = decoder_cfg

    patch_size = trans_model_cfg['patch_size']
    print(trans_model_cfg)

    n_ = batch_size; c_ = 3; h_ = image_height; w_ = image_width  
    input = torch.rand(n_, c_, h_, w_)
    mask = torch.rand(n_, c_, h_, w_)

    print(f'(input, mask).shape: {input.shape, mask.shape}')
    model = create_transformer(
        model_cfg = trans_model_cfg,
        decoder = decoder
    ).cuda()

    from torchsummary import summary 
    summary(model = model, input_size = (c_, h_, w_),  batch_size = n_)
