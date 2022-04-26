import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from segmentation_models_pytorch.base import modules as md

from seg.model.segmentation_models_pytorch.encoders.efficientnet import EfficientNetEncoder

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = SCSEModule(in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = SCSEModule(in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # print(f'x: {x.shape, type(x)}')
        # print(f'cSE(x): {self.cSE(x).shape, type(self.cSE(x))},')
        # print(f'sSE(x): {self.sSE(x).shape, type(self.sSE(x))},')
        return x * self.cSE(x) + x * self.sSE(x)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = SCSEModule(in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = SCSEModule(in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class SCSE_Decoder(nn.Module):
    def __init__(
        self,
        encoder_channels, 
        decoder_channels, 
        n_blocks,
        center=False,
    ):
        """
        @encoder_channels: number of output channels in each layer of encoder
        @decoder_channels: number of output channels in each layer of decoder
        """
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=True)
        else:
            self.center = nn.Identity()

        assert len(in_channels) == len(skip_channels) == len(out_channels)

        self.blocks = nn.ModuleList(
            DecoderBlock(in_ch, skip_ch, out_ch) 
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        )
        print(f'inputs:')
        print(f'\tencoder_channels: {encoder_channels}')
        print(f'\tdecoder_channels: {decoder_channels}')

        print(f'Determined args:')
        print(f'\thead_channels: {head_channels}')
        print(f'\tin_channels: {in_channels}')
        print(f'\tskip_channels: {skip_channels}')
        print(f'\tout_channels: {out_channels}')

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)

class Activation(nn.Module):
    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)

from timm.models.layers import trunc_normal_
def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class NewCNN(nn.Module):
    def __init__(
        self,
        encoder_channels=(3, 64, 48, 80, 224, 640),
        decoder_channels=(256, 128, 64, 32, 16), 
        num_classes=1,
        activation=None,
    ):
        super().__init__()
        self.encoder = EfficientNetEncoder(
            stage_idxs=(11, 18, 38, 55),
            out_channels=encoder_channels,
            model_name='efficientnet-b7',
            depth=5,
        )
        self.decoder = SCSE_Decoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            activation=activation,
            kernel_size=3,
        )
        print(f'Initializing weights...')
        self.apply(init_weights)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    
    # def init_weights(self):
    #     for m in self.decoder.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    #         elif isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #     for m in self.encoder.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    #         elif isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #     for m in self.segmentation_head.modules():
    #         if isinstance(m, (nn.Linear, nn.Conv2d)):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    # net = SCSE_Decoder(
    #     encoder_channels = (3, 64, 48, 80, 224, 640),
    #     decoder_channels = (256, 128, 64, 32, 16),
    #     n_blocks=5,
    # )

    net = NewCNN(
        encoder_channels = (3, 64, 48, 80, 224, 640),
        decoder_channels = (256, 128, 64, 32, 16),
        num_classes=1,
    ).cuda()

    x = torch.randn((1, 3, 256, 256)).cuda()
    output = net(x)