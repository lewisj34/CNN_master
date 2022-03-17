import torch 
import torch.nn as nn 
import torchvision 

from torch.nn import functional as F


ALLOWABLE_CNN_BACKBONES = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 
    'resnet152', 'vgg16', 'vgg19', 'densenet121', 'densenet161', 'densenet169', 
    'densenet201', 'unet_encoder', None]

def get_backbone(name, pretrained=True):
    """ 
    Loading backbone, defining names for skip-connections and encoder output. 
    """
    # loading backbone model
    if name == 'resnet18':
        backbone = torchvision.models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        backbone = torchvision.models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        backbone = torchvision.models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        backbone = torchvision.models.resnet101(pretrained=pretrained)
    elif name == 'resnet152':
        backbone = torchvision.models.resnet152(pretrained=pretrained)
    elif name == 'vgg16':
        backbone = torchvision.models.vgg16_bn(pretrained=pretrained).features
    elif name == 'vgg19':
        backbone = torchvision.models.vgg19_bn(pretrained=pretrained).features
    # elif name == 'inception_v3':
    #     backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
    elif name == 'densenet121':
        backbone = torchvision.models.densenet121(pretrained=True).features
    elif name == 'densenet161':
        backbone = torchvision.models.densenet161(pretrained=True).features
    elif name == 'densenet169':
        backbone = torchvision.models.densenet169(pretrained=True).features
    elif name == 'densenet201':
        backbone = torchvision.models.densenet201(pretrained=True).features
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    # specifying skip feature and output names
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    elif name == 'vgg16':
        # TODO: consider using a 'bridge' for VGG models, there is just a MaxPool between last skip and backbone output
        feature_names = ['5', '12', '22', '32', '42']
        backbone_output = '43'
    elif name == 'vgg19':
        feature_names = ['5', '12', '25', '38', '51']
        backbone_output = '52'
    # elif name == 'inception_v3':
    #     feature_names = [None, 'Mixed_5d', 'Mixed_6e']
    #     backbone_output = 'Mixed_7c'
    elif name.startswith('densenet'):
        feature_names = [None, 'relu0', 'denseblock1', 'denseblock2', 'denseblock3']
        backbone_output = 'denseblock4'
    elif name == 'unet_encoder':
        feature_names = ['module1', 'module2', 'module3', 'module4']
        backbone_output = 'module5'
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    return backbone, feature_names, backbone_output

class CNN_BRANCH_WITH_BACKBONE(nn.Module):
    def __init__(
        self, 
        n_channels, 
        n_classes, 
        patch_size, 
        backbone_name=None,
        bilinear=True,
        pretrained=True,
        with_fusion=False,
        with_attention=False, 
        with_superficial=False,
        input_size=256,
    ):
        super(CNN_BRANCH_WITH_BACKBONE, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.patch_size = patch_size 
        self.backbone_name = backbone_name
        self.bilinear = bilinear # i dont think this is used honestly... can prob take it out...
        self.pretrained = pretrained
        self.with_fusion = with_fusion
        self.with_attention = with_attention
        self.with_superficial = with_superficial
        self.input_size = input_size

        assert self.input_size == 256 or self.input_size == 512, 'input_size must be manually set to 256 or 512'

        assert self.patch_size == 16 or self.patch_size == 32, \
            'Patch size must be {16, 32}'
        assert backbone_name in ALLOWABLE_CNN_BACKBONES

        # build encoder 
        print(f'backbone chosen: {backbone_name}')
        print(f'manually inserting input size in {__file__} for CNN_BRANCH_WITH_BACKBONE')
        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(
            backbone_name, pretrained=pretrained)
        shortcut_chs, bb_out_chs = self.infer_skip_channels()


        encoder_freeze = False
        if encoder_freeze:                          # from model input in prev. 
            self.freeze_encoder()                   # from model input in prev.

        # build decoder 
        decoder_filters = (256, 128, 64, 32, 16)    # from model input in prev. 
        parametric_upsampling = True                # from model input in prev. 
        decoder_use_batchnorm = True                # from model input in prev. 
        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[:len(self.shortcut_features)]  # avoiding having more blocks than skip connections
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            print('upsample_blocks[{}] in: {}   out: {}'.format(i, filters_in, filters_out))
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=decoder_use_batchnorm,
                                                      with_attn=self.with_attention,
                                                      with_superficial=self.with_superficial))
            
        self.final_conv = nn.Conv2d(decoder_filters[-1], n_classes, kernel_size=(1, 1))


        self.replaced_conv1 = False  # for accommodating  inputs with different number of channels later


    def freeze_encoder(self):
        """ 
        Freezing encoder parameters, the newly initialized decoder parameters 
        are remaining trainable. 
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *input):
        # print(f'Beginning of encoder in {__file__}')
        x, features = self.forward_backbone(*input)
        # print(f'End of backbone.')

        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]; 
            x = upsample_block(x, skip_features)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            # print(f'name, child, output of forward backbone: {name, x.shape}')
                
            if name in self.shortcut_features:
                features[name] = x; # print(f'name in shortcut features: {name}')
                if self.input_size == 256:
                    if x.shape[3] == 16:
                        self.x_1_16_coarse = x; # print(f'self.x_1_16_coarse.shape: {self.x_1_16_coarse.shape}')
                    elif x.shape[3] == 32:
                        self.x_1_8_coarse = x; # print(f'self.x_1_8_coarse.shape: {self.x_1_8_coarse.shape}')
                    elif x.shape[3] == 64:
                        self.x_1_4_coarse = x; # print(f'self.x_1_4_coarse.shape: {self.x_1_4_coarse.shape}')
                    elif x.shape[3] == 128:
                        self.x_1_2_coarse = x; # print(f'self.x_1_2_coarse.shape: {self.x_1_2_coarse.shape}')
                if self.input_size == 512:
                    if x.shape[3] == 16:
                        self.x_1_32_coarse = x; # print(f'self.x_1_32_coarse.shape: {self.x_1_32_coarse.shape}')
                    elif x.shape[3] == 32:
                        self.x_1_16_coarse = x; # print(f'self.x_1_16_coarse.shape: {self.x_1_16_coarse.shape}')
                    elif x.shape[3] == 64:
                        self.x_1_8_coarse = x; # print(f'self.x_1_8_coarse.shape: {self.x_1_8_coarse.shape}')
                    elif x.shape[3] == 128:
                        self.x_1_4_coarse = x; # print(f'self.x_1_4_coarse.shape: {self.x_1_4_coarse.shape}')
                    elif x.shape[3] == 256:
                        self.x_1_2_coarse = x; # print(f'self.x_1_2_coarse.shape: {self.x_1_2_coarse.shape}')    

            if name == self.bb_out_name:
                break

        return x, features
    
    def infer_skip_channels(self):
        """
        Getting the number of channels at skip connections and at the output of 
        the encoder. 
        """
        x = torch.zeros(1, 3, 256, 256)
        has_fullres_features = self.backbone_name.startswith('vgg') or self.backbone_name == 'unet_encoder'
        channels = [] if has_fullres_features else [0]  # only VGG has features at full resolution

        # forward run in backbone to count channels (dirty solution but works for *any* Module)
        total_output_shapes = list()
        for name, child in self.backbone.named_children():
            x = child(x)
            total_output_shapes.append(x.shape)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break

        # for i in range(len(total_output_shapes)):
        #     print(f'i: {i}, shape: {total_output_shapes[i]}')
        # for i in range(len(channels)):
        #     print(f'i: {i}, shape: {channels[i]}')
        # print(f'out_channels: {out_channels}')

        return channels, out_channels   

    def get_dimensions(self, N_in, C_in, H_in, W_in, printXDimensions=True):
        dummy_tensor = torch.zeros(N_in, C_in, H_in, W_in)
        x = self.forward(dummy_tensor)
        if printXDimensions:
            print(f'Running a forward pass of UNet')
            print(f'self.x_1_2.shape: {self.x_1_2.shape}')
            print(f'self.x_1_4.shape: {self.x_1_4.shape}')
            print(f'self.x_1_8.shape: {self.x_1_8.shape}')
            print(f'self.x_1_16.shape: {self.x_1_16.shape}')
            if self.patch_size == 32:
                print(f'self.x_1_32.shape: {self.x_1_32.shape}')
        del dummy_tensor

class UpsampleBlock(nn.Module):

    # TODO: separate parametric and non-parametric classes?
    # TODO: skip connection concatenated OR added

    def __init__(
        self, 
        ch_in, 
        ch_out=None, 
        skip_in=0, 
        use_bn=True, 
        parametric=False, 
        with_attn=False,
        with_superficial=False,
    ):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric
        self.with_attn = with_attn
        self.with_superficial = with_superficial

        ch_out = ch_in/2 if ch_out is None else ch_out

        # first convolution: either transposed conv, or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        # modified JL here 
        if self.with_attn:
            if skip_in is not 0:
                conv2_in = ch_out
            else:
                conv2_in = ch_out if not parametric else ch_out + skip_in
        else:
            conv2_in = ch_out if not parametric else ch_out + skip_in

        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

        # attention 
        from seg.model.attention.grid_attention import GridAttentionBlock2D_TORR as AttentionBlock2D
        from seg.model.attention.grid_attention import init_weights
        if self.with_attn:
            if skip_in is not 0:
                print(f'Creating attention block. in_channels, gating_channels, inter_channels: {ch_in, skip_in, ch_out}')
                self.attn = AttentionBlock2D(
                    in_channels = ch_in,
                    gating_channels = skip_in,
                    inter_channels = ch_out,
                    mode='concatenation_softmax',
                    sub_sample_factor=(1,1),
                )
                for m in self.attn.modules():
                    if isinstance(m, nn.Conv2d):
                        init_weights(m, init_type='kaiming')
                    elif isinstance(m, nn.BatchNorm2d):
                        init_weights(m, init_type='kaiming')

        # other stuffs
        from seg.model.siddnet.siddnet import SuperficialModule_subblock, CCMSubBlock, RCCMModule
        if self.with_superficial:
            self.super_mod1 = SuperficialModule_subblock(nIn=ch_out, d=1, kSize=3, dkSize=3)
            self.super_mod2 = SuperficialModule_subblock(nIn=ch_out, d=1, kSize=3, dkSize=3)
            self.super_mod3 = SuperficialModule_subblock(nIn=ch_out, d=1, kSize=3, dkSize=3)
            self.ccm1 = CCMSubBlock(nIn = ch_out, nOut = ch_out, kSize = 3, d = 2)
            self.ccm2 = CCMSubBlock(nIn = ch_out, nOut = ch_out, kSize = 3, d = 2)
            self.ccm3 = CCMSubBlock(nIn = ch_out, nOut = ch_out, kSize = 3, d = 2)

    def forward(self, x, skip_connection=None):
        if self.with_attn:
            if skip_connection is not None:
                # print(f'x.shape, skip_cxn.shape: {x.shape, skip_connection.shape}')
                self.attn(x, skip_connection)

        x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                                                             align_corners=None)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if self.with_superficial:
            x = self.super_mod1(x)
            x = self.super_mod1(x)
            x = self.super_mod1(x)
            x = self.ccm1(x)
            x = self.ccm2(x)
            x = self.ccm3(x)

        if not self.with_attn:
            if skip_connection is not None:
                x = torch.cat([x, skip_connection], dim=1)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary 
    model = CNN_BRANCH_WITH_BACKBONE(
        n_channels = 3, 
        n_classes = 1,
        patch_size = 16, 
        backbone_name='resnet18',
        bilinear=True,
        pretrained=True,
        with_fusion=True,
        input_size=256
    )
    summary(
        model=model.cpu(),
        input_size=(3, 256, 256),
        batch_size=2,
        device='cpu'
    )

# seg.model.CNN.CNN_backboned