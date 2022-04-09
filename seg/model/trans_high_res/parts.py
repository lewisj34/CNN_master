import torch.nn as nn

class Elementary(nn.Module):
    """
    Uses same number of channels across two standard 3 by 3 convolutions, that 
    results in the exact same dimensions throughout, that is with the same
    height and width dims, and the number of `out_chans` throughout all layers. 
    """
    def __init__(
        self,
        in_chans,
        out_chans,
        bn_momentum=1,
        relu_inplace=False,
        fuse_type=None,
    ):
        super(Elementary, self).__init__() 
        self.conv1 = nn.Conv2d(
            in_chans, 
            out_chans, 
            kernel_size=3, 
            stride=1,
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            out_chans, 
            momentum=bn_momentum
        )
        self.relu1 = nn.ReLU(
            inplace=relu_inplace
        )
        self.conv2 = nn.Conv2d(
            out_chans, 
            out_chans, 
            kernel_size=3, 
            stride=1,
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            out_chans, 
            momentum=bn_momentum,
        )
        self.relu2 = nn.ReLU(
            inplace=relu_inplace
        )
        self.fuse_type = fuse_type

    def forward(self, x, g=None):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.fuse_type is not None:
            if g is not None:
                if self.fuse_type == 'sum':
                    print(f'Double check that the input to this network has been normalized in some capacity or else this will screw up. Location: {__file__}')
                    out = out + g
                elif self.fuse_type == 'attn':
                    out = attn(x,g)
                else:
                    raise ValueError(f'fuse_type: {self.fuse_type} not impl.')
            else:
                raise ValueError(f'No input for variable g in forward, yet fuse_type has been selected. Ensure there is an input in forward for this value. ')

        out = self.relu2(out)
        
        return out

class iElementaryPlusInterChannels(nn.Module):
    """
    Same as Elementary in the sense that H and W don't change, but this module
    allows for the application of an intermediate amount of channels starting
    with the first convolution. Also has an attention mechanism. 
    """
    def __init__(
        self,
        in_chans,
        inter_chans,
        out_chans,
        bn_momentum=1,
        relu_inplace=False,
        fuse_type='sum',
    ):
        super(iElementaryPlusInterChannels, self).__init__() 
        self.conv1 = nn.Conv2d(
            in_chans, 
            inter_chans, 
            kernel_size=3, 
            stride=1,
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            inter_chans, 
            momentum=bn_momentum
        )
        self.relu1 = nn.ReLU(
            inplace=relu_inplace
        )
        self.conv2 = nn.Conv2d(
            inter_chans, 
            inter_chans, 
            kernel_size=3, 
            stride=1,
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            inter_chans, 
            momentum=bn_momentum,
        )
        self.relu2 = nn.ReLU(
            inplace=relu_inplace
        )
        self.conv3 = nn.Conv2d(
            inter_chans, 
            out_chans, 
            kernel_size=3, 
            stride=1,
            padding=1, 
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(
            out_chans, 
            momentum=bn_momentum,
        )
        self.relu3 = nn.ReLU(
            inplace=relu_inplace
        )
        self.fuse_type = fuse_type

    def forward(self, x, g=None):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if g is not None:
            if self.fuse_type == 'sum':
                print(f'Double check that the input to this network has been normalized in some capacity or else this will screw up. Location: {__file__}')
                out = out + g
            elif self.fuse_type == 'attn':
                out = attn(x,g)
            else:
                raise ValueError(f'fuse_type: {self.fuse_type} not impl.')

        out = self.relu3(out)
        
        return out

class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_chans,
        out_chans, 
        stride=1, 
        bn_momentum=0.1, 
        relu_inplace=False, 
        downsample=None
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_chans, 
            out_chans, 
            kernel_size=3, 
            stride=stride,
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            out_chans, 
            momentum=bn_momentum
        )
        self.relu = nn.ReLU(
            inplace=relu_inplace
        )
        self.conv2 = nn.Conv2d(
            out_chans, 
            out_chans, 
            kernel_size=3, 
            stride=stride,
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            out_chans, 
            momentum=bn_momentum
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out = out + residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    def __init__(
        self, 
        in_chans, 
        inter_chans,
        out_chans, 
        stride=1, 
        bn_momentum=0.1, 
        relu_inplace=False, 
        downsample=None,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_chans, 
            inter_chans, 
            kernel_size=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            inter_chans, 
            momentum=bn_momentum
        )
        self.conv2 = nn.Conv2d(
            inter_chans, 
            inter_chans, 
            kernel_size=3, 
            stride=stride,
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(
            inter_chans, 
            momentum=bn_momentum
        )
        self.conv3 = nn.Conv2d(
            inter_chans, 
            out_chans, 
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(
            out_chans,
            momentum=bn_momentum
        )
        self.relu = nn.ReLU(
            inplace=relu_inplace
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out = out + residual
        out = self.relu(out)

        return out

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

