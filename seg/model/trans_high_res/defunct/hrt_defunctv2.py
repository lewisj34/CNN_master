
class Stage0(nn.Module):
    """
    Just a quick succession of activations, convolutions, and ReLUs to an 
    intermediate size, or, upsample to the original input size, after 
    downsampling to intermediate size with aforementioned layer types. 
    Module contains sequence of layers as following:
        conv1(x)
        bn1(x)
        relu1(x)
        conv2(x)
        bn2(x)
        relu2(x)
    if (upsampled to output size):
        upsample(x) -> input_size
    """
    def __init__(
        self, 
        in_chans=3,
        out_chans=64, 
        BN_momentum=0.1,
        relu_inplace=False,
        full_size=False,
        align_corners=True,
    ):
        super(Stage0, self).__init__()

        self.full_size = full_size 

        self.conv1 = nn.Conv2d(
            in_channels=in_chans,   # 3
            out_channels=out_chans, # 64
            kernel_size=3, 
            stride=2, 
            padding=1, 
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=out_chans,
            momentum=BN_momentum,
        )
        self.relu1 = nn.ReLU(
            inplace=relu_inplace,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_chans,
            out_channels=out_chans,
            kernel_size=3, 
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=out_chans,
            momentum=BN_momentum,
        )
        self.relu2 = nn.ReLU(
            inplace=relu_inplace,
        )
        self.upsample = nn.Upsample(
            scale_factor=4,
            mode = 'bilinear',
            align_corners=align_corners,
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.upsample(x)
        return x 


from .parts import blocks_dict

class Stage1(nn.Module):
    """
    First seqeuence of high-resolution convolutions.
        @num_branches: number of sequences of high resolution convolutions, 
            per module. 1 branch would be 1 sequence of high-res convs. 
            Controls the size at each level. 
        @num_modules: number of progressive convolution modules at each branch
        @num_blocks: 
    """
    def __init__(
        self,
        in_chans,
        out_chans,
        num_branches=1,
        num_blocks=4,
    ):

        # creating layer 1 
        blocks = list()
        blocks.append(Bottleneck(in_chans, out_chans))
        

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Stage(nn.Module):
    """
    Main code for creating a stage. 
        @num_levels: the number of downsampled representation sizes to run 
            through a series of modules for. Type: int. Ex: 3, 3 scales to run 
            series of modules at. 
        @level_depth: the number of modules to run at each level specified above.
            Type: tuple. Ex corresp to above param, (3, 4, 5): 3 modules at
            lowest size feature map (ex: N, C, 64, 64), 4 modules at midsize 
            (ex: N, C, 128, 128), and then 5 modules at highest level, 
            (ex: N, C, 256, 256). Can set channels with next param.
            ** NOTE: therefore num_levels == len(level_depth) **
        @type_module: Type of module to run at each level. Will be same for 
            entire level. However, will be different AT each level. 
        @fuse_type: Defines how to fuse the output from the outputs of each 
            level. Options: {'sum', 'concat'}
        @out_chans: the number of channels to output for the whole stage.
    """
    def __init__(
        self,
        num_levels=3,
        level_depth=(3, 4, 5), 
        type_module=('Bottleneck', 'Basic', 'Basic'),
        fuse_type='sum',
        out_chans=64,
    ):
        super(Stage, self).__init__()
        print(f'Number of levels chosen: {num_levels}')
        print(f'len(mods_per_level): {len(level_depth)}')

        assert num_levels == len(level_depth) == len(type_module)
        assert fuse_type == 'sum' or fuse_type == 'concat'

        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=out_chans, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )

        for i in range(num_levels):
            print(f'i: {i, level_depth[i], type_module[i]}')
    
    def forward(self, x):
        x = self.conv1(x)
        return x 
