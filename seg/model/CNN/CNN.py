import torch
import torch.nn as nn 
import torchvision

from seg.utils.check_parameters import count_parameters

from .CNN_parts import DoubleConv, Down, Up, OutConv, DownASPP
from torchsummary import summary 
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

class CNN_BRANCH(nn.Module):
    def __init__(self, n_channels, n_classes, patch_size, use_ASPP=False, bilinear=True):
        super(CNN_BRANCH, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.patch_size = patch_size 
        assert self.patch_size == 16 or self.patch_size == 32, \
            'Patch size must be {16, 32}' 

        # if use ASPP introduce ASPP modules in the downsample operations 
        if use_ASPP:
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = DownASPP(64, 128)
            self.down2 = DownASPP(128, 256)
            self.down3 = DownASPP(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = DownASPP(512, 1024 // factor)
        # just use normal downsampling operations (conv + maxpool)
        else:
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            factor = 2 if bilinear else 1
            self.down4 = Down(512, 1024 // factor)

        if self.patch_size == 32:
            self.down5 = Down(1024 // factor, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

        print(f'Initializing weights...')
        self.apply(init_weights)

    def forward(self, x):       # [1, 3, 256, 256]      [1, 3, 512, 512]        
        x1 = self.inc(x);       # [1, 64, 256, 256]     [1, 64, 512, 512]  
        x2 = self.down1(x1);    # [1, 128, 128, 128]    [1, 128, 256, 256]  
        x3 = self.down2(x2);    # [1, 256, 64, 64]      [1, 256, 128, 128]  
        x4 = self.down3(x3);    # [1, 512, 32, 32]      [1, 512, 64, 64]  
        x5 = self.down4(x4);    # [1, 512, 16, 16]      [1, 512, 32, 32]  
        if self.patch_size == 32:
            x6 = self.down5(x5)
        self.x_1_2 = x2; 
        self.x_1_4 = x3; 
        self.x_1_8 = x4; 
        self.x_1_16 = x5; 
        if self.patch_size == 32:
            self.x_1_32 = x6; 

        printXDimensions = False
        if printXDimensions:
            print(f'self.x_1_2.shape: {self.x_1_2.shape}')
            print(f'self.x_1_4.shape: {self.x_1_4.shape}')
            print(f'self.x_1_8.shape: {self.x_1_8.shape}')
            print(f'self.x_1_16.shape: {self.x_1_16.shape}')
            if self.patch_size == 32:
                print(f'self.x_1_32.shape: {self.x_1_32.shape}')
                
        x = self.up1(x5, x4) #; print(f'output up1, x.shape: {x.shape}')
        x = self.up2(x, x3) #; print(f'output up2, x.shape: {x.shape}')
        x = self.up3(x, x2) #; print(f'output up3, x.shape: {x.shape}')
        x = self.up4(x, x1) #; print(f'output up4, x.shape: {x.shape}')
        logits = self.outc(x) #; print(f'output outc, x.shape: {logits.shape}')
        return logits

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


class modUNet(nn.Module):
    def __init__(
        self, 
        n_channels=3, 
        n_classes=1, 
        patch_size=16, 
        bilinear=True,
        channels=[64, 128, 256, 512, 1024, 512, 256, 128, 64],
    ):
        """
        modifiable unet params
        """
        super(modUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.patch_size = patch_size 
        assert self.patch_size == 16 or self.patch_size == 32, \
            'Patch size must be {16, 32}' 

        self.inc = DoubleConv(n_channels, channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(channels[3], channels[4])

        self.up1 = Up(channels[4] + channels[3], channels[5], bilinear)
        self.up2 = Up(channels[5] + channels[2], channels[6], bilinear)
        self.up3 = Up(channels[6] + channels[1], channels[7], bilinear)
        self.up4 = Up(channels[7] + channels[0], channels[8], bilinear)

        self.outc = OutConv(channels[8], n_classes)

        print(f'Initializing weights...')
        self.apply(init_weights)

    def forward(self, x):       # [1, 3, 256, 256]      [1, 3, 512, 512]        
        x1 = self.inc(x);       # [1, 64, 256, 256]     [1, 64, 512, 512]  
        x2 = self.down1(x1);    # [1, 128, 128, 128]    [1, 128, 256, 256]  
        x3 = self.down2(x2);    # [1, 256, 64, 64]      [1, 256, 128, 128]  
        x4 = self.down3(x3);    # [1, 512, 32, 32]      [1, 512, 64, 64]  
        x5 = self.down4(x4);    # [1, 512, 16, 16]      [1, 512, 32, 32]  
        if self.patch_size == 32:
            x6 = self.down5(x5)
        self.x_1_2 = x2; 
        self.x_1_4 = x3; 
        self.x_1_8 = x4; 
        self.x_1_16 = x5; 
        if self.patch_size == 32:
            self.x_1_32 = x6; 

        printXDimensions = False
        if printXDimensions:
            print(f'self.x_1_2.shape: {self.x_1_2.shape}')
            print(f'self.x_1_4.shape: {self.x_1_4.shape}')
            print(f'self.x_1_8.shape: {self.x_1_8.shape}')
            print(f'self.x_1_16.shape: {self.x_1_16.shape}')
            if self.patch_size == 32:
                print(f'self.x_1_32.shape: {self.x_1_32.shape}')
                
        x = self.up1(x5, x4) #; print(f'output up1, x.shape: {x.shape}')
        x = self.up2(x, x3) #; print(f'output up2, x.shape: {x.shape}')
        x = self.up3(x, x2) #; print(f'output up3, x.shape: {x.shape}')
        x = self.up4(x, x1) #; print(f'output up4, x.shape: {x.shape}')
        logits = self.outc(x) #; print(f'output outc, x.shape: {logits.shape}')
        return logits

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


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class UNetREDUCED(nn.Module):
    def __init__(self, n_channels, n_classes, patch_size, bilinear=True):
        super(UNetREDUCED, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.patch_size = patch_size 
        assert self.patch_size == 16 or self.patch_size == 32, \
            'Patch size must be {16, 32}' 

        factor = 4
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128 // factor)
        self.down2 = Down(128 // factor, 256 // factor)
        self.down3 = Down(256 // factor, 512 // factor)
        factor = 4
        self.down4 = Down(512 // factor, 1024 // factor)

        if self.patch_size == 32:
            self.down5 = Down(1024 // factor, 1024 // factor)

        self.up1 = Up(1024 // factor + 512 // factor, 512 // factor, bilinear)
        self.up2 = Up(512 // factor + 256 // factor, 256 // factor, bilinear)
        self.up3 = Up(256 // factor + 128 // factor, 128 // factor, bilinear)
        self.up4 = Up(128 // factor + 64, 64, bilinear)

        self.outc = OutConv(64, n_classes)

        print(f'Initializing weights...')
        self.apply(init_weights)

    def forward(self, x):       # [1, 3, 256, 256]      [1, 3, 512, 512]        
        x1 = self.inc(x);       # [1, 64, 256, 256]     [1, 64, 512, 512]  
        x2 = self.down1(x1);    # [1, 128, 128, 128]    [1, 128, 256, 256]  
        x3 = self.down2(x2);    # [1, 256, 64, 64]      [1, 256, 128, 128]  
        x4 = self.down3(x3);    # [1, 512, 32, 32]      [1, 512, 64, 64]  
        x5 = self.down4(x4);    # [1, 512, 16, 16]      [1, 512, 32, 32]  
        if self.patch_size == 32:
            x6 = self.down5(x5)
        self.x_1_2 = x2; 
        self.x_1_4 = x3; 
        self.x_1_8 = x4; 
        self.x_1_16 = x5; 
        if self.patch_size == 32:
            self.x_1_32 = x6; 

        printXDimensions = False
        if printXDimensions:
            print(f'self.x_1_2.shape: {self.x_1_2.shape}')
            print(f'self.x_1_4.shape: {self.x_1_4.shape}')
            print(f'self.x_1_8.shape: {self.x_1_8.shape}')
            print(f'self.x_1_16.shape: {self.x_1_16.shape}')
            if self.patch_size == 32:
                print(f'self.x_1_32.shape: {self.x_1_32.shape}')
                
        x = self.up1(x5, x4) #; print(f'output up1, x.shape: {x.shape}')
        x = self.up2(x, x3) #; print(f'output up2, x.shape: {x.shape}')
        x = self.up3(x, x2) #; print(f'output up3, x.shape: {x.shape}')
        x = self.up4(x, x1) #; print(f'output up4, x.shape: {x.shape}')
        logits = self.outc(x) #; print(f'output outc, x.shape: {logits.shape}')
        return logits

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


if __name__ == '__main__':
    model = UNetREDUCED(3, 1, 16)
    x = torch.rand((10, 3, 256, 256))
    out = model(x)
    model.get_dimensions(10, 3, 256, 256)

    x2 = torch.rand((10, 3, 256, 256))
    model2 = CNN_BRANCH(3, 1, 16)
    out2 = model2(x2)
    model2.get_dimensions(10, 3, 256, 256)
    count_parameters(model2)

    x3 = torch.rand((10, 3, 256, 256))
    model3 = modUNet(3, 1, 16, True, channels=[64 // 2, 128 // 2, 256 // 2, 512 // 2, 1024 // 2, 512 // 2, 256 // 2, 128 // 2, 64 // 2])
    out3 = model2(x3)
    model3.get_dimensions(10, 3, 256, 256)
    count_parameters(model3)