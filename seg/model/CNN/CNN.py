import torch
import torch.nn as nn 
import torchvision

from .CNN_parts import DoubleConv, Down, Up, OutConv, DownASPP
from torchsummary import summary 


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
                
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x) 
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

if __name__ == '__main__':
    model = CNN_BRANCH(3, 1, 16, True)
    from torchsummary import summary 
    N, C, H, W = 2, 3, 256, 256
    summary(model.cuda(), input_size=(C, H, W), batch_size=N)

    # dim_list = model.get_dimensions(N, C, H, W)