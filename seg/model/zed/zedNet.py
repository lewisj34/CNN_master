import torch 
import torch.nn as nn 

from .parts import * 

class zedNet(nn.Module):
    def __init__(
        self,
        n_channels, 
        n_classes,
        patch_size,
        bilinear=True,
        attention=True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.patch_size = patch_size 
        assert self.patch_size == 16 or self.patch_size == 32, \
            'Patch size must be {16, 32}' 

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        if self.patch_size == 32:
            self.down5 = Down(1024 // factor, 1024 // factor)
            if attention==True:
                raise ValueError(f'patch_size 32 not supported for attention yet ')

        self.up1 = UpAttention(1024 // factor, 1024 // factor, 512 // factor, bilinear)
        self.up2 = UpAttention(512 // factor, 256, 256 // factor, bilinear)
        self.up3 = UpAttention(256 // factor, 128, 128 // factor, bilinear)
        self.up4 = UpAttention(128 // factor, 64, 64, bilinear)

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



