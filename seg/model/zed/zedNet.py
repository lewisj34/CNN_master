import torch 
import torch.nn as nn

from seg.utils.check_parameters import count_parameters 

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

        skip_cxns_chans = [1024 // factor, 256, 128, 64]
        # using SCSE attention for all of them 
        # self.up1 = UpAttention(1024 // factor, skip_cxns_chans[0], 512 // factor, bilinear)
        # self.up2 = UpAttention(512 // factor, skip_cxns_chans[1], 256 // factor, bilinear)
        # self.up3 = UpAttention(256 // factor, skip_cxns_chans[2], 128 // factor, bilinear)
        # self.up4 = UpAttention(128 // factor, skip_cxns_chans[3], 64, bilinear)

        # using SCSE for just the last one and normal for the rest 

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.att1 = SCSEModule(in_channels=256//factor, reduction=16)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.att2 = NouveauAttention(64, reduction=2, AvgPoolKernelSize=11, AvgPoolPadding=5)
        self.up4 = Up(128, 64, bilinear)
        self.att3 = NouveauAttention(64, reduction=2, AvgPoolKernelSize=31, AvgPoolPadding=15)

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
        x = self.att1(x)
        x = self.up3(x, x2)
        x = self.att2(x)
        x = self.up4(x, x1)
        x = self.att3(x)
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

class zedNetMod(nn.Module):
    def __init__(
        self,
        n_channels, 
        n_classes,
        num_output_channels = [64, 128, 256, 512, 512, 256, 128, 64, 64],
        patch_size=16,
        bilinear=True,
    ):
        """
        Same as before but just in this case we accept params to modify the 
        number of channels. 
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.patch_size = patch_size 
        assert self.patch_size == 16, f'patch_size: {patch_size}. Must be 16. '

        self.inc = DoubleConv(n_channels, num_output_channels[0])
        self.down1 = Down(num_output_channels[0], num_output_channels[1])
        self.down2 = Down(num_output_channels[1], num_output_channels[2])
        self.down3 = Down(num_output_channels[2], num_output_channels[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(num_output_channels[3], num_output_channels[4])

        self.up1 = Up(num_output_channels[4] + num_output_channels[3], num_output_channels[5], bilinear)
        self.up2 = Up(num_output_channels[5] + num_output_channels[2], num_output_channels[6], bilinear)
        self.att1 = SCSEModule(in_channels=num_output_channels[6], reduction=16)
        self.up3 = Up(num_output_channels[6] + num_output_channels[1], num_output_channels[7], bilinear)
        self.att2 = NouveauAttention(num_output_channels[7], reduction=2, AvgPoolKernelSize=11, AvgPoolPadding=5)
        self.up4 = Up(num_output_channels[7] + num_output_channels[0], num_output_channels[8], bilinear)
        self.att3 = NouveauAttention(num_output_channels[8], reduction=2, AvgPoolKernelSize=31, AvgPoolPadding=15)

        self.outc = OutConv(num_output_channels[8], n_classes)

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
        x = self.att1(x)
        x = self.up3(x, x2)
        x = self.att2(x)
        x = self.up4(x, x1)
        x = self.att3(x)
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

class zedNetDWSep(nn.Module):
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

        self.inc = DoubleConvDWSep(n_channels, 64)
        self.down1 = DownDWSep(64, 128)
        self.down2 = DownDWSep(128, 256)
        self.down3 = DownDWSep(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownDWSep(512, 1024 // factor)

        if self.patch_size == 32:
            self.down5 = DownDWSep(1024 // factor, 1024 // factor)
            if attention==True:
                raise ValueError(f'patch_size 32 not supported for attention yet ')

        skip_cxns_chans = [1024 // factor, 256, 128, 64]
        # using SCSE attention for all of them 
        # self.up1 = UpAttention(1024 // factor, skip_cxns_chans[0], 512 // factor, bilinear)
        # self.up2 = UpAttention(512 // factor, skip_cxns_chans[1], 256 // factor, bilinear)
        # self.up3 = UpAttention(256 // factor, skip_cxns_chans[2], 128 // factor, bilinear)
        # self.up4 = UpAttention(128 // factor, skip_cxns_chans[3], 64, bilinear)

        # using SCSE for just the last one and normal for the rest 

        self.up1 = UpDWSep(1024, 512 // factor, bilinear)
        self.up2 = UpDWSep(512, 256 // factor, bilinear)
        self.att1 = SCSEModule(in_channels=256//factor, reduction=16)
        self.up3 = UpDWSep(256, 128 // factor, bilinear)
        self.att2 = NouveauAttention(64, reduction=2, AvgPoolKernelSize=11, AvgPoolPadding=5)
        self.up4 = UpDWSep(128, 64, bilinear)
        self.att3 = NouveauAttention(64, reduction=2, AvgPoolKernelSize=31, AvgPoolPadding=15)

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
        x = self.att1(x)
        x = self.up3(x, x2)
        x = self.att2(x)
        x = self.up4(x, x1)
        x = self.att3(x)
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


if __name__ == '__main__':
    # from torchsummary import summary
    # summary(model=zedNet(3, 1, 16, True, True).cuda())

    model = zedNetMod(3, 1, num_output_channels=[64, 128, 256, 512, 512, 256, 128, 64, 64], patch_size=16)
    count_parameters(model)
    model = zedNetMod(3, 1, num_output_channels=[64 // 2, 128 // 2, 256 // 2, 512 // 2, 512 // 2, 256 // 2, 128 // 2, 64 // 2, 64 // 2], patch_size=16)
    count_parameters(model)
