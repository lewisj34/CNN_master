import torch 
import torch.nn as nn

from .parts import * 

class zedNetDWSepWithCCMinAllOfItNoPSEncoder(nn.Module):
    def __init__(
        self,
        n_channels_in, 
        n_classes,
        patch_size,
        bilinear=True,
        attention=True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.patch_size = patch_size 
        assert self.patch_size == 16 or self.patch_size == 32, \
            'Patch size must be {16, 32}' 

        self.fuseAdd = nn.Sequential(
            nn.Conv2d(n_channels_in, n_channels_in, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_channels_in),
            nn.SiLU(False),
        )

        factor = 2

        self.up1 = UpDWSepCCMReverse(n_channels_in, 512 // factor, bilinear)
        self.up2 = UpDWSepCCMReverse(256, 256 // factor, bilinear)
        self.att1 = SCSEModule(in_channels=256//factor, reduction=16)
        self.up3 = UpDWSepCCMReverse(128, 128 // factor, bilinear)
        self.att2 = NouveauAttention(64, reduction=2, AvgPoolKernelSize=11, AvgPoolPadding=5)
        self.up4 = UpDWSepCCM(64, 64, bilinear)
        self.att3 = NouveauAttention(64, reduction=2, AvgPoolKernelSize=31, AvgPoolPadding=15)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):      
        self.x_1_16 = self.fuseAdd(x); # print(f'x_1_16: \t {self.x_1_16.shape}')
        x = self.up1(x); self.x_1_8 = x; # print(f'x_1_8: \t {self.x_1_8.shape}')
        x = self.up2(x); self.x_1_4 = x; # print(f'x_1_4: \t {self.x_1_4.shape}')
        x = self.att1(x)
        x = self.up3(x); self.x_1_2 = x; # print(f'x_1_2: \t {self.x_1_2.shape}')
        x = self.att2(x)
        x = self.up4(x)
        x = self.att3(x)
        logits = self.outc(x)
        return logits

    # def get_dimensionsNoPSEncoder(self, N_in, C_in, H_in, W_in, printXDimensions=True):
    #     dummy_tensor = torch.zeros(N_in, C_in, H_in, W_in)
    #     x = self.forward(dummy_tensor)
    #     if printXDimensions:
    #         print(f'Running a forward pass of UNet')
    #         print(f'self.x_1_2.shape: {self.x_1_2.shape}')
    #         print(f'self.x_1_4.shape: {self.x_1_4.shape}')
    #         print(f'self.x_1_8.shape: {self.x_1_8.shape}')
    #         print(f'self.x_1_16.shape: {self.x_1_16.shape}')
    #         if self.patch_size == 32:
    #             print(f'self.x_1_32.shape: {self.x_1_32.shape}')
    #     del dummy_tensor

class zedNetNoPSDecoder(nn.Module):
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
        self.down1 = DownDWSepCCM(64, 128)
        self.down2 = DownDWSepCCM(128, 256)
        self.down3 = DownDWSepCCM(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownDWSepCCM(512, 1024 // factor)

        self.up1 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024 // factor, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU6(True),
        )
        self.outc = OutConv(128, n_classes)

    def forward(self, x):       # [1, 3, 256, 256]      [1, 3, 512, 512]        
        x1 = self.inc(x);       # [1, 64, 256, 256]     [1, 64, 512, 512]  
        x2 = self.down1(x1);    # [1, 128, 128, 128]    [1, 128, 256, 256]  
        x3 = self.down2(x2);    # [1, 256, 64, 64]      [1, 256, 128, 128]  
        x4 = self.down3(x3);    # [1, 512, 32, 32]      [1, 512, 64, 64]  
        x5 = self.down4(x4);    # [1, 512, 16, 16]      [1, 512, 32, 32]  
        self.x_1_2 = x2; 
        self.x_1_4 = x3; 
        self.x_1_8 = x4; 
        self.x_1_16 = x5; 
                
        x = self.up1(x5)
        x = self.conv1(x)
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


if __name__ == "__main__":
    # model = zedNetDWSepWithCCMinAllOfItNoPSEncoder(
    #     n_channels_in=64,
    #     n_classes=1,
    #     patch_size=16,
    # ).cuda()
    # x = torch.randn((2, 64, 32, 32), device='cuda')
    # y = model(x)
    model = zedNetNoPSDecoder(
        n_channels=3,
        n_classes=1,
        patch_size=16,
    ).cuda()
    x = torch.randn((2, 3, 512, 512), device='cuda')
    y = model(x)
    print(f' output of model.shape: {y.shape} ')

"""
python -m seg.model.zed.AblationStudiesZedNet
"""