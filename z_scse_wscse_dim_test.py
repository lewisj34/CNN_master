import torch
import torch.nn as nn 

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # self.cSE = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels, in_channels // reduction, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels // reduction, in_channels, 1),
        #     nn.Sigmoid(),
        # )
        self.cse_GAP = nn.AdaptiveAvgPool2d(1)
        self.cse_conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.cse_relu = nn.ReLU(inplace=True)
        self.cse_conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.cse_sigm = nn.Sigmoid()

        # self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())
        self.sse_conv = nn.Conv2d(in_channels, 1, 1)
        self.sse_sigm = nn.Sigmoid()

    def forward(self, x):
        cse_out = self.cse_GAP(x);          print(f'out scse_cse_GAP: \t {cse_out.shape}')
        cse_out = self.cse_conv1(cse_out);  print(f'out scse_cse_conv1: \t {cse_out.shape}')
        cse_out = self.cse_relu(cse_out);   print(f'out scse_cse_relu: \t {cse_out.shape}')
        cse_out = self.cse_conv2(cse_out);  print(f'out scse_cse_conv2: \t {cse_out.shape}')
        cse_out = self.cse_sigm(cse_out);   print(f'out scse_cse_sigm: \t {cse_out.shape}')

        sse_out = self.sse_conv(x);         print(f'out scse_sse_conv: \t {sse_out.shape}')
        sse_out = self.sse_sigm(sse_out);   print(f'out scse_sse_sigm: \t {sse_out.shape}')

        return x * cse_out + x * sse_out

class NouveauAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, AvgPoolKernelSize=31, AvgPoolPadding=15):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AvgPool2d(kernel_size=AvgPoolKernelSize, stride=1, padding=AvgPoolPadding),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

        self.cse_GAP = nn.AvgPool2d(kernel_size=AvgPoolKernelSize, stride=1, padding=AvgPoolPadding)
        self.cse_conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3, padding=1)
        self.cse_relu = nn.ReLU(inplace=True)
        self.cse_conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=3, padding=1)
        self.cse_sigm = nn.Sigmoid()

        # self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())
        self.sse_conv = nn.Conv2d(in_channels, 1, 1)
        self.sse_sigm = nn.Sigmoid()

    def forward(self, x):
        cse_out = self.cse_GAP(x);          print(f'out wscse_cse_WAP: \t {cse_out.shape}')
        cse_out = self.cse_conv1(cse_out);  print(f'out wscse_cse_conv1: \t {cse_out.shape}')
        cse_out = self.cse_relu(cse_out);   print(f'out wscse_cse_relu: \t {cse_out.shape}')
        cse_out = self.cse_conv2(cse_out);  print(f'out wscse_cse_conv2: \t {cse_out.shape}')
        cse_out = self.cse_sigm(cse_out);   print(f'out wscse_cse_sigm: \t {cse_out.shape}')

        sse_out = self.sse_conv(x);         print(f'out wscse_sse_conv: \t {sse_out.shape}')
        sse_out = self.sse_sigm(sse_out);   print(f'out wscse_sse_sigm: \t {sse_out.shape}')

        return x * cse_out + x * sse_out

if __name__ == '__main__':
    x1 = torch.randn((2, 128, 256, 256), device='cuda')
    x2 = torch.randn((2, 128, 256, 256), device='cuda')
    scse = SCSEModule(x1.shape[1], reduction=16).cuda()
    wscse = NouveauAttention(x2.shape[1], reduction=16).cuda()

    y1 = scse(x1)
    print(f'y1.shape: {y1.shape}')
    print('\n', '#' * 35, '\n')

    y2 = wscse(x2)
    print(f'y2.shape: {y2.shape}')