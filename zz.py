from seg.model.zed.parts import *
from torchsummary import summary 

class NouveauAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, AvgPoolKernelSize=31, AvgPoolPadding=15):
        super().__init__()
        self.cSE = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.AvgPool2d(kernel_size=AvgPoolKernelSize, stride=1, padding=AvgPoolPadding),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

def main():
    attention_mod = NouveauAttention(
        in_channels = 512, 
        reduction = 8,
        AvgPoolKernelSize=11,
        AvgPoolPadding=5,
    )
    summary(attention_mod, (512, 128, 128), 1, 'cpu')

    x = torch.randn(1, 512, 128, 128)
    out = attention_mod(x)
    print(f'out.shape: {out.shape}')


if __name__ == '__main__':
    main()


