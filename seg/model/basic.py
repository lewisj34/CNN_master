import torch
import torch.nn as nn

class BASIC(nn.Module):
    def __init__(self):
        super(BASIC, self).__init__()
        print(f'General CNN initialized.')
        self.conv = nn.Conv2d(3, 1, 3, 1, 1, 1, bias=False)
        # self.relu = nn.ReLU(True)
    def forward(self, x):
        x = self.conv(x)
        # x = self.relu(x[:, 0, :, :])
        return x
    
if __name__ == '__main__':
    x = torch.randn((10, 3, 256, 256), device='cuda')
    model = BASIC().cuda()
    for i in range(1000):
        out = model(x)
        print(f'out.shape:{out.shape}')
