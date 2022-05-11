import torch
import torch.nn as nn 
import numpy as np 



class BNRdilatedconv3x3x3(nn.Module):
  def __init__(
    self,
    in_planes, 
    out_planes, 
    stride=1, 
    groups=1, 
  ):
    """
    Does a dilated convolution 3x3x3. 
      1st conv3x3 is dilation=1
      2nd conv3x3 is dilation=2
      3rd conv3x3 is dilation=3
    Visual found here: 
      https://www.researchgate.net/figure/3-3-convolution-kernels-with-different-dilation-rate-as-1-2-and-3_fig9_323444534
    """
    super(BNRdilatedconv3x3x3, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, groups=groups, bias=False, dilation=1)
    self.bn1 = nn.BatchNorm2d(out_planes)
    self.relu1 = nn.ReLU()

    self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride,
                  padding=2, groups=groups, bias=False, dilation=2)
    self.bn2 = nn.BatchNorm2d(out_planes)
    self.relu2 = nn.ReLU()

    self.conv3 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride,
                  padding=3, groups=groups, bias=False, dilation=3)
    self.bn3 = nn.BatchNorm2d(out_planes)
    self.relu3 = nn.ReLU()

  def forward(self, x):
    x = self.conv1(x);  print(f'x: {x.shape}')
    x = self.bn1(x);  print(f'x: {x.shape}')
    x = self.relu1(x);  print(f'x: {x.shape}')
    x = self.conv2(x);  print(f'x: {x.shape}')
    x = self.bn2(x);  print(f'x: {x.shape}')
    x = self.relu2(x);  print(f'x: {x.shape}')
    x = self.conv3(x);  print(f'x: {x.shape}')
    x = self.bn3(x);  print(f'x: {x.shape}')
    x = self.relu3(x);  print(f'x: {x.shape}')
    return x 

class DilatedCNN(nn.Module):
  def __init__(self):
    super(DilatedCNN,self).__init__()
    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 9, stride = 1, padding = 0, dilation=2)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size = 3, stride = 1, padding= 0, dilation = 2)
    self.relu2 = nn.ReLU()
    
    self.fclayers = nn.Sequential(
      nn.Linear(2304,120),
      nn.ReLU(),
      nn.Linear(120,84),
      nn.ReLU(),
      nn.Linear(84,10)
    )
  def forward(self, x: torch.Tensor):
    x = self.conv1(x); print(f'x: {x.shape})')
    x = self.relu1(x); print(f'x: {x.shape})')
    x = self.conv2(x); print(f'x: {x.shape})')
    x = self.relu2(x); print(f'x: {x.shape})')
    x = x.view(-1,2304)
    x = self.fclayers(x)
    return x

if __name__ == '__main__':
    x = torch.randn((10, 3, 32, 32), device='cuda')
    model = BNRdilatedconv3x3x3(3, 64).cuda()
    out = model(x)