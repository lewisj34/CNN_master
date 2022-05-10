import torch
import torch.nn as nn 
import numpy as np 


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
    model = DilatedCNN().cuda()
    out = model(x)