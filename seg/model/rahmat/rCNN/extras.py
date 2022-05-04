import torch 
import torch.nn as nn 


class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes, output_stride=16):
        super(ASPP, self).__init__()
 
        if output_stride == 16:
            dilations = [1, 2, 3, 4]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
 
        self.aspp1 = _ASPPModule(inplanes, 64, 1, padding=1, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 64, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 64, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 64, 3, padding=dilations[3], dilation=dilations[3])
 
        self.conv1 = nn.Conv2d(256, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, outplanes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
 
    def forward(self, x):
        x1 = self.aspp1(x) #64,62,30
        x2 = self.aspp2(x) #64,64,32
        x3 = self.aspp3(x) #64,64,32
        x4 = self.aspp4(x) #64,64,32
        x = torch.cat((x1, x2, x3, x4), dim=1)
 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x  = self.conv2(x) 
        x = self.bn2(x)
        x = self.relu2(x)
        return self.dropout(x)


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.pointwise = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=padding, dilation=dilation,groups=planes, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.pointwise(x)
        return x
 

if __name__ == '__main__':
    model = ASPP(
        inplanes=32,
        outplanes=10

    )

    from torchsummary import summary 
    summary(
        model = model.cuda(),
        input_size = (32, 256, 256),
        batch_size = 10, 

    )