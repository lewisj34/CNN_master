import torch
import torch.nn as nn

from seg.model.general.DW_sep import count_parameters

class InputProjectionA(nn.Module):
    # '''
    # This class projects the input image to the same spatial dimensions as the feature map.
    # For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    # this class will generate an output of 56x56x3
    # '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(2, stride=2, padding=0))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input

class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        # self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        # self.conv1 = nn.Conv2d(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
        # self.act = nn.ReLU()
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output

class EntranceFlowIDSNet(nn.Module):
    def __init__(self):
        '''
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        self.init_block = nn.Sequential(  
            CBR(nIn=3, nOut=24, kSize=3, stride=1),
            CBR(nIn=24, nOut=24, kSize=3, stride=1),
            CBR(nIn=24, nOut=24, kSize=3, stride=1),
        )
        self.level1 = CBR(24, 24, 3, 2) 
        # self.level1 = nn.Sequential()
        self.sample1 = InputProjectionA(1)  
        self.sample2 = InputProjectionA(2)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        input = self.init_block(input)                                          # torch.Size([1, 24, 256, 256]) 
        output0 = self.level1(input); print(f'level1: \t {output0.shape}')      # torch.Size([1, 24, 128, 128])
        inp1 = self.sample1(input); print(f'input_proj1: \t {inp1.shape}')      # torch.Size([1, 24, 128, 128])
        inp2 = self.sample2(input); print(f'inp2: \t\t {inp2.shape}')           # torch.Size([1, 24, 64, 64])

        return inp2 


if __name__ == '__main__':
    input = torch.randn((1, 3, 256, 256), device='cpu')
    model = EntranceFlowIDSNet()
    output = model(input)

    count_parameters(model)
