import torch
import torch.nn as nn
import torch.nn.functional as F

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CCMSubBlock(nn.Module):
    '''
    This class defines the dilated convolution.
    '''
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d

        combine_kernel = 2 * d - 1

        self.conv = nn.Sequential(
            nn.Conv2d(nIn, nIn, kernel_size=(combine_kernel, 1), stride=stride, padding=(padding - 1, 0),
                        groups=nIn, bias=False),
            nn.BatchNorm2d(nIn),
            nn.PReLU(nIn),
            nn.Conv2d(nIn, nIn, kernel_size=(1, combine_kernel), stride=stride, padding=(0, padding - 1),
                        groups=nIn, bias=False),
            nn.BatchNorm2d(nIn),
            nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False,
                        dilation=d),
            nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False))

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class CCMModule(nn.Module):
    def __init__(self, nIn, nOut, ratio=[2,4,8],  ratio2=[1,4,7]):
        super().__init__()
        n = int(nOut // 3)
        n1 = nOut - 3 * n
        #
        print(n1,n)
        self.c1 = C(nIn, n, 3, 2)
        # SB-1 Sublock 1
        self.sb1 = CCMSubBlock(n, n + n1, 3, 1, ratio[0])
        self.sb1_r = CCMSubBlock(n + n1, n + n1, 3, 1, ratio2[0]) # Right side submodule
        # # SB-2 Sublock 2
        self.sb2 = CCMSubBlock(n, n, 3, 1, ratio[1])
        self.sb2_r = CCMSubBlock(n, n, 3, 1, ratio2[1])
        # # SB-2 Sublock 2
        self.sb3 = CCMSubBlock(n, n, 3, 1, ratio[2])
        self.sb3_r = CCMSubBlock(n, n, 3, 1, ratio2[2])

        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.bn_sub1 = BNPReLU(n + n1)
        self.bn_sub2 = BNPReLU(n)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)

        sb1 = self.bn_sub1(self.sb1(output1))
        sb1_r = self.sb1_r(sb1)

        sb2 = self.bn_sub2(self.sb2(output1))
        sb2_r = self.sb2_r(sb2)

        sb3 = self.bn_sub2(self.sb3(output1))
        sb3_r = self.sb3_r(sb3)

        combine = torch.cat([sb1_r, sb2_r, sb3_r], 1)

        output = self.bn(combine)
        output = self.act(output)
        return output

class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
        # self.act = nn.ReLU()

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output

class RCCMModule(nn.Module):
    '''
    
        
    '''

    def __init__(self, nIn, nOut, add=True, ratio=[2,4,8]):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. 
        '''
        super().__init__()
        n = int(nOut // 3)
        n1 = nOut - 3 * n
        #  a
        self.c1 = C(nIn, n, 1, 1)
        # SB-1 Sublock 1
        # self.d1 = CCMSubBlock(n, n + n1, 3, 1, ratio[0])
        self.d1 = nn.Conv2d(n, n+n1, kernel_size=3, padding=1)
        # # SB-2 Sublock 2
        self.d2 = CCMSubBlock(n, n, 3, 1, ratio[1])
        # # SB-2 Sublock 2
        self.d3 = CCMSubBlock(n, n, 3, 1, ratio[2])
        # self.d4 = Double_CDilated(n, n, 3, 1, 12)
        # self.conv =C(nOut, nOut, 1,1)

        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)

        combine = torch.cat([d1, d2, d3], 1)

        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output
