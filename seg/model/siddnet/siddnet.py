import torch
import torch.nn as nn

basic_0 = 24
basic_1 = 48
basic_2 = 56
basic_3 = 24


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

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


class SuperficialModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3, repeat=3):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(repeat):
            self.net.add_module(f"SuperficialModule_subblock_{i+1}", SuperficialModule_subblock(nIn))
        self.net.add_module("bnrelu", BNPReLU(nIn))

    def forward(self, x):
        out = self.net(x)
        return out


class SuperficialModule_subblock(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.conv3x3_i  = Conv(nIn // 2, nIn // 2, kSize, 1, padding=1, groups=nIn // 2,  bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                             padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                             padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                              padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                              padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br1 = self.conv3x3_i(br1)

        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br2 = self.conv3x3_i(br2)

        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input

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


class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output


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


# CCM sub-block structure
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
        if d == 1:
            self.conv =nn.Sequential(
                nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False,
                          dilation=d),
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False)
            )
        else:
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

# In the CCM module, were also implemented in the comprehensive branch first
# the height and width of the are reduced in half and the depth of channel
# is reduced by one third. The output is then input to the three sub-blocks (SB).


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



# In the CCM module, were also implemented in the comprehensive branchfirst
# the height and width of the are reduced in half and the depth of cahnnel
# is reduced by one third. The output is then input to the three sub-blocks (SB).


# RCCM  Residual concentrated comprehensive convolution modules (CCM)
# In a similar manner, the depth of channel of RCCM channel is also reduced by one third
# before feding it to the SB-1, SB-2, and SB-3.
#  concentrated comprehensive convolution modules (CCM)
#  In RCCM the input is added to the output of the three sub blocks SB1, SB2, and SB3
# The residual use the concepts of the residual network.
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


# RCCM  Residual concentrated comprehensive convolution modules (CCM)
# In a similar manner, the depth of channel of RCCM channel is also reduced by one third
# before feding it to the SB-1, SB-2, and SB-3.
#  concentrated comprehensive convolution modules (CCM)
#  In RCCM the input is added to the output of the three sub blocks SB1, SB2, and SB3
# The residual use the concepts of the residual network.



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



class IDSNetComprehensive(nn.Module):
    '''
    
    '''

    def __init__(self, classes=2, p=5, q=3, upsample2GTsize=False):
        '''
        :param p: depth multiplier
        :param q: depth multiplier
        @param upsample2GTsize: you input this (john) to get the output
            of the model to be the same size as the ground truth 
        '''
        super().__init__()

        self.upsample2GTsize = upsample2GTsize

        self.init_block = nn.Sequential(  # new initial block added 2021-11-20
            CBR(nIn=3, nOut=basic_0, kSize=3, stride=1),
            CBR(nIn=basic_0, nOut=basic_0, kSize=3, stride=2),
            CBR(nIn=basic_0, nOut=basic_0, kSize=3, stride=2)
        )
        self.level1 = CBR(basic_0, basic_0, 3, 2) #
        # self.level1 = nn.Sequential()
        self.sample1 = InputProjectionA(1)  #
        self.sample2 = InputProjectionA(2)

        self.b1 = BR(basic_0 + basic_0)
        self.level2_0 = CCMModule(basic_0 + basic_0, basic_1, ratio=[1, 2, 3])  # , ratio=[1,2,3]

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(
                RCCMModule(basic_1, basic_1, ratio=[1, 3, 4]))  # , ratio=[1,3,4]
        self.b2 = BR(basic_1 * 2 + basic_0)

        self.level3_0 = RCCMModule(basic_1 * 2 + basic_0, basic_2, add=False,
                                   ratio=[1, 3, 5])  # , ratio=[1,3,5]

        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(RCCMModule(basic_2, basic_2))
        self.b3 = BR(basic_2 * 2)

        self.upsample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
        )

        self.ComprehensiveClassifier = C(basic_2 * 2, classes, 1, 1)

        if self.upsample2GTsize:
            self.upsample_final = nn.UpsamplingBilinear2d(scale_factor = 4)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        input = self.init_block(input) # w/4
        output0 = self.level1(input)  # w/8, h/4 Channel input (Ci) = 3, channel output (Co) = 24
        inp1 = self.sample1(input)  # size reduced w/8 and h/4, Channel input = 3,  Co = 3

        inp2 = self.sample2(input)  # size reduced w/16 and h/4, Channel input = 3, Co=3
        # print("output0:", output0.shape)
        # print("inp1:", inp1.shape)
        # print("inp2:", inp2.shape)
        output0_cat = self.b1(torch.cat([output0, inp1], 1)) # concatenation (1 mean channelwise ) Ci =27, Co=27
        output1_0 = self.level2_0(output0_cat)  # down-sampled  CCM-1, Ci =27, Co=48
        # print("output1_0:", output0.shape)

        for i, layer in enumerate(self.level2):   # RCCM-1 X 5
            if i == 0:
                output1 = layer(output1_0)    #,  Ci =48, Co=48

            else:
                output1 = layer(output1) #,  Ci =48, Co=48

        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1)) #CM-1 , CCM-1, P2, Ci = 48+48+3=99, Co=99
        # print("output1_cat:", output1_cat.shape)
        output2_0 = self.level3_0(output1_cat)  # down-sampled RCCM-1 X 2,   Ci=99,  Co=56
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)    # Ci=56,  Co=56
            else:
                output2 = layer(output2)    # Ci=56,  Co=56

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))    # Ci=56 +56 =112,  Co=56 +56 =112
        output2_cat = self.upsample(output2_cat)
        classifier = self.ComprehensiveClassifier(output2_cat)   # Ci=112,  Co=2

        if self.upsample2GTsize:
            classifier = self.upsample_final(classifier)
        return classifier

    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True

    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True

class IDSNet(nn.Module):
    '''
    
    '''

    def __init__(self, classes=2, p=5, q=3, stage1_W=None):
        '''
        
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        self.encoder = IDSNetComprehensive(classes, p, q)
        if stage1_W != None:
            self.encoder.load_state_dict(torch.load(stage1_W))
            print('Encoder loaded!')
        # # load the encoder modules
        del self.encoder.ComprehensiveClassifier

        self.upsample = nn.Sequential(
            nn.Conv2d(kernel_size=(1, 1), in_channels=basic_2*2, out_channels=basic_3,bias=False),
            nn.BatchNorm2d(basic_3),
            nn.UpsamplingBilinear2d(scale_factor=4),

        )

        self.Superficial = nn.Sequential(
            SuperficialModule(basic_0),
            # Conv(3, basic_0, 1, 1, padding=0, bn_acti=False),  # 1x1 conv
            BR(basic_0),
            # nn.AvgPool2d(2, stride=2, padding=0), # downsampling by 2
            # nn.AvgPool2d(2, stride=2, padding=0) # downsampling by 2
        )
        self.classifier = nn.Sequential(
            BR(basic_3),
            nn.UpsamplingBilinear2d(scale_factor=4),
            nn.Conv2d(kernel_size=(1, 1), in_channels=basic_3, out_channels=classes, bias=False),
        )
        self.upsample2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
        )

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0_i = self.encoder.init_block(input) # w/4
        output0 = self.encoder.level1(output0_i) # w/8
        inp1 = self.encoder.sample1(output0_i) # w/8
        inp2 = self.encoder.sample2(output0_i) # w/16
        output0_cat = self.encoder.b1(torch.cat([output0, inp1], 1))

        output1_0 = self.encoder.level2_0(output0_cat)  # down-sampled


        for i, layer in enumerate(self.encoder.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.encoder.b2(torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.encoder.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.encoder.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.encoder.b3(torch.cat([output2_0, output2], 1))   # 112

        Comprehensive = self.upsample(output2_cat)  # 24
        SuperFicial =  self.Superficial(output0_i)
        classifier = self.classifier(Comprehensive) # without superficial module
        # classifier = self.classifier(Comprehensive + SuperFicial)
        return classifier

    def freeze(self):
        # To freeze the residual layers
        for param in self.parameters():
            param.require_grad = False
        for param in self.classifier.parameters():
            param.require_grad = True

    def unfreeze(self):
        # Unfreeze all layers
        for param in self.parameters():
            param.require_grad = True

def Stage1_IDSNet(**kwargs):
    print("train only IDSNetComprehensive")
    model = IDSNetComprehensive(**kwargs)
    return model

def Stage2_IDSNet(classes, p, q, stage1_W=None):
    print("train All network")
    model = IDSNet(classes, p, q, stage1_W)
    return model

# if __name__ == '__main__':
#     import time
#     import numpy as np
#     import torch
#     import os
#     # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#     # from etc.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
#     model = Stage2_IDSNet(classes=1, p=1, q=5).cuda()
#     # model = CCMSubBlock(3, 3, 3).cuda()
#     # print(model)
#     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#     params = sum([np.prod(p.size()) for p in model_parameters])
#     print("params: ", params)
#     # for name, param in model.named_parameters():
#     #     if param.requires_grad:
#     #         print (name, len(param))
#     batch = torch.FloatTensor(10, 3, 480, 640).cuda()
#     out = model(batch)
#     print(out.shape)

#     from torchsummary import summary 
#     summary(model, (3, 256, 256), 10)

if __name__ == '__main__':
    model = CCMSubBlock(
        nIn = 32, 
        nOut = 3,
        kSize=3,
        d=2
    ).cuda()

    from torchsummary import summary
    summary(
        model = model,
        input_size = (32, 128, 128),
        batch_size = 10 
    )

