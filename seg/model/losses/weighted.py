'''
Modified slightly and taken from:
https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Loss-Function-Reference-for-Keras-&-PyTorch 
'''

import torch
import torch.nn as nn 
import torch.nn.functional as F

ALPHA = 0.5
BETA = 0.5
GAMMA = 1

class Weighted(nn.Module):
    def __init__(
        self,
    ):
        super(Weighted, self).__init__()
        print(f'Loss choice: Weighted.')


    def forward(self, inputs, targets, smooth=0.001):
        '''
        Calculates the loss of the network by weighting BCE and IoU
        Args:
            @seg_map (N = Batch Size, C = 1, H = Input Height, W = Input Width) 
            output segmentation map from network
            @ground_truth (N = Batch Size, C = 1, H = Input Height, W = Input Width) 
            mask or ground truth map with pixel by pixel labelling of class 
        '''
        # weight to output pixels that focuses on boundary pixels 
        weit = 1 + 5*torch.abs(
            F.avg_pool2d(
                targets, kernel_size=31, stride=1, padding=15) - targets) 
                # output dim of this is same as seg_map and ground_truth (NCHW)

        # print(weit)
        wbce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') # reduction ensures the function will return a loss value for each element 
        # output will be of dimension: torch.Size([16, 1, 192, 256])
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)) 
        # output will be of dim: torch.Size([16, 1])

        pred = torch.sigmoid(inputs)
        inter = ((pred * targets)*weit).sum(dim=(2, 3)) # out_dim = [16, 1]
        union = ((pred + targets)*weit).sum(dim=(2, 3)) # out dim = [16, 1]
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()