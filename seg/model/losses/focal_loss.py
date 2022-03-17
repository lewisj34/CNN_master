'''
Good for: combatting extremely imbalanced datasets where positive cases were 
relatively rare. From the paper: "Focal Loss for Dense Object Detection". R
etrievable here: https://arxiv.org/abs/1708.02002.
Modified slightly and taken from:
https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Loss-Function-Reference-for-Keras-&-PyTorch 
'''

import torch
import torch.nn as nn 
import torch.nn.functional as F


ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(
        self, 
        nonlin=None
    ):
        super(FocalLoss, self).__init__()

        nonlins = ['sigmoid', 'relu', None]
        assert nonlin in nonlins, f'nonlinearity choices: {nonlins}' 
        self.nonlin = nonlin

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        if self.nonlin is not None:
            if self.nonlin == 'sigmoid':
                inputs = F.sigmoid(inputs)       
            elif self.nonlin == 'relu':
                inputs = F.relu(inputs)        
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss