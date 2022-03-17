'''
Modified slightly and taken from:
https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Loss-Function-Reference-for-Keras-&-PyTorch 
'''

import torch
import torch.nn as nn 
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(
        self, 
        nonlin=None
    ):
        super(DiceBCELoss, self).__init__()

        nonlins = ['sigmoid', 'relu', None]
        assert nonlin in nonlins, f'nonlinearity choices: {nonlins}' 
        self.nonlin = nonlin

    def forward(self, inputs, targets, smooth=1):
        
        if self.nonlin is not None:
            if self.nonlin == 'sigmoid':
                inputs = F.sigmoid(inputs)       
            elif self.nonlin == 'relu':
                inputs = F.relu(inputs)      
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs.float(), targets.float(), reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE