'''
Designed to optimize segmentation on imbalanced medical datasets by utilizing 
constants that can adjust how harshly different types of error are penalized in 
the loss function. 

Specifically we can see that the loss is formualted as:
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  

Therefore we can penalize FP or FN depending on the params, alpha and beta. 

Modified slightly and taken from:
https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Loss-Function-Reference-for-Keras-&-PyTorch 
'''

import torch
import torch.nn as nn 
import torch.nn.functional as F

ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(
        self,
        nonlin=None,
    ):
        super(TverskyLoss, self).__init__()

        nonlins = ['sigmoid', 'relu', None]
        assert nonlin in nonlins, f'nonlinearity choices: {nonlins}' 
        self.nonlin = nonlin

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        if self.nonlin is not None:
            if self.nonlin == 'sigmoid':
                inputs = F.sigmoid(inputs)       
            elif self.nonlin == 'relu':
                inputs = F.relu(inputs)    
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky