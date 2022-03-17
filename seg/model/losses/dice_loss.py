'''
Modified slightly and taken from:
https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch#Loss-Function-Reference-for-Keras-&-PyTorch 
'''

import torch
import torch.nn as nn 
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(
        self, 
        nonlin=None
    ):
        super(DiceLoss, self).__init__()

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
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

if __name__ == '__main__':
    output = torch.rand(10, 1, 256, 256)
    ground_truth = torch.rand(10, 1, 256, 256)

    loss_fn = DiceLoss()

    loss_val_new = loss_fn(output, ground_truth, smooth=.001)


    # old method 
    from seg.utils.iou_dice import mean_dice_score
    loss_list = torch.zeros(output.shape[0])
    
    for i in range(output.shape[0]):
        print(ground_truth[i, :, :, :].squeeze(0).shape, output[i, :, :, :].squeeze(0).shape)
        loss_list[i] = (mean_dice_score(ground_truth[i, :, :, :].squeeze(0).data.numpy(), output[i, :, :, :].squeeze(0).data.numpy()))
    print(f'old loss val: {loss_list.mean()}')
    print(f'new loss val: {loss_val_new}')