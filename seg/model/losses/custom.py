import torch
import torch.nn as nn 
import torch.nn.functional as F

"""
Alright, goal is to feed in a list of segmentation maps for forward. 
Perform a loss for each, while using a specific weighting function for each. 
Right now the weights for each segmentation map will be the same. 
"""

class MultiScaleIoU(nn.Module):
    def __init__(
        self, 
        # nonlin=None,
        num_seg_maps=7, 
    ):
        super(MultiScaleIoU, self).__init__()

        print(f'MultiscaleIoU initialized')
        # nonlins = ['sigmoid', 'relu', None]
        # assert nonlin in nonlins, f'nonlinearity choices: {nonlins}' 
        # self.nonlin = nonlin

        self.num_seg_maps = num_seg_maps
        self.loss_mat = torch.ones((self.num_seg_maps), device='cuda')
        self.weights_ = torch.ones((self.num_seg_maps), device='cuda')
        # self.weights = nn.Parameter(data=self.weights_, requires_grad=True)
        # self.weights = self.weights_

    def forward(
        self, 
        targets: torch.Tensor, 
        input_full: torch.Tensor,
        input_full_cnn: torch.Tensor,
        input_full_trans: torch.Tensor,
        input_1_2: torch.Tensor=None, 
        input_1_4: torch.Tensor=None, 
        input_1_8: torch.Tensor=None, 
        input_1_16: torch.Tensor=None, 
    ):
        # if self.nonlin is not None:
        #     if self.nonlin == 'sigmoid':
        #         inputs = F.sigmoid(inputs)       
        #     elif self.nonlin == 'relu':
        #         inputs = F.relu(inputs)     
        loss_1_1 = self.full_scale_loss(input_full, targets)
        loss_1_1_cnn = self.full_scale_loss(input_full_cnn, targets)
        loss_1_2_trans = self.full_scale_loss(input_full_trans, targets)
        loss_1_2 = self.half_scale_loss(input_1_2, targets)
        loss_1_4 = self.quarter_scale_loss(input_1_4, targets)
        loss_1_8 = self.eigth_scale_loss(input_1_8, targets)
        loss_1_16 = self.sixteenth_scale_loss(input_1_16, targets)

        self.loss_mat[0] = loss_1_1
        self.loss_mat[1] = loss_1_1_cnn
        self.loss_mat[2] = loss_1_2_trans
        self.loss_mat[3] = loss_1_2
        self.loss_mat[4] = loss_1_4
        self.loss_mat[5] = loss_1_8
        self.loss_mat[6] = loss_1_16

        weighted_avg_loss = self.loss_mat.mean()
        
        return 1 - loss_1_1

    def full_scale_loss(self, pred: torch.Tensor, mask: torch.Tensor):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()

    def half_scale_loss(self, pred: torch.Tensor, mask: torch.Tensor):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()

    def quarter_scale_loss(self, pred: torch.Tensor, mask: torch.Tensor):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()

    def eigth_scale_loss(self, pred: torch.Tensor, mask: torch.Tensor):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()

    def sixteenth_scale_loss(self, pred: torch.Tensor, mask: torch.Tensor):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()

if __name__ == '__main__':
    output = torch.rand(10, 1, 256, 256)
    # ground_truth = torch.rand(10, 1, 256, 256)

    # loss_fn = DiceLoss()

    # loss_val_new = loss_fn(output, ground_truth, smooth=.001)


    # # old method 
    # from seg.utils.iou_dice import mean_dice_score
    # loss_list = torch.zeros(output.shape[0])
    
    # for i in range(output.shape[0]):
    #     print(ground_truth[i, :, :, :].squeeze(0).shape, output[i, :, :, :].squeeze(0).shape)
    #     loss_list[i] = (mean_dice_score(ground_truth[i, :, :, :].squeeze(0).data.numpy(), output[i, :, :, :].squeeze(0).data.numpy()))
    # print(f'old loss val: {loss_list.mean()}')
    # print(f'new loss val: {loss_val_new}')