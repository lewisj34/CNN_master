import torch
import torch.nn as nn 
import torch.nn.functional as F

class MultiScaleIoUBCELoss(nn.Module):
    def __init__(
        self,
        num_losses,
        epoch_unfreeze,
    ):
        super(MultiScaleIoUBCELoss, self).__init__()
        self.num_losses = num_losses
        self.epoch_unfreeze = epoch_unfreeze

        # generate weights 
        self.W_l = nn.Parameter(torch.ones(self.num_losses))
        print(f'W_l: {self.W_l}')
        print(f'W_l.shape: {self.W_l.shape}')
                

    def structure_loss(self, pred, mask):
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        return (wbce + wiou).mean()

    def forward(self, lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, gts, epoch, smooth = 0.001):
        if epoch > self.epoch_unfreeze:
            loss5 = (self.W_l[0]) * self.structure_loss(lateral_map_5, gts)
            loss4 = (self.W_l[1]) * self.structure_loss(lateral_map_4, gts)
            loss3 = (self.W_l[2]) * self.structure_loss(lateral_map_3, gts)
            loss2 = (self.W_l[3]) * self.structure_loss(lateral_map_2, gts) 
        else:
            loss5 = self.structure_loss(lateral_map_5, gts)
            loss4 = self.structure_loss(lateral_map_4, gts)
            loss3 = self.structure_loss(lateral_map_3, gts)
            loss2 = self.structure_loss(lateral_map_2, gts) 
        # print(f'self.W_l: {self.W_l}')
        
        loss = loss2 + loss3 + loss4 + loss5
        return loss

# def structure_loss(pred, mask):
#     weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)*weit).sum(dim=(2, 3))
#     union = ((pred + mask)*weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1)/(union - inter+1)
#     return (wbce + wiou).mean()

if __name__ == '__main__':
    loss_fn = MultiScaleIoUBCELoss(4)
