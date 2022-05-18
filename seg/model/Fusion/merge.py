import torch
import torch.nn as nn
import torch.nn.functional as F

class merge(nn.Module):
    def __init__(
        self,
        num_seg_maps=3,
    ):
        """
        Takes in three segmentation maps and merges them. 
        """
        super().__init__()
        self.num_seg_maps = num_seg_maps

        self.avg_pool = nn.AdaptiveAvgPool2d(1)



    
    def forward(self, seg_maps: list):
        x_out = torch.cat(seg_maps, dim=1)

        x_out = self.avg_pool(x_out)
        return x_out
        
if __name__ == '__main__':
    tensor_list = list()

    num_seg_maps = 3
    for i in range(num_seg_maps):
        tensor_list.append(torch.randn((2, 1, 256, 256), device='cuda'))

    model = merge(num_seg_maps=num_seg_maps).cuda()

    out = model(tensor_list)

    print(f'out.shape: {out.shape}')