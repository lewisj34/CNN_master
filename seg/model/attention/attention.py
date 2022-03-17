"""

Just thoughts I've had through reading: 
    <Attention-Gated Networks for Improving Ultrasound Scan Plane Detection>

"""

import torch 
import torch.nn as nn 

class AttentionDown(nn.Module):
    """
    Merge output segmentation map with the intermeidate activation maps 
    throughout the CNN.

    This module downsamples/maxpools the segmentation output map to the same 
    size as the attention map, then converts the attention coefficients. 
    """
    def __init__(
        self, 
        in_chan_act_map,
        att_stage,
    ):
        super(AttentionDown, self).__init__()

        self.att_stage = att_stage
        self.in_chan_act_map = in_chan_act_map

        # generate layer to get seg map to same size as activation map
        if self.att_stage == '1_2':
            self.down_seg = nn.MaxPool2d(kernel_size=2)
        elif self.att_stage == '1_4':
            self.down_seg = nn.MaxPool2d(kernel_size=4)
        elif self.att_stage == '1_8':
            self.down_seg = nn.MaxPool2d(kernel_size=8)
        elif self.att_stage == '1_16':
            self.down_seg = nn.MaxPool2d(kernel_size=16)
        elif self.att_stage == '1_32':
            self.down_seg = nn.MaxPool2d(kernel_size=32)
        else:
            stages = ['1_2', '1_4', '1_8', '1_16', '1_32']
            raise ValueError(f'att_stage: {att_stage} invalid. Choices: {stages}')

        # layer to get activation map to same number as seg map output channels
        self.att_conv = nn.Conv2d(
            in_channels = self.in_chan_act_map,
            out_channels = 1, 
            kernel_size = 1,
        )

        # compatability coefficient layers 
        


    def forward(
        self,
        act_map,
        seg_map,
    ):
        # downsample the segmentation map 
        seg_map_down = self.down_seg(seg_map)

        
