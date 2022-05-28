import torch
import torch.nn as nn
import torch.nn.functional as F

from .DW_sep import SeparableConv2D

class BasicConv2d_sep(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d_sep, self).__init__()
        self.conv = SeparableConv2D(
            in_planes, 
            out_planes, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding, 
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RFB_separable(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_separable, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d_sep(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d_sep(in_channel, out_channel, 1),
            BasicConv2d_sep(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d_sep(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d_sep(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d_sep(in_channel, out_channel, 1),
            BasicConv2d_sep(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d_sep(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d_sep(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d_sep(in_channel, out_channel, 1),
            BasicConv2d_sep(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d_sep(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d_sep(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d_sep(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d_sep(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

