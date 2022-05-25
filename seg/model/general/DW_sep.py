import torch
import torch.nn as nn


def count_parameters(module: nn.Module, trainable: bool = True) -> int:

    if trainable:
        num_parameters = sum(p.numel() for p in module.parameters()
                             if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in module.parameters())

    return num_parameters


def conv_parameters(in_channels, out_channels, kernel_size, bias) -> int:

    num_parameters = in_channels * out_channels * kernel_size[0] * kernel_size[
        1]

    if bias:
        num_parameters += out_channels

    return num_parameters


def separable_conv_parameters(in_channels, out_channels, kernel_size,
                              bias) -> int:

    num_parameters = in_channels * kernel_size[0] * kernel_size[
        1] + in_channels * out_channels

    if bias:
        num_parameters += (in_channels + out_channels)

    return num_parameters


class DepthwiseConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros') -> None:
        super().__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.depthwise_conv(x)

        return x


class PointwiseConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 device=None,
                 dtype=None) -> None:
        super().__init__()

        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(1, 1),
                                        stride=1,
                                        padding=0,
                                        dilation=1,
                                        groups=1,
                                        bias=bias,
                                        padding_mode='zeros')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pointwise_conv(x)

        return x


class SeparableConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros') -> None:
        super().__init__()

        self.depthwise_conv = DepthwiseConv2D(in_channels=in_channels,
                                              out_channels=in_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              groups=in_channels,
                                              bias=bias,
                                              padding_mode=padding_mode)

        self.pointwise_conv = PointwiseConv2D(in_channels=in_channels,
                                              out_channels=out_channels,
                                              bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x