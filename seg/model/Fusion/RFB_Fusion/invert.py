import torch
import torch.nn as nn
import torch.nn.functional as F 

class NegativeInvert(nn.Module):
    def __init__(
        self, 
        expand_channels,
        scale_factor,
    ):
        super(NegativeInvert, self).__init__()
        """
        Calls the negative inverse sigmoid function that appears in PraNet
        Takes in a smaller representation of the desired target and multiples it 
        with a same size feature map from somewhere else in the network.  
            1. Scales it up to output size. 
            2. Expands it to the number of channels the output has. 
            3. Multiples the two features.
            @expand_channels: number of channels it needs to be expanded to. 
            @scale_factor: the size to scale it up to. 
        """

        self.scale_factor = scale_factor
        self.chans = expand_channels

    def forward(self, x_small, x_in):
        inv = F.interpolate(
            x_small, 
            scale_factor=self.scale_factor, 
            mode='bilinear'
        )

        x = -1 * (torch.sigmoid(inv)) + 1
        x = x.expand(-1, self.chans, -1, -1).mul(x_in)
        return x 

if __name__ == '__main__':
    x_small = torch.randn((2, 1, 11, 11))
    x_in = torch.randn((2, 1024, 22, 22))

    model = NegativeInvert(
        expand_channels=1024, 
        scale_factor=2
    )

    x_out = model(x_small, x_in)
    print(f'COMPLETE.')