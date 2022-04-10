import torch 
import torch.nn as nn 
from einops import rearrange

from .blocks import Block
from .utils import init_weights

class DecoderNew(nn.Module):
    def __init__(
        self, 
        num_classes, 
        patch_size, 
        d_model,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size 
        self.d_model = d_model

        self.head1 = nn.Linear(self.d_model, 256)
        # self.head2 = nn.Linear(256, 256 * 256)
        # self.head3 = nn.Linear(self.d_model // 4, self.num_classes)
        # self.conv1 = nn.ConvTranspose1d(
        #     in_channels=self.d_model // 8,
        #     out_channels=self.d_model // 4,
        #     kernel_size=1,
        # )
        # # self.conv2 = nn.Conv1d(
        # #     in_channels = self.d_model // 4,
        # #     out_channels = 256 ** 2,
        # #     kernel_size = 256,
        # # )
        # self.conv2 = nn.ConvTranspose1d(
        #     in_channels=self.d_model // 4,
        #     out_channels=self.d_model // 2,
        #     kernel_size=3,
        # )
        # self.conv3 = nn.ConvTranspose1d(
        #     in_channels=self.d_model // 2,
        #     out_channels=self.d_model,
        #     kernel_size=3,
        # )


        # self.head_out = nn.Linear(self.d_model // 2, self.num_classes)



        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.head1(x); 
        x = torch.unsqueeze(x, 1); 
        # x = self.head2(x); print(f'x.shape: {x.shape}')
        # x = self.head2(x); print(f'x.shape: {x.shape}')
        # x = self.head3(x); print(f'x.shape: {x.shape}')
        # x = rearrange(x, 'b (h w) c -> b c h w', h = 256); print(f'x.shape: {x.shape}')

        # x = self.head1(x); print(f'x.shape: {x.shape}')
        # x = torch.transpose(x, 1, 2)
        # x = self.conv1(x); print(f'x.shape: {x.shape}')
        # x = self.conv2(x); print(f'x.shape: {x.shape}')
        # x = torch.transpose(x, 2, 1); print(f'x.shape: {x.shape}')
        # x = self.conv3(x); print(f'x.shape: {x.shape}')

        # x = self.head_out(x); print(f'x.shape: {x.shape}') 
        # x = rearrange(x, "b (h w) c -> b c h w", h=GS)
        return x

class DecoderNewFull(nn.Module):
    def __init__(
        self, 
        num_classes, 
        patch_size, 
        d_model,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size 
        self.d_model = d_model

        self.head1 = nn.Linear(self.d_model, self.d_model // 2)
        self.head2 = nn.Linear(self.d_model // 2, self.d_model // 4)
        self.head3 = nn.Linear(self.d_model // 4, self.d_model // 8)
        self.head_out = nn.Linear(self.d_model // 8, self.num_classes)

        # self.head = nn.Linear(self.d_model, num_classes)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head1(x)
        x = self.head2(x)
        x = self.head3(x)
        x = self.head_out(x) 
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)
        return x



if __name__ == '__main__':
    # standard givens or model relationships     
    im_size = (256, 256)
    patch_size = 16
    d_model = 768 

    num_patches = im_size[0] // patch_size * im_size[1] // patch_size 

    # test new deocder out 
    x = torch.rand(2, num_patches, d_model)
    print(f'Test input shape to new decoder : {x.shape}')

    model = DecoderNew(
        num_classes = 1, 
        patch_size = patch_size, 
        d_model = d_model,
    ).cuda()

    model.forward(x = x.cuda(), im_size = im_size)

