import torch.nn as nn
from prettytable import PrettyTable

def count_parameters(model: nn.Module):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    # print(table)
    print(f"Total Trainable Params of {model._get_name()}: {total_params/1000000}M")
    return total_params

if __name__ == '__main__':
    from seg.model.CNN.CNN_plus import UNet_plain
    unet = UNet_plain(3, 1, 16)
    count_parameters(unet)

    from z import EffNet_B3, EffNet_B4
    effnetb3 = EffNet_B3(
        encoder_channels = (3, 40, 32, 48, 136, 384),
        decoder_channels = (256, 128, 64, 32, 16),
        num_classes=1,
    )
    effnetb4 = EffNet_B4(
        encoder_channels = (3, 48, 32, 56, 160, 448),
        decoder_channels = (256, 128, 64, 32, 16),
        num_classes=1,
    )
    count_parameters(effnetb3)
    count_parameters(effnetb4)
