import torch 
import numpy as np 

from seg.utils.check_parameters import count_parameters

if __name__ == '__main__':
    from seg.model.CNN.CNN import CNN_BRANCH
    from seg.model.zed.zedNet import zedNet
    count_parameters(
        CNN_BRANCH(
            n_channels=3,
            n_classes=1, 
            patch_size=16,
        )
    )

    count_parameters(
        zedNet(
            n_channels=3,
            n_classes=1,
            patch_size=16
        )
    )

