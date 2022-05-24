# script.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import types
import argparse
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.drop = nn.Dropout(p=0.5)
    def forward(self, x):
        print('fc1.weight {}'.format(self.fc1.weight))
        x = self.fc1(x)
        x = self.drop(x)
        print('x {}'.format(x))
        return x
def main():
    parser = argparse.ArgumentParser(description='fdsa')
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    model = MyModel().to(args.gpu)
    model = DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        output_device=args.local_rank,
    )
    for i in range(2):
        model.zero_grad()
        x = torch.randn(1, 2, device=args.gpu)
        out = model(x)
        print('iter {}, out {}'.format(i, out))
        out.mean().backward()

if __name__ == "__main__":
    main()