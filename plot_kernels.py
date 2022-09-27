import torch 
import torchvision.models as models
import matplotlib.pyplot as plt 

def plot_kernels(tensor, num_cols=6):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

import os 
import torch
import click
import yaml
import time
import logging 
import numpy as np 
import socket

from pathlib import Path
from imageio import imwrite
from seg.model.PraNet.lib.PraNet_Res2Net import PraNet
from seg.model.SwinUNet.vision_transformer import SwinUnet
from seg.model.losses.IoU_BCE_MultiScale import MultiScaleIoUBCELoss
from seg.model.losses.custom import MultiScaleIoU
from seg.model.transformer.create_model import create_transformer
from seg.model.transformer.create_modelV2 import create_transformerV2
from seg.utils.check_parameters import count_parameters
from seg.utils.data.generate_npy import split_and_convert_to_npyV2
from seg.utils.err_codes import crop_err1, crop_err2, crop_err3,resize_err1, resize_err2, resize_err3
from seg.model.CNN.CNN_backboned import CNN_BRANCH_WITH_BACKBONE
from seg.model.CNN.CNN_plus import UNet_plain
from seg.model.Fusion.FusionNetwork import OldFusionNetwork, SimplestFusionNetwork, NewFusionNetwork
from seg.model.losses.focal_tversky import FocalTverskyLoss
from seg.model.losses.iou_loss import IoULoss
from seg.model.losses.weighted import Weighted
from seg.model.losses.dice_loss import DiceLoss
from seg.model.losses.dicebce_loss import DiceBCELoss
from seg.model.losses.focal_loss import FocalLoss
from seg.model.losses.tversky import TverskyLoss
from seg.utils.inferenceV2 import InferenceModule, SegmentationMetrics
from seg.utils.t_dataset import get_tDataset, get_tDatasets_master

from seg.utils.visualize import visualizeModelOutputfromDataLoader, plot_test_valid_loss
from seg.utils.dataset import get_TestDatasetV2, get_dataset, getTestDatasetForVisualization
from seg.utils.sched import WarmupPoly
from engineV2 import train_one_epochV2

ALLOWABLE_DATASETS = ['kvasir', 'CVC_ClinicDB', 'ETIS', 'CVC_ColonDB', 'master', 'master_merged']
ALLOWABLE_MODELS = ["OldFusionNetwork", "SimplestFusionNetwork", "UNet_plain", \
    "UNet_backboned", "just_trans", "swinunet"]
ALLOWABLE_CNN_MODELS = ['unet']
ALLOWABLE_CNN_BACKBONES = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 
    'resnet152', 'vgg16', 'vgg19', 'densenet121', 'densenet161', 'densenet169', 
    'densenet201', 'unet_encoder', None]
BEST_LOSS_OPTIONS = ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir', 'ALL']

# try typing this in upon restarting 
#  python train.py --num_epochs 20 --resume '/home/john/Documents/Dev_Linux/segmentation/trans_isolated/seg/current_checkpoints/Transformer/Transformer-8.pth'

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

@click.command(help='')
@click.option('--results_dir', type=str, default='results/DataParallel/DataParallel_11')
@click.option('--checkpoint_pth', type=str, default='results/DataParallel/DataParallel_11/current_checkpoints/DataParallel-218.pth')
@click.option('--dataset', type=str, default='master')
@click.option('--save_dir', type = str, default='seg/data/master')
def main(
    dataset,
    results_dir,
    checkpoint_pth,
    save_dir,
):
    if socket.gethostname() == 'ce-yc-dlearn6.eng.umanitoba.ca' or socket.gethostname() == 'ce-yc-dlearn5.eng.umanitoba.ca':
        dataset_file_location = '/home/lewisj34_local/Dev/Datasets/master_polyp'
        print(f'Manually adjusting dataset_file_location to: {dataset_file_location}')
    
    final_cfg = yaml.load(open(Path(results_dir) / "final_config.yml", "r"),
        Loader=yaml.FullLoader)
    
    model_name = final_cfg['model_name']
    cnn_model_cfg = final_cfg['cnn_model_cfg']
    trans_model_cfg = final_cfg['trans_model_cfg']
    
    image_size = (final_cfg['final_image_height'], final_cfg['final_image_width'])

    if model_name == 'NewZedFusionNetworkDWSepWithCCMinDWModuleInEveryUpDownModule':
            from seg.model.Fusion.NewFusionNetwork import NewZedFusionNetworkDWSepWithCCMinDWModuleInEveryUpDownModule
            model = NewZedFusionNetworkDWSepWithCCMinDWModuleInEveryUpDownModule(
                cnn_model_cfg,
                trans_model_cfg
            ).cuda()
    else:
        raise ValueError(f'invalid model: {model_name}')

    num_gpu = torch.cuda.device_count()
    print(f'Number of GPUs: {num_gpu}')
    for i in range(num_gpu):
        print(f'Device name: {torch.cuda.get_device_name(i)}')
    if num_gpu > 1:
        model = torch.nn.DataParallel(model)

    print(f'Resuming at path: {os.path.basename(checkpoint_pth)}')
    checkpoint = torch.load(checkpoint_pth)
    model.load_state_dict(checkpoint['model_state_dict'])

    # model.load_state_dict(torch.load(checkpoint_pth))
    model.cuda()
    model.eval()

    save_path = results_dir + '/tests/'
    os.makedirs(save_path, exist_ok=True)
    print(f'os.save_path: {save_path}')

    model = model.module
    for name, module in model.named_children():
        if name == 'cnn_branch':
            print(f'cnn_branch')
            for name_cnn, module_cnn in module.named_children():
                print(f'\t layer: {name_cnn}')
                for name_cnn_sub_layer, module_cnn_sub_layer in module_cnn.named_children():
                    print(f'\t\t sub_layer: {name_cnn_sub_layer}')
                    for name_cnn_sub_layer_i, module_cnn_sub_layer_i in module_cnn_sub_layer.named_children():
                        print(f'\t\t\t sub_layer_i: {name_cnn_sub_layer_i}')
                        # if name_cnn == 'inc':
                        #     layer_of_interest = module_cnn_sub_layer[0]
                        #     layer = layer_of_interest.double()
                        #     tensor = layer.weight.data.cpu().numpy()
                        #     plot_kernels(tensor)
        if name == 'trans_branch':
            print(f'trans_branch')
            for name_trans, module_trans in module.named_children():
                print(f'\t layer: {name_trans}')
                for name_trans_sub_layer, module_trans_sub_layer in module_trans.named_children():
                    print(f'\t\t sub_layer: {name_trans_sub_layer}')
                    for name_trans_sub_layer_i, module_trans_sub_layer_i in module_trans_sub_layer.named_children():
                        print(f'\t\t\t sub_layer_i: {name_trans_sub_layer_i}')
                        if name_trans == 'encoder' and name_trans_sub_layer_i == "proj":
                            layer_of_interest = module_trans_sub_layer_i
                            layer = layer_of_interest.double()
                            tensor = layer.weight.data.cpu().numpy()
                            plot_kernels(np.transpose(tensor, (0, 2, 3, 1)))
                            plot_kernels(tensor[0:16, :, : , :])
        print(f'module: {name}')


    # mm = model.double()
    # filters = mm.modules
    
    # body_model = [i for i in mm.children()][0]

    # layer1 = body_model[0]
    # tensor = layer1.weight.data.numpy()
    # plot_kernels(tensor)

if __name__ == '__main__':
    main()