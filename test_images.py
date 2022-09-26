import os 
import torch
import click
import yaml
import time
import logging 
import numpy as np 
import socket
import random
import progressbar 

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
    final_cfg = yaml.load(open(Path(results_dir) / "final_config.yml", "r"),
        Loader=yaml.FullLoader)
    
    model_name = final_cfg['model_name']
    cnn_model_cfg = final_cfg['cnn_model_cfg']
    trans_model_cfg = final_cfg['trans_model_cfg']
    
    image_size = (final_cfg['final_image_height'], final_cfg['final_image_width'])

    if model_name == 'NewFusionNetwork':
        model = NewFusionNetwork(
            cnn_model_cfg,
            trans_model_cfg,
            cnn_pretrained=False,
            with_fusion=True,
        ).cuda()
    elif model_name == 'new_fusion_zed':
        from seg.model.Fusion.NewFusionNetwork import NewZedFusionNetwork
        model = NewZedFusionNetwork(
            cnn_model_cfg,
            trans_model_cfg,
            with_fusion=True,
        ).cuda()
    elif model_name == 'NewZedFusionNetworkDWSepWithCCMinDWModule':
        from seg.model.Fusion.NewFusionNetwork import NewZedFusionNetworkDWSepWithCCMinDWModule
        """
        BEST results so far 
        """
        model = NewZedFusionNetworkDWSepWithCCMinDWModule(
            cnn_model_cfg,
            trans_model_cfg
        ).cuda()
    elif model_name == 'NewZedFusionNetworkDWSepWithCCMinDWModuleInEveryUpDownModule':
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

    save_path = results_dir + '/tests/' + dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    print(f'os.save_path: {save_path}')

    # dataset stuff 
    if dataset != 'master':
        test_loader = get_tDataset(
            image_root = save_dir + "/data_train.npy",
            gt_root = save_dir + "/mask_train.npy",
            normalize_gt = False,
            batch_size = 1,
            normalization = "vit",
            num_workers = 4, 
            pin_memory=True,
        )
        with progressbar.ProgressBar(max_value=len(test_loader)) as bar:
            for i, image_gts in enumerate(test_loader):
                time.sleep(0.1)
                bar.update(i)
                
                images, gts = image_gts
                images = images.cuda()
                gts = gts.cuda()

                with torch.no_grad():
                    output = model(images)

                # output = model(images)
                output = output.sigmoid().data.cpu().numpy().squeeze()
                
                name = dataset + '_' + str(i)
                output_name = name + '_output.jpg'
                image_name = name + '_image.jpg'
                gt_name = name + '_gt.jpg'

                images = images.cpu().numpy().squeeze()
                gts = gts.cpu().numpy().squeeze().squeeze()
                images = np.transpose(images, (1,2,0))
                imwrite(save_path + output_name, output)
                imwrite(save_path + image_name, images)
                imwrite(save_path + gt_name, gts)



    else:
        # NOTE: ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir']
        test_loader = get_tDatasets_master(
            save_dir=save_dir,
            normalize_gt=False,
            batch_size=1, 
            normalization="vit", 
            num_workers=4,
            pin_memory=True,
        )
        tests = ['Kvasir', 'CVC_ClinicDB', 'CVC_ColonDB', 'CVC_300', 'ETIS']
        for i in range(len(tests)):
            print(f'test: {tests[i]}')

            for j, image_gts in enumerate(test_loader[i]):
                images, gts = image_gts
                images = images.cuda()
                gts = gts.cuda()

                with torch.no_grad():
                    output = model(images)

                output = model(images)
                output = output.sigmoid().data.cpu().numpy().squeeze()
                
                name = tests[i] + '_' + str(j)
                output_name = name + '_output.jpg'
                image_name = name + '_image.jpg'
                gt_name = name + '_gt.jpg'

                images = images.cpu().numpy().squeeze()
                gts = gts.cpu().numpy().squeeze().squeeze()
                # if j == 3:
                #     ans = input(f'Proceed with the rest of the dataset? [y/n]... ')
                #     if ans == 'y':
                #         break
                #     else:
                #         exit(1)
                print(f'output.shape: {output.shape}')
                print(f'images.shape: {images.shape}'); 
                images = np.transpose(images, (1,2,0))
                print(f'images.shape: {images.shape}')
                print(f'gts.shape: {gts.shape}')
                imwrite(save_path + output_name, output)
                imwrite(save_path + image_name, images)
                imwrite(save_path + gt_name, gts)


    


if __name__ == '__main__':
    main()
