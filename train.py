import os 
import torch
import click
import yaml
import time
import logging 

from pathlib import Path
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
@click.option('--dataset', type=str, default='master')
@click.option('--model_name', type=str, default='OldFusionNetwork') 
@click.option('--backbone', type=str, default='vit_base_patch16_384')
@click.option('--decoder', type=str, default='linear')
@click.option('--cnn_model_name', type=str, default='unet')
@click.option('--cnn_backbone', type=str, default='resnet18')
@click.option('--image_height', type=int, default=256)
@click.option('--image_width', type=int, default=256)
@click.option('--crop_height', type=int, default=None)
@click.option('--crop_width', type=int, default=None)
@click.option('--reimport_data', type=bool, default=False)
@click.option('--dropout', type=int, default=0.0)
@click.option('--drop_path_rate', type=int, default=0.1)
@click.option('--n_cls', type=int, default=1)
@click.option('--save_dir', type=str, default='seg/data')
@click.option('--num_epochs', type=int, default=1)
@click.option('--learning_rate', type=float, default=2.1052e-05) # old: 7e-5
@click.option('--batch_size', type=int, default=16)
@click.option('--model_checkpoint_name', type=str, default=None)
@click.option('--speed_test', type=bool, default=False, help='if True, runs FPS measurements of network every 5 epochs')
@click.option('--resume', type=str, default=None, help='path to checkpoint')
@click.option('--cnn_branch_checkpt', type=str, default=None, help='path to cnn branch checkpt') # /home/john/Documents/Dev_Linux/segmentation/trans_isolated/seg/current_checkpoints/CNN_BRANCH_WITH_BACKBONE/CNN_BRANCH_WITH_BACKBONE-98.pth
@click.option('--trans_branch_checkpt', type=str, default=None, help='path to transformer branch checkpt') # /home/john/Documents/Dev_Linux/segmentation/trans_isolated/seg/current_checkpoints/Transformer/Transformer-22.pth
@click.option('--lr_sched', type=str, default='warmpoly', help='step, poly, multistep, warmpoly')
@click.option('--loss_type', type=str, default='iou', help='iou, weight, lovasz')
@click.option('--best_loss_option', type=str, default='ALL')
@click.option('--dataset_file_location', type=str, default=None, help='where is the dataset corresponding to `dataset` saved')
@click.option('--num_output_trans', type=int, default=64, help='where is the dataset corresponding to `dataset` saved')
def main(
    dataset,
    model_name,
    backbone,
    decoder,
    cnn_model_name,
    cnn_backbone,
    image_height,
    image_width, 
    crop_height,
    crop_width,
    reimport_data,
    dropout,
    drop_path_rate,
    n_cls,
    save_dir,
    num_epochs,
    learning_rate,
    batch_size,
    model_checkpoint_name,
    speed_test,
    resume,
    cnn_branch_checkpt,
    trans_branch_checkpt,
    lr_sched,
    loss_type,
    best_loss_option,
    dataset_file_location,
    num_output_trans,
):
    assert dataset in ALLOWABLE_DATASETS, 'invalid dataset'
    assert cnn_model_name in ALLOWABLE_CNN_MODELS, 'invalid cnn model choice'
    assert cnn_backbone in ALLOWABLE_CNN_BACKBONES, 'invalid cnn backbone'
    # assert backbone in default_cfgs, 'invalid backbone choice'

    save_dir = save_dir + '/' + dataset 

    # import data and generate dataset 
    """
    ****************************** USAGE ****************************** 
    4 Situations we need to check and make sure model can read properly:
        1. Just resize image, no crop. 
            -> given by crop_size == None. 
        2. Resize image and then crop. 
            -> both image_size and crop_size given. 
        3. No resizing of image, just crop the image. (Needs an additional check here potentially 
            when iterating through the dir and openCVing the images, lets run
            a test on this. - shouldnt be hard) 
            -> given by image_size, image_width = 0. 
            -> crop_size must not be None 
    """
    if crop_height == 0 or crop_width == 0:
        assert crop_height == 0 and crop_width == 0, crop_err3(crop_height, crop_width)
        print(f'Crop_height or crop_width was manually input in command line as 0. Resetting thes values to None type.')
        crop_height = None
        crop_width = None 

    if crop_height is not None or crop_width is not None:
        """
        2 situations apply when crop_size is given. 
            1. Resize and then crop. 
            2. No resize, just crop. 
        """
        # following check applies to both situations
        assert crop_height is not None and crop_width is not None, crop_err1(crop_height, crop_width)
        
        # situation 1: resize and then crop
        if image_height != 0 or image_width != 0:
            print(f'Resizing and cropping.')
            assert image_height != 0 and image_width != 0, resize_err1(image_height, image_width)
            
            if image_height < crop_height or image_width < crop_width:
                raise ValueError(crop_err2(crop_height, crop_width, image_height, image_width))
            resize_size = (image_height, image_width)
            crop_size = (crop_height, crop_width)
            image_size = crop_size
        
        # situation 2: just crop (because image_height and width are given as 0) 
        else:
            print(f'Cropping but not resizing the input data.')
            assert image_height == 0 and image_width == 0, resize_err2(image_height, image_width)
            resize_size = (None, None)
            crop_size = (crop_height, crop_width)
            image_size = (crop_height, crop_width)

    else:
        print(f'Reszing but not cropping the input data.')
        assert image_height != 0 and image_width != 0, resize_err3(image_height, image_width)
        resize_size = (image_height, image_width)
        crop_size = (None, None)
        image_size = resize_size

    # end of image resize and cropping checks 
    # print results of resize and cropping values inputted to program 
    print(f'resize_size: {resize_size}')
    print(f'crop_size: {crop_size}')
    print(f'image_size: {image_size}')

    if resize_size[0] != resize_size[1] or crop_size[0] != crop_size[1] \
        or image_size[0] != image_size[1]:
        print(f'********* Warning: input image dims are not square. *********')

    # now that we have resize size and crop size, split the images into train
    # test and val sets, and then convert those images to binary for easier
    # access and less model runtime when running successive training iterations
    split_and_convert_to_npyV2(
        dataset = dataset,
        save_dir = save_dir, 
        resize_size = resize_size,
        crop_size = crop_size,
        image_size = image_size,
        reimport_data = reimport_data, 
        num_classes = n_cls,
        parent_dir = dataset_file_location,
    )

    # import model details for transformer 
    trans_cfg = yaml.load(open(Path(__file__).parent / "transformer_presets.yml", "r"), 
        Loader=yaml.FullLoader)
    trans_model_cfg = trans_cfg['model'][backbone]
    decoder_cfg = trans_cfg["decoder"][decoder]

    # images
    trans_model_cfg['image_size'] = image_size
    trans_model_cfg["dropout"] = dropout
    trans_model_cfg["drop_path_rate"] = drop_path_rate
    trans_model_cfg['backbone'] = backbone 
    trans_model_cfg['n_cls'] = n_cls
    decoder_cfg['name'] = decoder 
    trans_model_cfg['decoder'] = decoder_cfg

    patch_size = trans_model_cfg['patch_size']

    assert trans_model_cfg['patch_size'] == 16 \
        or trans_model_cfg['patch_size'] == 32, 'Patch size == {16, 32}'
    

    # set up cnn stuff 
    cnn_cfg = yaml.load(open(Path(__file__).parent / "cnn_config.yml", "r"), 
        Loader=yaml.FullLoader)
    cnn_model_cfg = cnn_cfg['model'][cnn_model_name]
    cnn_model_cfg['image_size'] = image_size
    cnn_model_cfg['patch_size'] = patch_size
    cnn_model_cfg['batch_size'] = batch_size
    cnn_model_cfg['num_classes'] = n_cls
    cnn_model_cfg['in_channels'] = 3 # this isn't anywhere in transformer but 
    cnn_model_cfg['backbone'] = cnn_backbone

    print(f'trans_model_cfg:\n {trans_model_cfg}')
    print(f'cnn_model_cfg:\n {cnn_model_cfg}')

    cnn_model_cfg['num_output_trans'] = num_output_trans
    trans_model_cfg['num_output_trans'] = num_output_trans

    if model_name == 'basic':
        from seg.model.basic import BASIC
        model = BASIC().cuda()
    elif model_name == "OldFusionNetwork":
        model = OldFusionNetwork(
            cnn_model_cfg, 
            trans_model_cfg,
            cnn_pretrained=False,
            with_fusion=True,
            with_aspp=False,
        ).cuda()
        if cnn_branch_checkpt is not None or trans_branch_checkpt is not None:
            # loading a checkpoint for a full model while having cnn checkpoint 
            # and transformer checkpoint loaded would make no sense (its either 
            # resume the whole model or one or both of the individual branches)
            assert resume is None 

            # checkpoint loading below uses same methodology as the if resume: 
            # structure does below, but we're only interested in loading the 
            # params of the model not the learning rate or optimizer settings
            # because we're using the trained params for the FUSED network to
            # see if we can get better results
            if cnn_branch_checkpt is not None:
                print(f'Loading CNN checkpoint at path: {os.path.basename(cnn_branch_checkpt)}') # IoU_max: ~0.78
                checkpoint = torch.load(cnn_branch_checkpt)
                model.cnn_branch.load_state_dict(checkpoint['model_state_dict'])
            if trans_branch_checkpt is not None:
                print(f'Loading Transformer checkpoint at path: {os.path.basename(trans_branch_checkpt)}')
                checkpoint = torch.load(trans_branch_checkpt)
                model.trans_branch.load_state_dict(checkpoint['model_state_dict'])
    elif model_name == "SimplestFusionNetwork":
        model = SimplestFusionNetwork(
            cnn_model_cfg, 
            trans_model_cfg, 
            with_weights=True
        ).cuda()
        if cnn_branch_checkpt is not None or trans_branch_checkpt is not None:
            # loading a checkpoint for a full model while having cnn checkpoint 
            # and transformer checkpoint loaded would make no sense (its either 
            # resume the whole model or one or both of the individual branches)
            assert resume is None 

            # checkpoint loading below uses same methodology as the if resume: 
            # structure does below, but we're only interested in loading the 
            # params of the model not the learning rate or optimizer settings
            # because we're using the trained params for the FUSED network to
            # see if we can get better results
            if cnn_branch_checkpt is not None:
                print(f'Loading CNN checkpoint at path: {os.path.basename(cnn_branch_checkpt)}') # IoU_max: ~0.78
                checkpoint = torch.load(cnn_branch_checkpt)
                model.cnn_branch.load_state_dict(checkpoint['model_state_dict'])
            if trans_branch_checkpt is not None:
                print(f'Loading Transformer checkpoint at path: {os.path.basename(trans_branch_checkpt)}')
                checkpoint = torch.load(trans_branch_checkpt)
                model.trans_branch.load_state_dict(checkpoint['model_state_dict'])
    elif model_name == "UNet_plain":
        model = UNet_plain(
            n_channels=3, 
            n_classes=1, 
            patch_size=16
        ).cuda()
    elif model_name == "UNet_backboned":
        assert image_size[0] == image_size[1], f'dk why this is in'
        model = CNN_BRANCH_WITH_BACKBONE(
            cnn_model_cfg['in_channels'], 
            cnn_model_cfg['num_classes'], 
            cnn_model_cfg['patch_size'], 
            cnn_model_cfg['backbone'], 
            pretrained=False,
            with_attention=False,
            with_superficial=False,
            input_size = image_size[0],
        ).cuda()
    elif model_name == 'just_trans':
        model = create_transformer(model_cfg = trans_model_cfg, decoder = 'linear').cuda()
    elif model_name == 'transV2':
        model = create_transformerV2(model_cfg = trans_model_cfg, decoder = 'linear').cuda()
    elif model_name == 'pranet':
        raise ValueError(f'PraNet needs modification of train_one_epoch to handle its multilevel loss function.')
        model = PraNet(channel=32).cuda()
    elif model_name == 'swinunet':
        model = SwinUnet().cuda()
    elif model_name == 'NewFusionNetwork':
        model = NewFusionNetwork(
            cnn_model_cfg,
            trans_model_cfg,
            cnn_pretrained=False,
            with_fusion=True,
        ).cuda()
    elif model_name == 'EffNet_B7':
        from z import EffNet_B7
        model = EffNet_B7(
            encoder_channels = (3, 64, 48, 80, 224, 640),
            decoder_channels = (256, 128, 64, 32, 16),
            num_classes=1,
        ).cuda()
    elif model_name == 'EffNet_B4':
        from z import EffNet_B4
        model = EffNet_B4(
            encoder_channels = (3, 48, 32, 56, 160, 448),
            decoder_channels = (256, 128, 64, 32, 16),
            num_classes=1,
        ).cuda()
    elif model_name == 'EffNet_B3':
        from z import EffNet_B3
        model = EffNet_B3(
            encoder_channels = (3, 40, 32, 48, 136, 384),
            decoder_channels = (256, 128, 64, 32, 16),
            num_classes=1,
        ).cuda()
    elif model_name == 'zedNet':
        from seg.model.zed.zedNet import zedNet
        model = zedNet(
            n_channels=3, 
            n_classes=1, 
            patch_size=16,
            bilinear=True,
            attention=True, 
        ).cuda()
    elif model_name == 'fusion_zed':
        from seg.model.Fusion.NewFusionNetwork import ZedFusionNetwork
        model = ZedFusionNetwork(
            cnn_model_cfg,
            trans_model_cfg,
            True,
        ).cuda()
    elif model_name == 'new_fusion_zed':
        from seg.model.Fusion.NewFusionNetwork import NewZedFusionNetwork
        model = NewZedFusionNetwork(
            cnn_model_cfg,
            trans_model_cfg,
            with_fusion=True,
        ).cuda()
    elif model_name == 'NewZedFusionNetworkNoOneHalf':
        from seg.model.Fusion.NewFusionNetwork import NewZedFusionNetworkNoOneHalf
        model = NewZedFusionNetworkNoOneHalf(
            cnn_model_cfg,
            trans_model_cfg,
            with_fusion=True,
        ).cuda()
    elif model_name == 'rahmat':
        from seg.model.rahmat.rFusion.rFusionNetwork import r_OldFusionNetwork
        model = r_OldFusionNetwork(
            cnn_model_cfg, 
            trans_model_cfg,
            cnn_pretrained=False,
            with_fusion=True,
            with_aspp=False,
        ).cuda()
    elif model_name == 'reduced_unet_fusion':
        # new fusion just with reduced factor of 3 instead of 2 
        from seg.model.Fusion.NewFusionReducedUnetCapacity import NewFusionNetworkReducedUNet
        model = NewFusionNetworkReducedUNet(
            cnn_model_cfg,
            trans_model_cfg,
            with_fusion=True,
        ).cuda()
    elif model_name == 'NewFusionNetworkWithMergingNo1_2NoWeightsForSegMaps':
        from seg.model.Fusion.CondensedFusion import NewFusionNetworkWithMergingNo1_2NoWeightsForSegMaps
        model = NewFusionNetworkWithMergingNo1_2NoWeightsForSegMaps(
            cnn_model_cfg,
            trans_model_cfg,
        ).cuda()
        print(f'model imported succesfully.')
    elif model_name == "NewFusionNetworkWithMergingNo1_2NoWeightsForSegMapsNoSqueeze":
        from seg.model.Fusion.CondensedFusion import NewFusionNetworkWithMergingNo1_2NoWeightsForSegMapsNoSqueeze
        model = NewFusionNetworkWithMergingNo1_2NoWeightsForSegMapsNoSqueeze(
            cnn_model_cfg,
            trans_model_cfg,
        ).cuda()
    elif model_name == "CondensedFusionNetwork":
        from seg.model.Fusion.CondensedFusion import CondensedFusionNetwork
        model = CondensedFusionNetwork(
            cnn_model_cfg,
            trans_model_cfg,
        ).cuda()
    elif model_name == 'ASPPFusionNetwork':
        from seg.model.Fusion.ASPP_fuse import ASPPFusionNetwork
        model = ASPPFusionNetwork(
            cnn_model_cfg,
            trans_model_cfg,
        ).cuda()
    elif model_name == 'SqueezeAndExitationFusion':
        from seg.model.Fusion.ASPP_fuse import SqueezeAndExitationFusion
        model = SqueezeAndExitationFusion(
            cnn_model_cfg,
            trans_model_cfg,
        ).cuda()
    elif model_name == 'NewFusionNetworkWeight':
        from seg.model.Fusion.NewFusionNetworkWithWeights import NewFusionNetworkWeight
        model = NewFusionNetworkWeight(
            cnn_model_cfg,
            trans_model_cfg,
        ).cuda()
    elif model_name == 'reduced_unet_fusion_small_patch':
        # new fusion just with reduced factor of 3 instead of 2 
        from seg.model.Fusion.NewFusionReducedUnetCapacity import NewFusionNetworkReducedUNetWithSmallPatch
        model = NewFusionNetworkReducedUNetWithSmallPatch(
            cnn_model_cfg,
            trans_model_cfg,
            with_fusion=True,
        ).cuda()
    elif model_name == 'FusionNetworkRFB':
        # this is GARBAGE 
        from seg.model.Fusion.RFB_Fusion.RFB_FusionNetwork import FusionNetworkRFB
        print(f'WARNING: RUNNING force_cudnn_initialization()')
        force_cudnn_initialization()
        model = FusionNetworkRFB(
            cnn_model_cfg,
            trans_model_cfg,
            out_chans = [256, 128, 64, 32],
            # out_chans = [64, 32, 16, 8]
            # out_chans = [512, 256, 128, 64],
            dilation1=1,
            dilation2=2,
            dilation3=3,
        ).cuda()
        # Total Trainable Params: 96.715725M
        count_parameters(model)
        exit(1)
    elif model_name == 'unet_invert':
        from seg.model.invert_unet import iUnet
        model = iUnet(
            n_channels=3,
            n_classes=1, 
            patch_size=16,
        ).cuda()
    else:
        raise ValueError(f'Invalid model_name: {model_name}')
    print(f'Model {model_name} loaded succesfully.')    
    ###########################################################################

    # multi-GPU training if possible - note this creates a problem in terms of results saving 
    num_gpu = torch.cuda.device_count()
    print(f'Number of GPUs: {num_gpu}')
    for i in range(num_gpu):
        print(f'Device name: {torch.cuda.get_device_name(i)}')
    if num_gpu > 1:
        if batch_size < num_gpu:
            raise ValueError(f'batch_size: {batch_size} < num_gpu: {num_gpu}. Cannot run parralelized training.')
            exit(1)
        useParallel=True
        if useParallel:
            model = torch.nn.DataParallel(model)
            model = model.cuda()
            print(f'Creating parallel training process. WARNING: This overwrites the model name and changes the name.')
            # print('Not inputting parallel training process right now. Too many problems with CE-YC-dlearn2.')
            # exit(1)

    # optimzer stuffs 
    params = model.parameters()
    beta1 = 0.5
    beta2 = 0.999
    optimizer = torch.optim.Adam(params, learning_rate, betas=(beta1, beta2))

    # scheduler stuff 
    if lr_sched == "multistep":
        decay1 = num_epochs // 2
        decay2 = num_epochs - num_epochs // 6
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[decay1, decay2], gamma=0.5)
    elif lr_sched == "step":
        step = num_epochs // 3
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5)
    elif lr_sched == "poly":
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / num_epochs)), 0.9)  ## scheduler 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  ## scheduler 2
    elif lr_sched == "warmpoly":
        scheduler = WarmupPoly(init_lr=learning_rate, total_ep=num_epochs,
                                warmup_ratio=0.05, poly_pow=0.90)
    else:
        scheduler = None

    if resume is not None:
        print(f'Resuming at path: {os.path.basename(resume)}')
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        learning_rate = optimizer.param_groups[0]['lr']
        grad_norm = 2.0 
    else:
        loss_init = 1e5
        best_loss = loss_init
        grad_norm = 2.0
        start_epoch = 1


    train_loader = get_dataset(
        dataset, 
        save_dir + "/data_train.npy", # location of train images.npy file (str)
        save_dir + "/mask_train.npy" , # location of train masks.npy file (str)
        batchsize=batch_size, 
        normalization="deit" if "deit" in backbone else "vit"
    )


    print("Begin training...")

    # to save results - get unique identifier to save checkpoints and results
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(f'results/{model._get_name()}'):
        os.mkdir(f'results/{model._get_name()}')
        os.mkdir(f'results/{model._get_name()}/{model._get_name()}_1')
        model_dir = f'results/{model._get_name()}/{model._get_name()}_1'
    else:
        i = 1
        while os.path.exists(f'results/{model._get_name()}/{model._get_name()}_{i}') \
            and os.path.exists(f'results/{model._get_name()}/{model._get_name()}_{i}/valid_loss_file.txt'): # check to see that the test loss files have been generated
            print(f'Dir: results/{model._get_name()}/{model._get_name()}_{i} already exists with save files.')
            i += 1
        model_dir = f'results/{model._get_name()}/{model._get_name()}_{i}'
        if not os.path.exists(f'results/{model._get_name()}/{model._get_name()}_{i}'):
            print(f'Dir: results/{model._get_name()}/{model._get_name()}_{i} doesnt exist. Creating new one.')
            os.mkdir(model_dir)
        else:
            print(f'Dir: results/{model._get_name()}/{model._get_name()}_{i} exists, but doesnt have any save files, using this one.')
    
    # create logger
    logging.basicConfig(filename=f'{model_dir}/LOG.log', level=logging.INFO, format='%(message)s')
    logging.info(f'model_dir: {model_dir}')

    # save details of training into yaml config file in results dir 
    total_model = {
        "dataset": dataset,
        "actual_model": model._get_name(),
        "model_name": model_name,
        "final_image_height": image_size[0],
        "final_image_width": image_size[1],

        "dropout": dropout,
        "drop_path_rate": drop_path_rate,
        "num_classes": n_cls,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lr_sched": lr_sched,
        "loss_type": loss_type,

        "trans_model_cfg": trans_model_cfg,
        "cnn_model_cfg": cnn_model_cfg,
    }
    for k, v in total_model.items():
        logging.info(f'{k}: {v}')
    logging.info('\n')
    results_file = open(f"{model_dir}/final_config.yml", "w")
    yaml.dump(total_model, results_file)
    results_file.close()

    checkpt_save_dir = model_dir + '/current_checkpoints/' 
    if not os.path.exists(checkpt_save_dir):
        os.mkdir(checkpt_save_dir)
    print(f'checkpoint_save_dir: {checkpt_save_dir}')

    if dataset != 'master':
        test_loss_list = list()
        test_iou_list = list()
        test_dice_list = list()
        valid_loss_list = list()
        valid_iou_list = list()
        valid_dice_list = list()
    else: # test_cls = ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir']
        test_CVC_300_loss_list = list()
        test_CVC_300_iou_list = list()
        test_CVC_300_dice_list = list()
        test_CVC_ClinicDB_loss_list = list()
        test_CVC_ClinicDB_iou_list = list()
        test_CVC_ClinicDB_dice_list = list()
        test_CVC_ColonDB_loss_list = list()
        test_CVC_ColonDB_iou_list = list()
        test_CVC_ColonDB_dice_list = list()
        test_ETIS_loss_list = list()
        test_ETIS_iou_list = list()
        test_ETIS_dice_list = list()
        test_Kvasir_loss_list = list()
        test_Kvasir_iou_list = list()
        test_Kvasir_dice_list = list()
        valid_loss_list = list()
        valid_iou_list = list()
        valid_dice_list = list()

    valid_loader = get_tDataset(
        image_root = save_dir + "/data_valid.npy",
        gt_root = save_dir + "/mask_valid.npy",
        normalize_gt = False,
        batch_size = 1,
        normalization = 'vit',
        num_workers = 4, 
        pin_memory=True,
    )
    if dataset != 'master':
        test_loader = get_tDataset(
            image_root = save_dir + "/data_test.npy",
            gt_root = save_dir + "/mask_test.npy",
            normalize_gt = False,
            batch_size = 1,
            normalization = "deit" if "deit" in backbone else "vit",
            num_workers = 4, 
            pin_memory=True,
        )
    else:
        # NOTE: ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir']
        test_loader = get_tDatasets_master(
            save_dir=save_dir,
            normalize_gt=False,
            batch_size=1, 
            normalization="deit" if "deit" in backbone else "vit", 
            num_workers=4,
            pin_memory=True,
        )

    if n_cls == 1:
        inferencer = InferenceModule(eps=0.0001, activation='0-1')
    elif n_cls == 2:
        inferencer = SegmentationMetrics(eps=1e-5, average=True, ignore_background=False, activation='sigmoid')
        raise NotImplementedError(f'n_cls=2 never works, always gives terrible metrics results and doesnt train')

    loss_fn_params = dict()
    if loss_type == "weight":
        loss_fn = Weighted()
    elif loss_type == "focal_tversky":
        loss_fn = FocalTverskyLoss(nonlin='sigmoid')
        loss_fn_params['alpha'] = 0.55 # put these in main() as a command line arg
        loss_fn_params['beta'] = 0.45 # put these in main() as a command line arg
        loss_fn_params['gamma'] = 1 # put these in main() as a command line arg 
    elif loss_type == "iou":
        loss_fn = IoULoss(nonlin='sigmoid')
    elif loss_type == "dice":
        loss_fn = DiceLoss(nonlin='sigmoid')
    elif loss_type == "diceBCE":
        loss_fn = DiceBCELoss(nonlin='sigmoid')
    elif loss_type == "focal_loss":
        loss_fn = FocalLoss(nonlin='sigmoid')
    elif loss_type == "tversky":
        loss_fn = TverskyLoss(nonlin='sigmoid')
    elif loss_type == "multiscaleIoUBCE":
        loss_fn = MultiScaleIoUBCELoss(num_losses=4, epoch_unfreeze=3)
    elif loss_type == 'custom':
        loss_fn = MultiScaleIoU(num_seg_maps=7)
    else:
        raise NotImplementedError(f'Just put more elif structures in here.')

    if dataset != 'master':
        for epoch in range(start_epoch, num_epochs + 1): 
            start = time.time()
            best_loss, meanTestLoss, meanTestIoU, meanTestDice, \
                meanValidLoss, meanValidIoU, meanValidDice = train_one_epochV2(
                curr_epoch = epoch, 
                total_epochs = num_epochs,
                train_loader = train_loader, 
                valid_loader = valid_loader,
                test_loader = test_loader,
                model = model, 
                inferencer = inferencer,
                num_classes = n_cls,
                optimizer = optimizer,
                batch_size = batch_size,
                grad_norm = grad_norm, 
                best_loss = best_loss,
                model_checkpoint_name = model_checkpoint_name,
                checkpt_save_dir = checkpt_save_dir,
                speed_test = speed_test,
                scheduler = scheduler,
                loss_fn=loss_fn,
                loss_fn_params=loss_fn_params,
                dataset=dataset,    # doesnt matter
                best_loss_index=0,  # doesnt matter
                )
            test_loss_list.append(meanTestLoss)
            test_iou_list.append(meanTestIoU)
            test_dice_list.append(meanTestDice)
            valid_loss_list.append(meanValidLoss)
            valid_iou_list.append(meanValidIoU)
            valid_dice_list.append(meanValidDice)
            end = time.time()
            print('Time per epoch: {:.4f} (s)\n'.format(end - start))
            logging.info('Time per epoch: {:.4f} (s)\n'.format(end - start))        
    else: # dataset == master
        # take `best_loss` wrt ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir', 'ALL']
        best_losses = ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir', 'ALL']
        assert best_loss_option in BEST_LOSS_OPTIONS, \
            f'Given: {best_loss_option} invalid. Options: {BEST_LOSS_OPTIONS}'
        best_loss_index = BEST_LOSS_OPTIONS.index(best_loss_option)
        print(f'Taking best loss WRT: {best_loss_index, best_loss_option}.')
        logging.info(f'Taking best loss WRT: {best_loss_index, best_loss_option}.') 
        for epoch in range(start_epoch, num_epochs + 1): 
            start = time.time()
            best_loss, test_loss_matrix, \
                meanValidLoss, meanValidIoU, meanValidDice = train_one_epochV2(
                curr_epoch = epoch, 
                total_epochs = num_epochs,
                train_loader = train_loader, 
                valid_loader = valid_loader,
                test_loader = test_loader,
                model = model, 
                inferencer = inferencer,
                num_classes = n_cls,
                optimizer = optimizer,
                batch_size = batch_size,
                grad_norm = grad_norm, 
                best_loss = best_loss,
                model_checkpoint_name = model_checkpoint_name,
                checkpt_save_dir = checkpt_save_dir,
                speed_test = speed_test,
                scheduler = scheduler,
                loss_fn=loss_fn,
                loss_fn_params=loss_fn_params,
                dataset=dataset,    
                best_loss_index=best_loss_index, 
                )
            test_CVC_300_loss_list.append(test_loss_matrix[0, 0])
            test_CVC_300_iou_list.append(test_loss_matrix[0, 1])
            test_CVC_300_dice_list.append(test_loss_matrix[0, 2])
            test_CVC_ClinicDB_loss_list.append(test_loss_matrix[1, 0])
            test_CVC_ClinicDB_iou_list.append(test_loss_matrix[1, 1])
            test_CVC_ClinicDB_dice_list.append(test_loss_matrix[1, 2])
            test_CVC_ColonDB_loss_list.append(test_loss_matrix[2, 0])
            test_CVC_ColonDB_iou_list.append(test_loss_matrix[2, 1])
            test_CVC_ColonDB_dice_list.append(test_loss_matrix[2, 2])
            test_ETIS_loss_list.append(test_loss_matrix[3, 0])
            test_ETIS_iou_list.append(test_loss_matrix[3, 1])
            test_ETIS_dice_list.append(test_loss_matrix[3, 2])
            test_Kvasir_loss_list.append(test_loss_matrix[4, 0])
            test_Kvasir_iou_list.append(test_loss_matrix[4, 1])
            test_Kvasir_dice_list.append(test_loss_matrix[4, 2])
            valid_loss_list.append(meanValidLoss)
            valid_iou_list.append(meanValidIoU)
            valid_dice_list.append(meanValidDice)
            end = time.time()
            print('Time per epoch: {:.4f} (s)\n'.format(end - start))    
            logging.info('Time per epoch: {:.4f} (s)\n'.format(end - start)) 

    print('Plotting and writing results files...')
    if dataset != 'master':
        plot_test_valid_loss(
            test_loss_list, 
            valid_loss_list, 
            num_epochs - start_epoch + 1,
            save_dir = model_dir,
        )

        # save test and validation losses to txt file 
        test_loss_file = open(f'{model_dir}/test_loss_file.txt', 'w')
        test_iou_file = open(f'{model_dir}/test_iou_file.txt', 'w')
        test_dice_file = open(f'{model_dir}/test_dice_file.txt', 'w')
        valid_loss_file = open(f'{model_dir}/valid_loss_file.txt', 'w')
        valid_iou_file = open(f'{model_dir}/valid_iou_file.txt', 'w')
        valid_dice_file = open(f'{model_dir}/valid_dice_file.txt', 'w')

        for i in range(len(test_loss_list)):
            test_loss_file.write(str(test_loss_list[i]) + '\n')
            test_iou_file.write(str(test_iou_list[i]) + '\n')
            test_dice_file.write(str(test_dice_list[i]) + '\n')
            valid_loss_file.write(str(valid_loss_list[i]) + '\n')
            valid_iou_file.write(str(valid_iou_list[i]) + '\n')
            valid_dice_file.write(str(valid_dice_list[i]) + '\n')

        test_loss_file.close()
        test_iou_file.close()
        test_dice_file.close()
        valid_loss_file.close()
        valid_iou_file.close()
        valid_dice_file.close()

        # visualization of output seg. map (w/ input image and ground truth)
        image_root = save_dir + "/data_test.npy"
        gt_root = save_dir + "/mask_test.npy"
        test_dl = getTestDatasetForVisualization(image_root, gt_root)
        # load in state dictionary - if you want , this is how you would do it just 
        # uncomment and put in the file path for the .pth  
        # path_to_dict = '/home/john/Documents/Dev_Linux/segmentation/trans_isolated/ztest_Just_Transformer_25.pth'
        # model.load_state_dict(torch.load(path_to_dict))
        visualizeModelOutputfromDataLoader(test_dl, model, 4,save_dir=model_dir)
    else:
        plot_test_valid_loss(
            test_CVC_300_loss_list, 
            valid_loss_list, 
            num_epochs - start_epoch + 1,
            save_dir = model_dir,
            title='Loss Curve: CVC 300',
            save_name='loss_curve_CVC_300.png'
        )
        plot_test_valid_loss(
            test_CVC_ClinicDB_loss_list, 
            valid_loss_list, 
            num_epochs - start_epoch + 1,
            save_dir = model_dir,
            title='Loss Curve: CVC ClinicDB',
            save_name='loss_curve_CVC_ClinicDB.png'
        )
        plot_test_valid_loss(
            test_CVC_ColonDB_loss_list, 
            valid_loss_list, 
            num_epochs - start_epoch + 1,
            save_dir = model_dir,
            title='Loss Curve: CVC ColonDB',
            save_name='loss_curve_CVC_ColonDB.png'
        )
        plot_test_valid_loss(
            test_ETIS_loss_list, 
            valid_loss_list, 
            num_epochs - start_epoch + 1,
            save_dir = model_dir,
            title='Loss Curve: ETIS',
            save_name='loss_curve_ETIS.png'
        )
        plot_test_valid_loss(
            test_Kvasir_loss_list, 
            valid_loss_list, 
            num_epochs - start_epoch + 1,
            save_dir = model_dir,
            title='Loss Curve: Kvasir',
            save_name='loss_curve_kvasir.png'
        )

        # save test and validation losses to txt file - MASTER 
        test_CVC_300_loss_file = open(f'{model_dir}/test_CVC_300_loss_file.txt', 'w')
        test_CVC_300_iou_file = open(f'{model_dir}/test_CVC_300_iou_file.txt', 'w')
        test_CVC_300_dice_file = open(f'{model_dir}/test_CVC_300_dice_file.txt', 'w')
        test_CVC_ClinicDB_loss_file = open(f'{model_dir}/test_CVC_ClinicDB_loss_file.txt', 'w')
        test_CVC_ClinicDB_iou_file = open(f'{model_dir}/test_CVC_ClinicDB_iou_file.txt', 'w')
        test_CVC_ClinicDB_dice_file = open(f'{model_dir}/test_CVC_ClinicDB_dice_file.txt', 'w')
        test_CVC_ColonDB_loss_file = open(f'{model_dir}/test_CVC_ColonDB_loss_file.txt', 'w')
        test_CVC_ColonDB_iou_file = open(f'{model_dir}/test_CVC_ColonDB_iou_file.txt', 'w')
        test_CVC_ColonDB_dice_file = open(f'{model_dir}/test_CVC_ColonDB_dice_file.txt', 'w')
        test_ETIS_loss_file = open(f'{model_dir}/test_ETIS_loss_file.txt', 'w')
        test_ETIS_iou_file = open(f'{model_dir}/test_ETIS_iou_file.txt', 'w')
        test_ETIS_dice_file = open(f'{model_dir}/test_ETIS_dice_file.txt', 'w')
        test_Kvasir_loss_file = open(f'{model_dir}/test_Kvasir_loss_file.txt', 'w')
        test_Kvasir_iou_file = open(f'{model_dir}/test_Kvasir_iou_file.txt', 'w')
        test_Kvasir_dice_file = open(f'{model_dir}/test_Kvasir_dice_file.txt', 'w')

        valid_loss_file = open(f'{model_dir}/valid_loss_file.txt', 'w')
        valid_iou_file = open(f'{model_dir}/valid_iou_file.txt', 'w')
        valid_dice_file = open(f'{model_dir}/valid_dice_file.txt', 'w')

        for i in range(len(test_CVC_300_loss_list)):
            test_CVC_300_loss_file.write(str(test_CVC_300_loss_list[i]) + '\n')
            test_CVC_300_iou_file.write(str(test_CVC_300_iou_list[i]) + '\n')
            test_CVC_300_dice_file.write(str(test_CVC_300_dice_list[i]) + '\n')
        for i in range(len(test_CVC_ClinicDB_loss_list)):
            test_CVC_ClinicDB_loss_file.write(str(test_CVC_ClinicDB_loss_list[i]) + '\n')
            test_CVC_ClinicDB_iou_file.write(str(test_CVC_ClinicDB_iou_list[i]) + '\n')
            test_CVC_ClinicDB_dice_file.write(str(test_CVC_ClinicDB_dice_list[i]) + '\n')
        for i in range(len(test_CVC_ColonDB_loss_list)):
            test_CVC_ColonDB_loss_file.write(str(test_CVC_ColonDB_loss_list[i]) + '\n')
            test_CVC_ColonDB_iou_file.write(str(test_CVC_ColonDB_iou_list[i]) + '\n')
            test_CVC_ColonDB_dice_file.write(str(test_CVC_ColonDB_dice_list[i]) + '\n')
        for i in range(len(test_ETIS_loss_list)):
            test_ETIS_loss_file.write(str(test_ETIS_loss_list[i]) + '\n')
            test_ETIS_iou_file.write(str(test_ETIS_iou_list[i]) + '\n')
            test_ETIS_dice_file.write(str(test_ETIS_dice_list[i]) + '\n')
        for i in range(len(test_Kvasir_loss_list)):
            test_Kvasir_loss_file.write(str(test_Kvasir_loss_list[i]) + '\n')
            test_Kvasir_iou_file.write(str(test_Kvasir_iou_list[i]) + '\n')
            test_Kvasir_dice_file.write(str(test_Kvasir_dice_list[i]) + '\n')
        for i in range(len(valid_loss_list)):
            valid_loss_file.write(str(valid_loss_list[i]) + '\n')
            valid_iou_file.write(str(valid_iou_list[i]) + '\n')
            valid_dice_file.write(str(valid_dice_list[i]) + '\n')

        test_CVC_300_loss_file.close()
        test_CVC_300_iou_file.close()
        test_CVC_300_dice_file.close()
        test_CVC_ClinicDB_loss_file.close()
        test_CVC_ClinicDB_iou_file.close()
        test_CVC_ClinicDB_dice_file.close()
        test_CVC_ColonDB_loss_file.close()
        test_CVC_ColonDB_iou_file.close()
        test_CVC_ColonDB_dice_file.close()
        test_ETIS_loss_file.close()
        test_ETIS_iou_file.close()
        test_ETIS_dice_file.close()
        test_Kvasir_loss_file.close()
        test_Kvasir_iou_file.close()
        test_Kvasir_dice_file.close()

        valid_loss_file.close()
        valid_iou_file.close()
        valid_dice_file.close()

        # visualization of output seg. map (w/ input image and ground truth)
        CVC_300_image_root = save_dir + "/data_CVC_300_test.npy"
        CVC_300_gt_root = save_dir + "/mask_CVC_300_test.npy"
        test_dl = getTestDatasetForVisualization(CVC_300_image_root, CVC_300_gt_root)
        # load in state dictionary - if you want , this is how you would do it just 
        # uncomment and put in the file path for the .pth  
        # path_to_dict = '/home/john/Documents/Dev_Linux/segmentation/trans_isolated/ztest_Just_Transformer_25.pth'
        # model.load_state_dict(torch.load(path_to_dict))
        visualizeModelOutputfromDataLoader(test_dl, model, 4,save_dir=model_dir, title='CVC 300 Model Visualization', save_name='CVC_300_model_visual.png')

        CVC_ClinicDB_image_root = save_dir + "/data_CVC_ClinicDB_test.npy"
        CVC_ClinicDB_gt_root = save_dir + "/mask_CVC_ClinicDB_test.npy"
        test_dl = getTestDatasetForVisualization(CVC_ClinicDB_image_root, CVC_ClinicDB_gt_root)
        visualizeModelOutputfromDataLoader(test_dl, model, 4,save_dir=model_dir, title='CVC Clinic DB Model Visualization', save_name='CVC_ClinicDB_model_visual.png')
        CVC_ColonDB_image_root = save_dir + "/data_CVC_ColonDB_test.npy"
        CVC_ColonDB_gt_root = save_dir + "/mask_CVC_ColonDB_test.npy"
        test_dl = getTestDatasetForVisualization(CVC_ColonDB_image_root, CVC_ColonDB_gt_root)
        visualizeModelOutputfromDataLoader(test_dl, model, 4,save_dir=model_dir, title='CVC Colon DB Model Visualization', save_name='CVC_ColonDB_model_visual.png')
        ETIS_image_root = save_dir + "/data_ETIS_test.npy"
        ETIS_gt_root = save_dir + "/mask_ETIS_test.npy"
        test_dl = getTestDatasetForVisualization(ETIS_image_root, ETIS_gt_root)
        visualizeModelOutputfromDataLoader(test_dl, model, 4,save_dir=model_dir, title='ETIS Model Visualization', save_name='ETIS_model_visual.png')
        Kvasir_image_root = save_dir + "/data_Kvasir_test.npy"
        Kvasir_gt_root = save_dir + "/mask_Kvasir_test.npy"
        test_dl = getTestDatasetForVisualization(Kvasir_image_root, Kvasir_gt_root)
        visualizeModelOutputfromDataLoader(test_dl, model, 4,save_dir=model_dir, title='Kvasir Model Visualization', save_name='Kvasir_model_visual.png')
    print('Plotting and data write complete.')
    ###########################################################################
    ###########################################################################

    if isinstance(model, SimplestFusionNetwork):
        if model.with_weights:
            print(f'Model weights (w_cnn, w_trans): {model.w_cnn, model.w_trans}')



if __name__ == '__main__':
    main()
