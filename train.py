import os 
import torch
import click
import yaml
import time

from pathlib import Path
from seg.model.transformer.create_model import create_transformer
from seg.utils.data.generate_npy import split_and_convert_to_npyV2
from seg.utils.err_codes import crop_err1, crop_err2, crop_err3,resize_err1, resize_err2, resize_err3
from seg.model.CNN.CNN_backboned import CNN_BRANCH_WITH_BACKBONE
from seg.model.CNN.CNN_plus import UNet_plain
from seg.model.Fusion.FusionNetwork import OldFusionNetwork, SimplestFusionNetwork
from seg.model.losses.focal_tversky import FocalTverskyLoss
from seg.model.losses.iou_loss import IoULoss
from seg.model.losses.weighted import Weighted
from seg.model.losses.dice_loss import DiceLoss
from seg.model.losses.dicebce_loss import DiceBCELoss
from seg.model.losses.focal_loss import FocalLoss
from seg.model.losses.tversky import TverskyLoss
from seg.utils.inferenceV2 import InferenceModule, SegmentationMetrics
from seg.utils.t_dataset import get_tDataset

from seg.utils.visualize import visualizeModelOutputfromDataLoader, plot_test_valid_loss
from seg.utils.dataset import get_TestDatasetV2, get_dataset
from seg.utils.sched import WarmupPoly
from engineV2 import train_one_epochV2

ALLOWABLE_DATASETS = ['kvasir', 'CVC_ClinicDB', 'ETIS', 'CVC_ColonDB', 'master']
ALLOWABLE_MODELS = ["OldFusionNetwork", "SimplestFusionNetwork", "UNet_plain", \
    "UNet_backboned", "just_trans"]
ALLOWABLE_CNN_MODELS = ['unet']
ALLOWABLE_CNN_BACKBONES = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 
    'resnet152', 'vgg16', 'vgg19', 'densenet121', 'densenet161', 'densenet169', 
    'densenet201', 'unet_encoder', None]


# try typing this in upon restarting 
#  python train.py --num_epochs 20 --resume '/home/john/Documents/Dev_Linux/segmentation/trans_isolated/seg/current_checkpoints/Transformer/Transformer-8.pth'

@click.command(help='')
@click.option('--dataset', type=str, default='kvasir')
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
@click.option('--learning_rate', type=float, default=7e-5) # old: 7e-5 works way better than the new 10e-4
@click.option('--batch_size', type=int, default=16)
@click.option('--model_checkpoint_name', type=str, default=None)
@click.option('--speed_test', type=bool, default=False, help='if True, runs FPS measurements of network every 5 epochs')
@click.option('--resume', type=str, default=None, help='path to checkpoint')
@click.option('--cnn_branch_checkpt', type=str, default=None, help='path to cnn branch checkpt') # /home/john/Documents/Dev_Linux/segmentation/trans_isolated/seg/current_checkpoints/CNN_BRANCH_WITH_BACKBONE/CNN_BRANCH_WITH_BACKBONE-98.pth
@click.option('--trans_branch_checkpt', type=str, default=None, help='path to transformer branch checkpt') # /home/john/Documents/Dev_Linux/segmentation/trans_isolated/seg/current_checkpoints/Transformer/Transformer-22.pth
@click.option('--lr_sched', type=str, default='warmpoly', help='step, poly, multistep, warmpoly')
@click.option('--loss_type', type=str, default='iou', help='iou, weight, lovasz')
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
):

    assert model_name in ALLOWABLE_MODELS, 'invalid model_name'
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

    if model_name == "OldFusionNetwork":
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
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        print(f'Creating parallel training process. WARNING: This overwrites the model name and changes the name.')

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
            and os.path.exists(f'results/{model._get_name()}/{model._get_name()}_{i}/test_loss_file.txt'): # check to see that the test loss files have been generated
            print(f'Dir: results/{model._get_name()}/{model._get_name()}_{i} already exists with save files.')
            i += 1
        model_dir = f'results/{model._get_name()}/{model._get_name()}_{i}'
        if not os.path.exists(f'results/{model._get_name()}/{model._get_name()}_{i}'):
            print(f'Dir: results/{model._get_name()}/{model._get_name()}_{i} doesnt exist. Creating new one.')
            os.mkdir(model_dir)
        else:
            print(f'Dir: results/{model._get_name()}/{model._get_name()}_{i} exists, but doesnt have any save files, using this one.')

    checkpt_save_dir = model_dir + '/current_checkpoints/' 
    if not os.path.exists(checkpt_save_dir):
        os.mkdir(checkpt_save_dir)
    print(f'checkpoint_save_dir: {checkpt_save_dir}')

    test_loss_list = list()
    test_iou_list = list()
    test_dice_list = list()
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
    test_loader = get_tDataset(
        image_root = save_dir + "/data_test.npy",
        gt_root = save_dir + "/mask_test.npy",
        normalize_gt = False,
        batch_size = 1,
        normalization = 'vit',
        num_workers = 4, 
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
    else:
        raise NotImplementedError(f'Just put more elif structures in here.')

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
            )
        test_loss_list.append(meanTestLoss)
        test_iou_list.append(meanTestIoU)
        test_dice_list.append(meanTestDice)
        valid_loss_list.append(meanValidLoss)
        valid_iou_list.append(meanValidIoU)
        valid_dice_list.append(meanValidDice)
        end = time.time()
        print('Time per epoch: {:.4f} (s)\n'.format(end - start))        

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
    results_file = open(f"{model_dir}/final_config.yml", "w")
    yaml.dump(total_model, results_file)
    results_file.close()


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
    test_dl = get_TestDatasetV2(image_root, gt_root)
    # load in state dictionary - if you want , this is how you would do it just 
    # uncomment and put in the file path for the .pth  
    # path_to_dict = '/home/john/Documents/Dev_Linux/segmentation/trans_isolated/ztest_Just_Transformer_25.pth'
    # model.load_state_dict(torch.load(path_to_dict))
    visualizeModelOutputfromDataLoader(test_dl, model, 4,save_dir=model_dir)
    ###########################################################################
    ###########################################################################

    if isinstance(model, SimplestFusionNetwork):
        if model.with_weights:
            print(f'Model weights (w_cnn, w_trans): {model.w_cnn, model.w_trans}')



if __name__ == '__main__':
    main()
