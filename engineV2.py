import logging
import numpy as np
import torch
import time 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.autograd import Variable
from seg.model.losses.IoU_BCE_MultiScale import MultiScaleIoUBCELoss
from seg.model.losses.custom import MultiScaleIoU
from seg.model.losses.focal_loss import FocalLoss
from seg.model.losses.focal_tversky import FocalTverskyLoss
from seg.model.losses.tversky import TverskyLoss
from seg.utils.inferenceV2 import inference, inference_multi_class
from seg.utils.inference_func_master import inference_master
from seg.utils.soft_dice_loss import SoftDiceLoss

from seg.utils.utils import AvgMeter
from seg.utils.weighted_loss import weighted_loss
import seg.utils.lovasz_losses as L

from seg.utils.sched import WarmupPoly

def speed_testing(model, input_height, input_width, device_id=0, num_iters=10000):
    ''' 
    Evaluates the FPS for a model 
        @model: the pytorch model (nn.Module)
        @input_height: the resized input height for the model 
        @input_width: the resized input width for the model 
        @num_iters: the number of iterations to run the model for 
        @device_id: GPU id 
    '''
    # cuDnn configurations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    print('\n[SPEED EVAL]: Evaluating speed of model...')
    logging.info('\n[SPEED EVAL]: Evaluating speed of model...')
    model = model.to('cuda:{}'.format(device_id))
    random_input = torch.randn(1, 3, input_height, input_width).to('cuda:{}'.format(device_id))

    model.eval()

    time_list = []
    for i in tqdm(range(num_iters + 1)):
        torch.cuda.synchronize()
        tic = time.time()
        model(random_input)
        torch.cuda.synchronize()
        # the first iteration time cost much higher, so exclude the first iteration
        #print(time.time()-tic)
        time_list.append(time.time()-tic)
    time_list = time_list[1:]
    print(f'[SPEED EVAL]: Completed {num_iters} iterations of inference')
    print(f'[SPEED EVAL]: Total time elapsed {sum(time_list) / 60} mins')
    print(f'[SPEED EVAL]: Average time per iter {sum(time_list)/num_iters} sec')
    print('[SPEED EVAL]: FPS: {:.2f}'.format(1/(sum(time_list)/10000)))

    logging.info(f'[SPEED EVAL]: Completed {num_iters} iterations of inference')
    logging.info(f'[SPEED EVAL]: Total time elapsed {sum(time_list) / 60} mins')
    logging.info(f'[SPEED EVAL]: Average time per iter {sum(time_list)/num_iters} sec')
    logging.info('[SPEED EVAL]: FPS: {:.2f}'.format(1/(sum(time_list)/10000)))
    # print("\tTotal time cost: {}s".format(sum(time_list)))
    # print("\t     + Average time cost: {}s".format(sum(time_list)/10000))
    # print("\t     + Frame Per Second: {:.2f}".format(1/(sum(time_list)/10000)))

def train_one_epochV2(
    curr_epoch, 
    total_epochs,
    train_loader, 
    valid_loader,
    test_loader,
    model: nn.Module, 
    inferencer,
    num_classes,
    optimizer: optim.Adam,
    batch_size,
    grad_norm, 
    best_loss,
    model_checkpoint_name,
    checkpt_save_dir,
    speed_test,
    scheduler=None,
    loss_fn:nn.Module=None,
    loss_fn_params=None,
    eps=0.001,
    dataset='master', # only used to initiate what kind of test metric function is called
    best_loss_index=5,
    ):
    '''
    Trains model for one epoch.
    Args:
        @curr_epoch: current epoch
        @total_epochs: total number of epochs
        @train_loader: train loader (torch)
        @model: the model to train
        @optimizer: the optimizer 
        @batch_size: size of batch
        @grad_norm: gradient norm clip
        @best_loss: best current loss 
        @model_checkpoint_name: name underwhich checkpoint is saved
        @checkpt_save_dir: directory where checkpoints are to be saved 
        @data_dir: directory where train, valdiation, and test data are 
        @speed_test: whether it runs speeding tests or not to measure FPS
        @scheduler: learning rate scheduler 
        @dataset: name of dataset, only used to initiate the right infer. call
        @best_loss_index: only applies to master. Which testset 2 use 4 infer. 
            B/c we have 5 test sets that we measure model accuracy on, we can 
            theoretically take our best loss with respect to any one of them. 
            Will apply to the later determined test_loss_matrix (5 x 3) and det.
            with which to take the best loss from. Index is between 0 and 5, 
            and corresponds to the following list:    
            ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir', 'ALL']
    '''
    model.train()
    loss_record = AvgMeter()

    if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        scheduler.step(curr_epoch)
    elif isinstance(scheduler, WarmupPoly):
        curr_lr = scheduler.get_lr(curr_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
    else: 
        scheduler.step()

    for i, pack in enumerate(train_loader, start=1):
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        output = model(images)

        if isinstance(loss_fn, TverskyLoss):
            loss_val = loss_fn(
                inputs=output, 
                targets=gts, 
                smooth=eps, 
                alpha=loss_fn_params['alpha'], 
                beta=loss_fn_params['beta'], 
                gamma=loss_fn_params['gamma']
            )
        elif isinstance(loss_fn, FocalTverskyLoss):
            loss_val = loss_fn(
                inputs=output, 
                targets=gts, 
                smooth=eps, 
                alpha=loss_fn_params['alpha'], 
                beta=loss_fn_params['beta'], 
            )
        elif isinstance(loss_fn, FocalLoss):
            loss_val = loss_fn(
                inputs=output, 
                targets=gts, 
                smooth=eps, 
                alpha=loss_fn_params['alpha'], 
                gamma=loss_fn_params['gamma'],
            )
        elif isinstance(loss_fn, MultiScaleIoUBCELoss):
            loss_val = loss_fn(
                lateral_map_5 = F.interpolate(model.x_1_2, size=(256, 256)),
                lateral_map_4 = F.interpolate(model.x_1_4, size=(256, 256)),
                lateral_map_3 = F.interpolate(model.x_1_8, size=(256, 256)),
                lateral_map_2 = F.interpolate(model.x_1_16, size=(256, 256)), 
                gts = gts, 
                epoch = curr_epoch,
            )
        elif isinstance(loss_fn, MultiScaleIoU):
            loss_val = loss_fn(
                targets=gts, 
                input_full=output, # the global pooled output across all segmentation maps
                input_full_cnn=model.x_final_cnn,
                input_full_trans=model.x_final_trans,
                input_1_2=model.x_1_2, 
                input_1_4=model.x_1_4, 
                input_1_8=model.x_1_8, 
                input_1_16=model.x_1_16, 
            )
        else:
            loss_val = loss_fn(output, gts, smooth=eps) # note smooth not used for Weighted, also weight calls sigmoid 

        loss_val.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        loss_record.update(loss_val.data, batch_size)

        # scheduler stuffs 
        learning_rate_opt = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, WarmupPoly):
                learning_rate_sch = scheduler.get_lr(epoch=curr_epoch)
            else:
                learning_rate_sch = scheduler.get_last_lr()[0]

            # print(f'learning_rate_opt, learning_Rate_sch: {learning_rate_opt, learning_rate_sch}')
            assert learning_rate_opt == learning_rate_sch
            learning_rate = learning_rate_opt
        else:
            learning_rate = learning_rate_opt

        if i % 20 == 0 or i == len(train_loader):
            print('EPOCH: [{:03d}/{:03d}] \t STEP: [{:04d}/{:04d}] \t '
                  'LOSS: {:.4f} \t LR: {:.4e}'.  
                  format(curr_epoch, total_epochs, i, len(train_loader),
                         loss_record.show(), learning_rate))
            logging.info('EPOCH: [{:03d}/{:03d}] \t STEP: [{:04d}/{:04d}] \t '
                  'LOSS: {:.4f} \t LR: {:.4e}'.  
                  format(curr_epoch, total_epochs, i, len(train_loader),
                         loss_record.show(), learning_rate))

    if model_checkpoint_name is None:
        model_checkpoint_name = model._get_name()

    # checkpoint_path = checkpt_save_dir + '/'

    # os.makedirs(checkpoint_path, exist_ok=True)

    # old version: where we only need to return the individual tensors for a 
    # single test set (not across multiple different tests lets loaded into 
    # different dataloaders)
    if dataset != 'master': 
        if (curr_epoch+1) % 1 == 0:
            if num_classes == 1:
                meanValidLoss, meanValidIoU, meanValidDice  = inference(model, valid_loader, inferencer, loss_fn, 'valid')
                meanTestLoss, meanTestIoU, meanTestDice = inference(model, test_loader, inferencer, loss_fn, 'test')
            elif num_classes == 2:
                meanValidLoss, meanValidIoU, meanValidDice  = inference_multi_class(model, valid_loader, inferencer, loss_fn, 'valid')
                meanTestLoss, meanTestIoU, meanTestDice = inference_multi_class(model, test_loader, inferencer, loss_fn, 'test')         

            if meanTestLoss.item() < best_loss:
                print('New best loss: ', meanTestLoss.item())
                logging.info(f'New best loss: {meanTestLoss.item()}')
                best_loss = meanTestLoss.item()
                
                # old version for saving 
                # torch.save(model.state_dict(), checkpoint_path + f'{model_checkpoint_name}-%d.pth' % curr_epoch) 
                
                # new version where we try to save epoch info and other stuff
                # note in our updated code below, we DONT save the 'learning_rate', 
                # this isn't actually necessary as it stands right now (i.e. w/o the
                # learning rate scheduler), because the updated learning rate is 
                # saved in the optimizer.state_dict(), which can be accessed by the
                # following snippet of code: `lr = optimizer.param_groups[0]['lr']`
                # what I'm concerned about is when we input the scheduler into this 
                # code, what is going to happen, will the learning rate be the same? 
                # the code above should do that, but if they are different i think 
                # it would be better to save the learning rate from the lr scheduler
                # if they are not the same... 
                torch.save({
                    'epoch': curr_epoch,
                    'model_name': model._get_name(),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'learning_rate': learning_rate,
                    'loss': meanTestLoss.item(),
                }, checkpt_save_dir + f'{model_checkpoint_name}-%d.pth' % curr_epoch)

                print('Saving checkpoint to: ', 
                    checkpt_save_dir + f'{model_checkpoint_name}-%d.pth'% curr_epoch, "\n")
                checkpt_save_file = checkpt_save_dir + f'{model_checkpoint_name}-%d.pth'% curr_epoch
                logging.info(f'Saving checkpoint to: {checkpt_save_file}')

        if speed_test:
            if (curr_epoch+1) % 5 == 0: 
                speed_testing(
                    model, 
                    images.shape[2], 
                    images.shape[3], 
                    device_id=0, 
                    num_iters=10000
                )

        return best_loss, meanTestLoss.item(), meanTestIoU.item(), meanTestDice.item(), meanValidLoss.item(), meanValidIoU.item(), meanValidDice.item()

    else: # dataset == master 
        if (curr_epoch+1) % 1 == 0:
            assert num_classes == 1, f'num_classes: 2 not supported for master set'
            meanValidLoss, meanValidIoU, meanValidDice = inference_master(
                model=model,
                loader=valid_loader,
                inferencer=inferencer,
                loss_fn=loss_fn,
                test_type='valid',
            )
            test_loss_matrix = inference_master(
                model=model,
                loader=test_loader,
                inferencer=inferencer,
                loss_fn=loss_fn,
                test_type='test',
            )
            assert isinstance(best_loss_index, int), \
                f'best_loss_index must be int'
            if best_loss_index == 5: # take best_loss by averaging loss of ALL 
                # datasets value for the loss_fn 
                meanTestLoss = test_loss_matrix[:, 0].mean() 
            else:
                meanTestLoss = test_loss_matrix[best_loss_index, 0]

            if meanTestLoss < best_loss:
                print('New best loss: ', meanTestLoss)
                logging.info(f'New best loss: {meanTestLoss}')
                best_loss = meanTestLoss
                
                torch.save({
                    'epoch': curr_epoch,
                    'model_name': model._get_name(),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'learning_rate': learning_rate,
                    'loss': meanTestLoss.item(),
                }, checkpt_save_dir + f'{model_checkpoint_name}-%d.pth' % curr_epoch)

                print('Saving checkpoint to: ', 
                    checkpt_save_dir + f'{model_checkpoint_name}-%d.pth'% curr_epoch, "\n")
                checkpt_save_file = checkpt_save_dir + f'{model_checkpoint_name}-%d.pth'% curr_epoch
                logging.info(f'Saving checkpoint to: {checkpt_save_file}')

        if speed_test:
            if (curr_epoch+1) % 5 == 0: 
                speed_testing(
                    model, 
                    images.shape[2], 
                    images.shape[3], 
                    device_id=0, 
                    num_iters=10000
                )

        return best_loss, test_loss_matrix, meanValidLoss.item(), meanValidIoU.item(), meanValidDice.item()


