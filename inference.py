import torch 
import time 
from tqdm import tqdm 
import numpy as np 

from seg.utils.dataset import ValidDataset, TestDataset
from seg.utils.weighted_loss import weighted_loss
import seg.utils.lovasz_losses as L
from seg.utils.iou_dice import mean_dice_score, mean_iou_score, precision, recall, mean_precision, mean_recall
from seg.utils.accuracy import accuracy

def validate(model, data_dir, loss_type):
    '''
    Evaluate valiodation dataset
    '''
    model.eval()
    mean_loss = 0

    image_root = data_dir + "/data_valid.npy"
    gt_root = data_dir + "/mask_valid.npy"

    validLoader = ValidDataset(image_root, gt_root)

    dice_loss = list(); iou_loss = list(); loss_list = list(); acc_list = list()
    acc_v2list = list(); TP_list = list(); FP_list = list(); TN_list = list(); FN_list = list()
    prec_list = list(); recall_list = list()
    valid_size = validLoader.size

    for i in range(valid_size):
        images, masks = validLoader.load_data()
        images = images.cuda()

        with torch.no_grad():
            res = model(images)

        if loss_type == "weight":
            loss = weighted_loss(res, \
                torch.tensor(masks).unsqueeze(0).unsqueeze(0).cuda())
        elif loss_type == "lovasz":
            loss = L.lovasz_hinge(res, 
                torch.tensor(masks).unsqueeze(0).unsqueeze(0).cuda())
        else:
            print(f'loss type: {loss_type} not supported')
            exit(1)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        masks = 1*(masks>0.5)            
        res = 1*(res > 0.5)

        dice = mean_dice_score(masks, res)
        iou = mean_iou_score(masks, res)
        acc = np.sum(res == masks) / (res.shape[0]*res.shape[1])
        ACC, TP, FP, TN, FN = accuracy(masks.flatten(), res.flatten())
        prec = precision(TP, FP)
        rec = recall(TP, FN)

        loss_list.append(loss.item())
        dice_loss.append(dice)
        iou_loss.append(iou)
        acc_list.append(acc)
        acc_v2list.append(ACC)
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
        prec_list.append(prec)
        recall_list.append(rec)
    
    print("\nValidation Dataset Statistics: \n")
    print("\tLoss: {:.4f}".format(np.mean(loss_list)))
    print("\tDice: {:.4f}".format(np.mean(dice_loss)))
    print("\tIoU: {:.4f}".format(np.mean(iou_loss)))
    print("\tAccuracy: {:.4f}".format(np.mean(acc_list)))
    print("\tPrecision: {:.4f}".format(np.mean(prec_list)))
    print("\tRecall: {:.4f}".format(np.mean(recall_list)))
    print("\tTrue positive average: {:.4f}".format(np.mean(TP_list)))
    print("\tFalse positive average: {:.4f}".format(np.mean(FP_list)))
    print("\tTrue negative average: {:.4f}".format(np.mean(TN_list)))
    print("\tFalse negative average: {:.4f}".format(np.mean(FN_list)))


    return np.mean(loss_list), np.mean(iou_loss), np.mean(dice_loss)

def test(model, data_dir, loss_type):
    model.eval()
    mean_loss = 0

    image_root = data_dir + "/data_test.npy"
    gt_root = data_dir + "/mask_test.npy"

    testLoader = TestDataset(image_root, gt_root)

    dice_loss = list(); iou_loss = list(); loss_list = list(); acc_list = list()
    acc_v2list = list(); TP_list = list(); FP_list = list(); TN_list = list(); FN_list = list()
    prec_list = list(); recall_list = list()
    test_size = testLoader.size

    for i in range(test_size):
        images, masks = testLoader.load_data()
        images = images.cuda()

        # images.shape: torch.Size([1, 3, 256, 256]), (256, 256))
        # masks.shape: torch.size([256, 256])
        with torch.no_grad():
            res = model(images) # output: (256, 256)

        if loss_type == "weight":
            loss = weighted_loss(res, \
                torch.tensor(masks).unsqueeze(0).unsqueeze(0).cuda())
        elif loss_type == "lovasz":
            loss = L.lovasz_hinge(res, torch.tensor(masks).unsqueeze(0).unsqueeze(0).cuda())
        else:
            print(f'loss type: {loss_type} not supported')
            exit(1)

        res = res.sigmoid().data.cpu().numpy().squeeze() # output: (256, 256)

        masks = 1*(masks > 0.5) # output: (256, 256)           
        res = 1*(res > 0.5) # output: (256, 256)
        

        dice = mean_dice_score(masks, res)
        iou = mean_iou_score(masks, res)
        acc = np.sum(res == masks) / (res.shape[0]*res.shape[1])
        ACC, TP, FP, TN, FN = accuracy(masks.flatten(), res.flatten())
        prec = precision(TP, FP)
        rec = recall(TP, FN)

        loss_list.append(loss.item())
        dice_loss.append(dice)
        iou_loss.append(iou)
        acc_list.append(acc)
        acc_v2list.append(ACC)
        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)
        prec_list.append(prec)
        recall_list.append(rec)    

    print("\nTest Dataset Statistics: \n")
    print("\tLoss: {:.4f}".format(np.mean(loss_list)))
    print("\tDice: {:.4f}".format(np.mean(dice_loss)))
    print("\tIoU: {:.4f}".format(np.mean(iou_loss)))
    print("\tAccuracy: {:.4f}".format(np.mean(acc_list)))
    print("\tPrecision: {:.4f}".format(np.mean(prec_list)))
    print("\tRecall: {:.4f}".format(np.mean(recall_list)))
    print("\tTrue positive average: {:.4f}".format(np.mean(TP_list)))
    print("\tFalse positive average: {:.4f}".format(np.mean(FP_list)))
    print("\tTrue negative average: {:.4f}".format(np.mean(TN_list)))
    print("\tFalse negative average: {:.4f}".format(np.mean(FN_list)))

    return np.mean(loss_list), np.mean(iou_loss), np.mean(dice_loss)



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
    # print("\tTotal time cost: {}s".format(sum(time_list)))
    # print("\t     + Average time cost: {}s".format(sum(time_list)/10000))
    # print("\t     + Frame Per Second: {:.2f}".format(1/(sum(time_list)/10000)))



