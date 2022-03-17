import torch 
import numpy as np

from seg.utils.dataset import get_TestDatasetV2
from seg.model.Fusion.FusionNetwork import SimpleFusionNetwork, OldFusionNetwork, SimplestFusionNetwork
from seg.utils.iou_dice import mean_dice_score, mean_iou_score, precision, recall
from seg.utils.accuracy import accuracy
from seg.utils.visualize import visualizeModelOutputfromDataLoader

class Tester():
    def __init__(
        self,
        test_data_location, 
        model,
        model_pth,
        num_trials,
    ):
        '''
        For testing and inference of a model. Grabs and mIoU score and mDice 
        on a set number of instances to run the model through the test dataset. 
        Params:
            @test_data_location: .npy file location containing test data (dir)
            @model: the model to run the data through 
            @model_pth: the .pth object containing the trained model 
            @num_trials: the number of trials to measure IoU and Dice for 
        '''
        self.model = model
        self.model_pth = model_pth 
        self.num_trials = num_trials 

        # import data 
        image_root = test_data_location + '/data_test.npy'
        gt_root = test_data_location + '/mask_test.npy'
        self.test_dl = get_TestDatasetV2(image_root, gt_root)

        # load in state dictionary 
        # model.load_state_dict(torch.load(model_pth))

        # test results holder 
        self.mIoU = np.zeros(num_trials)
        self.mDice = np.zeros(num_trials)

        # print(f'mIoU.shape: {mIoU.shape}')
        # print(f'mIoU.shape: {mIoU.shape}')

    def run_once_through_test_data(self):
        '''
        runs through tests (IoU and Dice) for the input number of trials and 
        computes the average for these tests 
        '''
        model.eval()
        visualizeModelOutputfromDataLoader(self.test_dl, self.model)

        assert self.test_dl.batch_size == 1, 'batch_size must be == 1'

        dice_loss = list(); iou_loss = list(); loss_list = list(); acc_list = list()
        TP_list = list(); FP_list = list(); TN_list = list(); FN_list = list()
        prec_list = list(); recall_list = list()

        for i, image_gt in enumerate(self.test_dl):
            images, gts = image_gt

            images = images.cuda()
            gts = gts.cuda() 

            with torch.no_grad():
                output = model(images)
            output = output.sigmoid().data.cpu().numpy().squeeze().squeeze()
            gts = gts.data.cpu().numpy().squeeze().squeeze()

            gts = 1 * (gts > 0.5) # output: (256, 256)
            output = 1 * (output > 0.5) # output: (256, 256) 

            dice = mean_dice_score(gts, output)
            iou = mean_iou_score(gts, output)
            ACC, TP, FP, TN, FN = accuracy(gts.flatten(), output.flatten())
            prec = precision(TP, FP)
            rec = recall(TP, FN)
            dice_loss.append(dice)
            iou_loss.append(iou)
            acc_list.append(ACC)
            TP_list.append(TP)
            FP_list.append(FP)
            TN_list.append(TN)
            FN_list.append(FN)
            prec_list.append(prec)
            recall_list.append(rec)
        # print("\tLoss: {:.4f}".format(np.mean(loss_list)))
        print("\tDice: {:.4f}".format(np.mean(dice_loss)))
        print("\tIoU: {:.4f}".format(np.mean(iou_loss)))
        print("\tAccuracy: {:.4f}".format(np.mean(acc_list)))
        print("\tPrecision: {:.4f}".format(np.mean(prec_list)))
        print("\tRecall: {:.4f}".format(np.mean(recall_list)))
        print("\tTrue positive average: {:.4f}".format(np.mean(TP_list)))
        print("\tFalse positive average: {:.4f}".format(np.mean(FP_list)))
        print("\tTrue negative average: {:.4f}".format(np.mean(TN_list)))
        print("\tFalse negative average: {:.4f}".format(np.mean(FN_list)))
        avg_iou = np.mean(iou_loss)
        avg_dice = np.mean(dice_loss)
        return avg_iou, avg_dice

    def run_trials(self):
        for i in range(num_trials):
            self.mIoU[i], self.mDice[i] = self.run_once_through_test_data()
        return self.mIoU, self.mDice

if __name__ == '__main__':
    from default_cfgs import get_cfgs
    
    # generate model
    trans_model_cfg, cnn_model_cfg = get_cfgs()
    model = OldFusionNetwork(cnn_model_cfg, trans_model_cfg).cuda()
    model_pth = '/home/john/Documents/Dev_Linux/segmentation/trans_isolated/seg/current_checkpoints/OldFusionNetwork/OldFusionNetwork-28.pth'
    model.load_state_dict(torch.load(model_pth))
    test_data_location = 'seg/data/kvasir'
    num_trials = 1

    tester = Tester(
        test_data_location,
        model,
        model_pth,
        num_trials
    )

    mIoU, mDice = tester.run_trials()
    print(f'mIoU, mDice: {mIoU, mDice}')

