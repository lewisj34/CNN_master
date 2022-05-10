"""
Generates a plot from a .txt file containing IoU, Dice, or General loss

.txt file should be organized like: 
    0.743
    0.764
    0.773
    0.783
    0.797
    0.801
"""
import os
import numpy as np 
import matplotlib.pyplot as plt 

def generateIndividualPlot(loss_path):
    """
    Generates an individual plot of a .txt file containing a sequence of losses
    across epochs. Length of .txt file is taken to be the total number of epochs
        @loss_path: the path to the loss.txt file 
    """
    file_data = np.loadtxt(loss_path)
    for i in range(len(file_data)):
        print(file_data[i])    

def generateDualLossPlot(
    loss_path1, 
    loss_path2,
    curve_title,
    label_loss_path1,
    label_loss_path2,
):
    """
    Generates a plot of two .txt files (loss_path1, loss_path2). 
    Should be used with iou_loss from one model against iou_loss from another 
    model 
        @loss_path1: path to loss1.txt
        @loss_path2: path to loss2.txt
    """
    loss_data1 = np.loadtxt(loss_path1)
    loss_data2 = np.loadtxt(loss_path2)

    assert len(loss_data1) == len(loss_data2), \
        f'len of .txt files must be same. Lengths: {len(loss_path1), len(loss_path2)}'
    num_epochs = len(loss_data1)
    epoch_data = np.arange(1, num_epochs + 1, 1)
    assert len(epoch_data) == len(loss_data1) == len(loss_data2)

    plt.plot(epoch_data, loss_data1, label = label_loss_path1)
    plt.plot(epoch_data, loss_data2, label = label_loss_path2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.ylim(min(min(test_losses), min(valid_losses)), max(max(test_losses), max(valid_losses)))
    plt.title(curve_title)
    plt.legend()
    plt.show()

    # this is specific to the paths 
        # iou_loss1 = 'results/Transformer/Transformer_6/test_iou_file.txt'
        # iou_loss2 = 'results/Transformer/Transformer_7/test_iou_file.txt'
    max_loss1 = np.max(loss_data1)
    max_loss2 = np.max(loss_data2)

    mean_loss1_bw80and100 = np.mean(loss_data1[80:])
    mean_loss2_bw80and100 = np.mean(loss_data2[80:])
    print(f'max loss from loss_path1: {max_loss1}')
    print(f'max loss from loss_path2: {max_loss2}')
    print(f'mean_loss between_80_and_100 loss_path1: {mean_loss1_bw80and100}')
    print(f'mean_loss between_80_and_100 loss_path2: {mean_loss2_bw80and100}')

def getMajorStatisticsFromSingleLossPath(loss_path):
    loss_data = np.loadtxt(loss_path)
    print(f'file: {os.path.basename(loss_path)}')
    print(f'\tmax loss: {np.max(loss_data)}')
    print(f'\tmean loss between 80 and 100 epochs: {np.mean(loss_data[80:])}')

def getAllDatasetStatisticsFromDir(dir):
    valid_iou_path = dir + '/test_iou_file.txt'
    valid_dice_path = dir + '/test_dice_file.txt'
    valid_loss_path = dir + '/test_loss_file.txt'

    tests = ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir']
    for i in range(len(tests)):
        print(tests[i])
        iou_path = dir + '/test_' + tests[i] + '_iou_file.txt'
        dice_path = dir + '/test_' + tests[i] + '_dice_file.txt'
        iou_data = np.loadtxt(iou_path)
        dice_data = np.loadtxt(dice_path)
        print(f'\tmax iou loss: {np.max(iou_data)}')
        print(f'\tmean iou loss between 80 and 100 epochs: {np.mean(iou_data[750:])}')
        print(f'\tmax dice loss: {np.max(dice_data)}')
        print(f'\tmean dice loss between 80 and 100 epochs: {np.mean(dice_data[750:])}')

def getAllDatasetStatisticsFromListDir(list_dirs: list, start_epoch=80, end_epoch=None):
    tests = ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir']
    ious_avg = np.zeros((len(tests), len(list_dirs)))
    ious_max = np.zeros((len(tests), len(list_dirs)))
    dice_max = np.zeros((len(tests), len(list_dirs)))
    dice_avg = np.zeros((len(tests), len(list_dirs)))
    for j in range(len(list_dirs)):
        dir = list_dirs[j]
        valid_iou_path = dir + '/test_iou_file.txt'
        valid_dice_path = dir + '/test_dice_file.txt'
        valid_loss_path = dir + '/test_loss_file.txt'

        
        for i in range(len(tests)): # ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir']
            print(tests[i])
            iou_path = dir + '/test_' + tests[i] + '_iou_file.txt'
            dice_path = dir + '/test_' + tests[i] + '_dice_file.txt'
            iou_data = np.loadtxt(iou_path)
            dice_data = np.loadtxt(dice_path)
            print(f'\tmax iou loss: {np.max(iou_data)}')
            print(f'\tmax dice loss: {np.max(dice_data)}')
            if end_epoch==None:
                print(f'\tmean iou loss between {start_epoch} and end epochs: {np.mean(iou_data[start_epoch:])}')
                print(f'\tmean dice loss between {start_epoch} and end epochs: {np.mean(dice_data[start_epoch:])}')
            else: 
                print(f'\tmean iou loss between {start_epoch} and {end_epoch} epochs: {np.mean(iou_data[start_epoch:end_epoch])}')
                print(f'\tmean dice loss between {start_epoch} and {end_epoch} epochs: {np.mean(dice_data[start_epoch:end_epoch])}')
            if end_epoch == None:
                ious_max[i, j] = np.max(iou_data)
                ious_avg[i, j] = np.mean(iou_data[start_epoch:])
                dice_max[i, j] = np.max(dice_data)
                dice_avg[i, j] = np.mean(dice_data[start_epoch:])
            else:
                ious_max[i, j] = np.max(iou_data)
                ious_avg[i, j] = np.mean(iou_data[start_epoch:end_epoch])
                dice_max[i, j] = np.max(dice_data)
                dice_avg[i, j] = np.mean(dice_data[start_epoch:end_epoch])

    import pandas as pd 
    import os
    for i in range(len(list_dirs)):
        list_dirs[i] = os.path.basename(list_dirs[i])
    pd.DataFrame(ious_max).to_csv('ious_max.csv', header=list_dirs)
    pd.DataFrame(dice_max).to_csv('dice_max.csv', header=list_dirs)
    # pd.DataFrame(ious_max).to_csv('ious.csv', header=list_dirs)
    # pd.DataFrame(ious_max).to_csv('ious.csv', header=list_dirs)
    print(ious_max)
    print(ious_avg)
    print(dice_max)
    print(dice_avg)
if __name__ == '__main__':
    iou_loss1 = 'results/OldFusionNetwork/OldFusionNetwork_3/test_dice_file.txt'
    iou_loss2 = 'results/OldFusionNetwork/OldFusionNetwork_4/test_dice_file.txt'

    # generateDualLossPlot(
    #     iou_loss1, 
    #     iou_loss2,
    #     'Valid IoU Loss Comparison for Fusion Network',
    #     'With pretrained CNN',
    #     'Without pretrained CNN'
    # )

    # iou_newTransformerWMoreClasses = getMajorStatisticsFromSingleLossPath(
    #     'results/TransformerV2/TransformerV2_1/test_iou_file.txt')
    # dice_newTransformerWMoreClasses = getMajorStatisticsFromSingleLossPath(
    #     'results/TransformerV2/TransformerV2_1/test_dice_file.txt')
    # iou_og_transformer_w_decoderPlus = getMajorStatisticsFromSingleLossPath(
    #     'results/Transformer/Transformer_1/test_iou_file.txt')
    # dice_og_transformer_w_decoderPlus = getMajorStatisticsFromSingleLossPath(
    #     'results/Transformer/Transformer_1/test_dice_file.txt')
    # iou_original_version_no_decoderPlusTransformer = getMajorStatisticsFromSingleLossPath(
    #     'results/Transformer/Transformer_2/test_iou_file.txt')
    # dice_original_version_no_decoderPlusTransformer = getMajorStatisticsFromSingleLossPath(
    #     'results/Transformer/Transformer_2/test_dice_file.txt')

    getAllDatasetStatisticsFromListDir(['results/DataParallel/DataParallel_12'])
    # getAllDatasetStatisticsFromDir(dir='results/EffNet_B3/EffNet_B3_1')
    # getAllDatasetStatisticsFromDir(dir='results/EffNet_B4/EffNet_B4_1')
    # dirs = [
    #     'results/EffNet_B3/EffNet_B3_1', 
    #     'results/EffNet_B4/EffNet_B4_1', 
    #     'results/EffNetB_7/EffNet_B7_1',
    #     'results/UNet_plain_1'
    # ]
    # getAllDatasetStatisticsFromListDir(dirs)