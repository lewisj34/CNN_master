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

    test_iou_loss_path_CVC_ClinicDB = getMajorStatisticsFromSingleLossPath(
        'results/OldFusionNetwork/OldFusionNetwork_9/test_iou_file.txt')
    test_dice_loss_path_CVC_ClinicDB = getMajorStatisticsFromSingleLossPath(
        'results/OldFusionNetwork/OldFusionNetwork_9/test_dice_file.txt')
    valid_iou_loss_path_CVC_ClinicDB = getMajorStatisticsFromSingleLossPath(
        'results/OldFusionNetwork/OldFusionNetwork_9/valid_iou_file.txt')
    valid_dice_loss_path_CVC_ClinicDB = getMajorStatisticsFromSingleLossPath(
        'results/OldFusionNetwork/OldFusionNetwork_9/valid_dice_file.txt')