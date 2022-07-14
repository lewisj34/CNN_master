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
import socket 
import numpy as np 
import matplotlib.pyplot as plt 

def generateTriLossPlot(
    loss_path1, 
    loss_path2,
    loss_path3,
    curve_title,
    label_loss_path1,
    label_loss_path2,
    label_loss_path3,
    save_plot_name=None,
    showPlot=False,
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
    loss_data3 = np.loadtxt(loss_path3)

    assert len(loss_data1) == len(loss_data2), \
        f'len of .txt files must be same. Lengths: {len(loss_path1), len(loss_path2), len(loss_path3)}'
    num_epochs = len(loss_data1)
    epoch_data = np.arange(1, num_epochs + 1, 1)
    assert len(epoch_data) == len(loss_data1) == len(loss_data2) == len(loss_data3)

    SMALL_SIZE = 13
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.plot(epoch_data, loss_data1, label = label_loss_path1)
    plt.plot(epoch_data, loss_data2, label = label_loss_path2)
    plt.plot(epoch_data, loss_data3, label = label_loss_path3)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(curve_title)
    plt.legend()
    # plt.show()


    if save_plot_name is not None:
        print(f'Saving figures.')
        plt.savefig(save_plot_name)
        if showPlot:
            plt.show()
        else:
            plt.close()
    else:
        print(f'\nNot saving figures!')
        plt.show()


if __name__ == '__main__':

    parent_dir = 'loss_files_merged_combined_only_200'

    chart_dir = f'{parent_dir}/charts/'
    os.makedirs(chart_dir, exist_ok=True)

    valid_iou_path = f'{parent_dir}/valid_iou_file.txt'

    test_iou_path = f'{parent_dir}/test_iou_file.txt'

    train_iou_path = f'{parent_dir}/train_iou_file.txt'

    generateTriLossPlot(
        test_iou_path, 
        valid_iou_path, 
        train_iou_path, 
        f'IoU Loss Curve', 
        f'Test', 
        f'Valid', 
        f'Training', 
        save_plot_name=f'{chart_dir}/iou_plot.png',
    )