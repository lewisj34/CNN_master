"""
Combines test_{CVC_300, CVC_ClinicDB, CVC_ColonDB, ETIS, Kvasir}_iou_file.txt 
and then
Generates a plot from a .txt file containing IoU

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
    test_loss_paths: list, 
    valid_loss_path: str,
    loss_path3: str,
    curve_title: str,
    label_test: str = "Test",
    label_valid: str = "Valid",
    label_train: str = "Train",
    save_plot_name: str =None,
    showPlot: bool = False,
    max_epoch_to_plot: int = None,
):
    """
    Combines all the test_loss_paths: {5 majors} into a single test loss path 
    Generates a plot of two .txt files (loss_path1, loss_path2). 
    Should be used with iou_loss from one model against iou_loss from another 
    model 
        @loss_path1: path to loss1.txt
        @loss_path2: path to loss2.txt
    """
    test_loss_data_list = list()
    for i in range(len(test_loss_paths)):
        test_loss_data_list.append(np.loadtxt(test_loss_paths[i]))

    test_avg_data = np.zeros(test_loss_data_list[0].shape)
    for i in range(len(test_loss_data_list)):
        test_avg_data += test_loss_data_list[i]
    test_avg_data = test_avg_data / len(test_loss_data_list) 

    valid__data = np.loadtxt(valid_loss_path)
    train_data = np.loadtxt(loss_path3)

    assert len(train_data) == len(valid__data) == len(test_loss_data_list[0]) == len(test_loss_data_list[1]), \
        f'len of .txt files must be same. Lengths: {len(test_loss_data_list[0]), len(valid_loss_path), len(train_data)}'
    num_epochs = len(train_data)
    epoch_data = np.arange(1, num_epochs + 1, 1)

    assert len(epoch_data) == len(test_loss_data_list[0]) == len(valid__data) == len(train_data), \
        f'{len(epoch_data), len(test_loss_data_list[0]), len(valid__data), len(train_data)}'

    if max_epoch_to_plot == None:
        max_epoch_to_plot = num_epochs

    SMALL_SIZE = 13
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rcParams["font.family"] = "Times New Roman"

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.plot(epoch_data[:max_epoch_to_plot], test_avg_data[:max_epoch_to_plot], label = label_test)
    plt.plot(epoch_data[:max_epoch_to_plot], valid__data[:max_epoch_to_plot], label = label_valid)
    plt.plot(epoch_data[:max_epoch_to_plot], train_data[:max_epoch_to_plot], label = label_train)
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

    parent_dir = 'loss_files_07_13_2022_combined_full_400/'

    chart_dir = f'{parent_dir}/charts/'
    os.makedirs(chart_dir, exist_ok=True)

    valid_iou_path = f'{parent_dir}/valid_iou_file.txt'

    test_iou_path = [
        f'{parent_dir}/test_CVC_300_iou_file.txt',
        f'{parent_dir}/test_CVC_ClinicDB_iou_file.txt',
        f'{parent_dir}/test_CVC_ColonDB_iou_file.txt',
        f'{parent_dir}/test_ETIS_iou_file.txt',
        f'{parent_dir}/test_Kvasir_iou_file.txt',
    ]

    train_iou_path = f'{parent_dir}/train_iou_file.txt'

    # generateTriLossPlot(
    #     test_iou_path, 
    #     valid_iou_path, 
    #     train_iou_path, 
    #     f'IoU Loss Curve', 
    #     f'Test', 
    #     f'Valid', 
    #     f'Training', 
    #     save_plot_name=f'{chart_dir}/iou_plot_master_completed_then_merged_200.png',
    #     max_epoch_to_plot = 200,
    # )
    generateTriLossPlot(
        test_iou_path, 
        valid_iou_path, 
        train_iou_path, 
        f'IoU Loss Curve - Original Dataset', 
        f'Test', 
        f'Valid', 
        f'Training', 
        save_plot_name=f'{chart_dir}/iou_plot_master_completed_then_merged_400.png',
        max_epoch_to_plot = 400,
    )

