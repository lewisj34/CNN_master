"""
basically copy pasted from generate_plot_from_txt except now we can just run this from the command line 
in the main directory without haveing to run python -m seg.utils.generate_plot_from_txt
"""
import os 
import argparse
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

def getAllDatasetStatisticsFromListDir(list_dirs: list, start_epoch, end_epoch):
    # tests = ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir']
    tests = ['Kvasir', 'CVC_ClinicDB', 'CVC_ColonDB', 'CVC_300', 'ETIS']
    ious_avg = np.zeros((len(tests), len(list_dirs)))
    ious_max = np.zeros((len(tests), len(list_dirs)))
    dice_max = np.zeros((len(tests), len(list_dirs)))
    dice_avg = np.zeros((len(tests), len(list_dirs)))
    for j in range(len(list_dirs)):
        dir = list_dirs[j]
        valid_iou_path = dir + '/test_iou_file.txt'
        valid_dice_path = dir + '/test_dice_file.txt'
        valid_loss_path = dir + '/test_loss_file.txt'

        print(f'result_dir: {dir}')
        for i in range(len(tests)): # ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir']
            print(f'\t{tests[i]}')
            iou_path = dir + '/test_' + tests[i] + '_iou_file.txt'
            dice_path = dir + '/test_' + tests[i] + '_dice_file.txt'
            iou_data = np.loadtxt(iou_path)
            dice_data = np.loadtxt(dice_path)
            # print("{:.2f}".format(3.1415926));
            print("\t\tmax(dice, iou): ({:.3f}, {:.3f})".format(np.max(dice_data), np.max(iou_data)))
            print("\t\tavg(dice, iou): ({:.3f}, {:.3f}) for [{}, {}]".format(np.mean(dice_data[start_epoch:end_epoch]), np.mean(iou_data[start_epoch:end_epoch]), start_epoch, end_epoch))
            # print(f'\t\tmax(dice, iou): ({np.max(dice_data)}, {np.max(iou_data)})')
            # print(f'\t\tm(dice, iou): {np.mean(dice_data[start_epoch:end_epoch])} {np.mean(iou_data[start_epoch:end_epoch])} for [{start_epoch}, {end_epoch}]')
                # print(f'\t\tmean dice loss between {start_epoch} and {end_epoch} epochs: {np.mean(dice_data[start_epoch:end_epoch])}')
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

    for i in range(len(list_dirs)):
        list_dirs[i] = os.path.basename(list_dirs[i])
    if not os.path.isdir('garbage'):
        os.mkdir('garbage')
    pd.DataFrame(ious_max).to_csv('garbage/ious_max.csv', header=list_dirs)
    pd.DataFrame(dice_max).to_csv('garbage/dice_max.csv', header=list_dirs)
    # pd.DataFrame(ious_max).to_csv('ious.csv', header=list_dirs)
    # pd.DataFrame(ious_max).to_csv('ious.csv', header=list_dirs)
    print(ious_max)
    print(ious_avg)
    print(dice_max)
    print(dice_avg)

def ResultsStatsTableFromList(list_dirs: list, start_epoch, end_epoch):
    tests = ['Kvasir', 'CVC_ClinicDB', 'CVC_ColonDB', 'CVC_300', 'ETIS']

    for j in range(len(list_dirs)):
        dir = list_dirs[j]
        valid_iou_path = dir + '/test_iou_file.txt'
        valid_dice_path = dir + '/test_dice_file.txt'
        valid_loss_path = dir + '/test_loss_file.txt'

        print(f'result_dir: {dir}')
        iou_path_0 = dir + '/test_' + tests[0] + '_iou_file.txt'
        dice_path_0 = dir + '/test_' + tests[0] + '_dice_file.txt'
        iou_data_0 = np.loadtxt(iou_path_0)
        dice_data_0 = np.loadtxt(dice_path_0)

        iou_path_1 = dir + '/test_' + tests[1] + '_iou_file.txt'
        dice_path_1 = dir + '/test_' + tests[1] + '_dice_file.txt'
        iou_data_1 = np.loadtxt(iou_path_1)
        dice_data_1 = np.loadtxt(dice_path_1)

        iou_path_2 = dir + '/test_' + tests[2] + '_iou_file.txt'
        dice_path_2 = dir + '/test_' + tests[2] + '_dice_file.txt'
        iou_data_2 = np.loadtxt(iou_path_2)
        dice_data_2 = np.loadtxt(dice_path_2)

        iou_path_3 = dir + '/test_' + tests[3] + '_iou_file.txt'
        dice_path_3 = dir + '/test_' + tests[3] + '_dice_file.txt'
        iou_data_3 = np.loadtxt(iou_path_3)
        dice_data_3 = np.loadtxt(dice_path_3)

        iou_path_4 = dir + '/test_' + tests[4] + '_iou_file.txt'
        dice_path_4 = dir + '/test_' + tests[4] + '_dice_file.txt'
        iou_data_4 = np.loadtxt(iou_path_4)
        dice_data_4 = np.loadtxt(dice_path_4)

        SET_dice_avg = np.mean((np.mean(dice_data_0[start_epoch:end_epoch]), np.mean(dice_data_1[start_epoch:end_epoch]), np.mean(dice_data_2[start_epoch:end_epoch]), np.mean(dice_data_3[start_epoch:end_epoch]), np.mean(dice_data_4[start_epoch:end_epoch])))
        SET_iou_avg = np.mean((np.mean(iou_data_0[start_epoch:end_epoch]), np.mean(iou_data_1[start_epoch:end_epoch]), np.mean(iou_data_2[start_epoch:end_epoch]), np.mean(iou_data_3[start_epoch:end_epoch]), np.mean(iou_data_4[start_epoch:end_epoch])))

        # print("{:.2f}".format(3.1415926));
        print('\n\n')
        print(f'Param \t\t{tests[0]} \t\t{tests[1]} \t{tests[2]} \t{tests[3]} \t{tests[4]} \t\tAvg(dice, IoU) \tEpoch Range')
        print("max(dice, iou) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t---------- \t----------".format(np.max(dice_data_0), np.max(iou_data_0), np.max(dice_data_1), np.max(iou_data_1), np.max(dice_data_2), np.max(iou_data_2), np.max(dice_data_3), np.max(iou_data_3), np.max(dice_data_4), np.max(iou_data_4)))
        print("avg(dice, iou) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t[{}, {}]".format(np.mean(dice_data_0[start_epoch:end_epoch]), np.mean(iou_data_0[start_epoch:end_epoch]), np.mean(dice_data_1[start_epoch:end_epoch]), np.mean(iou_data_1[start_epoch:end_epoch]), np.mean(dice_data_2[start_epoch:end_epoch]), np.mean(iou_data_2[start_epoch:end_epoch]), np.mean(dice_data_3[start_epoch:end_epoch]), np.mean(iou_data_3[start_epoch:end_epoch]), np.mean(dice_data_4[start_epoch:end_epoch]), np.mean(iou_data_4[start_epoch:end_epoch]), SET_dice_avg, SET_iou_avg, start_epoch, end_epoch))
        print('\n\n')

def generate5Plot(
    list_dirs: list, 
    plot_: str,
    save_plot_name: str = None,
    showPlot: bool = False,
):
    """
    Generates a plot of two .txt files (loss_path1, loss_path2). 
    Should be used with iou_loss from one model against iou_loss from another 
    model 
        @loss_path1: path to loss1.txt
        @loss_path2: path to loss2.txt
    """
    assert plot_ == "iou" or plot_ == "dice" or plot_ == "loss", f'Loss must be iou, dice or loss'

    tests = ['Kvasir', 'CVC_ClinicDB', 'CVC_ColonDB', 'CVC_300', 'ETIS']
    iou_data = []
    dice_data = []
    valid_iou_data = ""
    valid_dice_data = ""

    for j in range(len(list_dirs)):
        dir = list_dirs[j]
        valid_iou_path = dir + '/valid_iou_file.txt'
        valid_dice_path = dir + '/valid_dice_file.txt'
        valid_iou_data = np.loadtxt(valid_iou_path)
        valid_dice_data = np.loadtxt(valid_dice_path)


        valid_loss_path = dir + '/test_loss_file.txt'

        print(f'result_dir: {dir}')
        iou_path_0 = dir + '/test_' + tests[0] + '_iou_file.txt'
        dice_path_0 = dir + '/test_' + tests[0] + '_dice_file.txt'
        iou_data.append(np.loadtxt(iou_path_0))
        dice_data.append(np.loadtxt(dice_path_0))

        iou_path_1 = dir + '/test_' + tests[1] + '_iou_file.txt'
        dice_path_1 = dir + '/test_' + tests[1] + '_dice_file.txt'
        iou_data.append(np.loadtxt(iou_path_1))
        dice_data.append(np.loadtxt(dice_path_1))

        iou_path_2 = dir + '/test_' + tests[2] + '_iou_file.txt'
        dice_path_2 = dir + '/test_' + tests[2] + '_dice_file.txt'
        iou_data.append(np.loadtxt(iou_path_2))
        dice_data.append(np.loadtxt(dice_path_2))

        iou_path_3 = dir + '/test_' + tests[3] + '_iou_file.txt'
        dice_path_3 = dir + '/test_' + tests[3] + '_dice_file.txt'
        iou_data.append(np.loadtxt(iou_path_3))
        dice_data.append(np.loadtxt(dice_path_3))

        iou_path_4 = dir + '/test_' + tests[4] + '_iou_file.txt'
        dice_path_4 = dir + '/test_' + tests[4] + '_dice_file.txt'
        iou_data.append(np.loadtxt(iou_path_4))
        dice_data.append(np.loadtxt(dice_path_4))

    num_epochs = len(iou_data[0])
    epoch_data = np.arange(1, num_epochs + 1, 1)

    if plot_ == "iou":
        SMALL_SIZE = 13
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16

        plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.plot(epoch_data, iou_data[0], label = tests[0])
        plt.plot(epoch_data, iou_data[1], label = tests[1])
        plt.plot(epoch_data, iou_data[2], label = tests[2])
        plt.plot(epoch_data, iou_data[3], label = tests[3])
        plt.plot(epoch_data, iou_data[4], label = tests[4])
        plt.plot(epoch_data, valid_iou_data, label = "Valid")
        plt.plot(epoch_data, valid_iou_data, label = "Train")

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # plt.ylim(min(min(test_losses), min(valid_losses)), max(max(test_losses), max(valid_losses)))
        plt.title("IoU Loss Curve Comparison")
        plt.legend()
        plt.show()
    elif plot_ == "dice":
        SMALL_SIZE = 13
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16
        plt.rcParams["font.family"] = "Times New Roman"

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.plot(epoch_data, dice_data[0], label = tests[0])
        plt.plot(epoch_data, dice_data[1], label = tests[1])
        plt.plot(epoch_data, dice_data[2], label = tests[2])
        plt.plot(epoch_data, dice_data[3], label = tests[3])
        plt.plot(epoch_data, dice_data[4], label = tests[4])
        plt.plot(epoch_data, valid_dice_data, label = "Valid/Train")

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # plt.ylim(min(min(test_losses), min(valid_losses)), max(max(test_losses), max(valid_losses)))
        plt.title("Dice Loss Curve Comparison")
        plt.legend()
        # plt.show()
    elif plot_ == "loss":
        print(f'NOT IMPLEMENTED YET.')
        exit(1)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', type=str)
    args = parser.parse_args()
    results_list = list(args.data)

    generate5Plot(
        list_dirs = list(args.data),
        plot_ = "iou",
    )

    generate5Plot(
        list_dirs = list(args.data),
        plot_ = "dice",
    )

