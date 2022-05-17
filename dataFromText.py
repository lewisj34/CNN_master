"""
basically copy pasted from generate_plot_from_txt except now we can just run this from the command line 
in the main directory without haveing to run python -m seg.utils.generate_plot_from_txt
"""
import os 
import argparse
import numpy as np
import pandas as pd 

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

        # print("{:.2f}".format(3.1415926));
        print('\n\n')
        print(f'Param \t\t{tests[0]} \t\t{tests[1]} \t{tests[2]} \t{tests[3]} \t{tests[4]} \t\tEpoch Range')
        print("max(dice, iou) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t----------".format(np.max(dice_data_0), np.max(iou_data_0), np.max(dice_data_1), np.max(iou_data_1), np.max(dice_data_2), np.max(iou_data_2), np.max(dice_data_3), np.max(iou_data_3), np.max(dice_data_4), np.max(iou_data_4)))
        print("avg(dice, iou) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t[{}, {}]".format(np.mean(dice_data_0[start_epoch:end_epoch]), np.mean(iou_data_0[start_epoch:end_epoch]), np.mean(dice_data_1[start_epoch:end_epoch]), np.mean(iou_data_1[start_epoch:end_epoch]), np.mean(dice_data_2[start_epoch:end_epoch]), np.mean(iou_data_2[start_epoch:end_epoch]), np.mean(dice_data_3[start_epoch:end_epoch]), np.mean(iou_data_3[start_epoch:end_epoch]), np.mean(dice_data_4[start_epoch:end_epoch]), np.mean(iou_data_4[start_epoch:end_epoch]), start_epoch, end_epoch))
        print('\n\n')
        for i in range(len(tests)): # ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir']
            print(f'\t{tests[i]}')
            iou_path = dir + '/test_' + tests[i] + '_iou_file.txt'
            dice_path = dir + '/test_' + tests[i] + '_dice_file.txt'
            iou_data = np.loadtxt(iou_path)
            dice_data = np.loadtxt(dice_path)
            # print("{:.2f}".format(3.1415926));
            print("\t\tmax(dice, iou): ({:.3f}, {:.3f})".format(np.max(dice_data), np.max(iou_data)))
            print("\t\tavg(dice, iou): ({:.3f}, {:.3f}) for [{}, {}]".format(np.mean(dice_data[start_epoch:end_epoch]), np.mean(iou_data[start_epoch:end_epoch]), start_epoch, end_epoch))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', type=str)
    parser.add_argument('--start_epoch', type=int, default=750)
    parser.add_argument('--end_epoch', type=int, default=800)
    args = parser.parse_args()
    results_list = list(args.data)

    getAllDatasetStatisticsFromListDir(
        list_dirs = list(args.data),
        start_epoch = args.start_epoch,
        end_epoch = args.end_epoch,
    )
    ResultsStatsTableFromList(
        list_dirs = list(args.data),
        start_epoch = args.start_epoch,
        end_epoch = args.end_epoch,
    )