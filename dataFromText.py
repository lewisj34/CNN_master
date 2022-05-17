"""
basically copy pasted from generate_plot_from_txt except now we can just run this from the command line 
in the main directory without haveing to run python -m seg.utils.generate_plot_from_txt
"""
import os 
import argparse
import numpy as np
import pandas as pd 

def getAllDatasetStatisticsFromListDir(list_dirs: list, start_epoch=700, end_epoch=None):
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
            print(f'\t\tmax iou loss: {np.max(iou_data)}')
            print(f'\t\tmax dice loss: {np.max(dice_data)}')
            if end_epoch==None:
                print(f'\t\tmean iou loss between {start_epoch} and end epochs: {np.mean(iou_data[start_epoch:])}')
                print(f'\t\tmean dice loss between {start_epoch} and end epochs: {np.mean(dice_data[start_epoch:])}')
            else: 
                print(f'\t\tmean iou loss between {start_epoch} and {end_epoch} epochs: {np.mean(iou_data[start_epoch:end_epoch])}')
                print(f'\t\tmean dice loss between {start_epoch} and {end_epoch} epochs: {np.mean(dice_data[start_epoch:end_epoch])}')
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
    pd.DataFrame(ious_max).to_csv('ious_max.csv', header=list_dirs)
    pd.DataFrame(dice_max).to_csv('dice_max.csv', header=list_dirs)
    # pd.DataFrame(ious_max).to_csv('ious.csv', header=list_dirs)
    # pd.DataFrame(ious_max).to_csv('ious.csv', header=list_dirs)
    print(ious_max)
    print(ious_avg)
    print(dice_max)
    print(dice_avg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', type=str)
    parser.add_argument('--start_epoch', type=int)
    parser.add_argument('--end_epoch', type=int)
    args = parser.parse_args()
    results_list = list(args.data)

    getAllDatasetStatisticsFromListDir(
        list_dirs = list(args.data),
        start_epoch = args.start_epoch,
        end_epoch = args.end_epoch,
    )