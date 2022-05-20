"""
basically copy pasted from generate_plot_from_txt except now we can just run this from the command line 
in the main directory without haveing to run python -m seg.utils.generate_plot_from_txt
"""
import os 
import argparse
import numpy as np
import pandas as pd 

def getAllDatasetStatisticsFromDir(dir, start_epoch, end_epoch):
    valid_iou_path = dir + '/valid_iou_file.txt'
    valid_dice_path = dir + '/valid_dice_file.txt'
    valid_loss_path = dir + '/valid_loss_file.txt'

    test_iou_path = dir + '/test_iou_file.txt'
    test_dice_path = dir + '/test_dice_file.txt'
    test_loss_path = dir + '/test_loss_file.txt'

    valid_iou_data = np.loadtxt(valid_iou_path)
    valid_dice_data = np.loadtxt(valid_dice_path)
    test_iou_data = np.loadtxt(test_iou_path)
    test_dice_data = np.loadtxt(test_dice_path)
    print('\n\n')
    print(f'Param \t\tTest Losses \tValid Losses \tEpoch Range')
    print("max(dice, iou) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t----------".format(np.max(test_dice_data), np.max(test_iou_data), np.max(valid_dice_data), np.max(valid_iou_data)))
    print("avg(dice, iou) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t[{}, {}]".format(np.mean(test_dice_data[start_epoch:end_epoch]), np.mean(test_iou_data[start_epoch:end_epoch]), np.mean(valid_dice_data[start_epoch:end_epoch]), np.mean(valid_iou_data[start_epoch:end_epoch]), start_epoch, end_epoch))
    print('\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--start_epoch', type=int, default=750)
    parser.add_argument('--end_epoch', type=int, default=800)
    args = parser.parse_args()
    results_list = list(args.data)

    getAllDatasetStatisticsFromDir(dir=args.data, start_epoch=args.start_epoch, end_epoch=args.end_epoch)