import argparse
import numpy as np 

def whipThroughTextFiles(dir: str):
        tests = ['Kvasir', 'CVC_ClinicDB', 'CVC_ColonDB', 'CVC_300', 'ETIS']
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

        num_epochs = len(iou_data_0)
        
        print(f'{tests[0]} \t\t{tests[1]} \t{tests[2]} \t{tests[3]} \t{tests[4]} \t\tAvg(dice, IoU)')
        for i in range(num_epochs):
            SET_dice_avg = np.mean((np.mean(dice_data_0[i]), np.mean(dice_data_1[i]), np.mean(dice_data_2[i]), np.mean(dice_data_3[i]), np.mean(dice_data_4[i])))
            SET_iou_avg = np.mean((np.mean(iou_data_0[i]), np.mean(iou_data_1[i]), np.mean(iou_data_2[i]), np.mean(iou_data_3[i]), np.mean(iou_data_4[i])))
            print("({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f})".format(np.mean(dice_data_0[i]), np.mean(iou_data_0[i]), np.mean(dice_data_1[i]), np.mean(iou_data_1[i]), np.mean(dice_data_2[i]), np.mean(iou_data_2[i]), np.mean(dice_data_3[i]), np.mean(iou_data_3[i]), np.mean(dice_data_4[i]), np.mean(iou_data_4[i]), SET_dice_avg, SET_iou_avg))

        # SET_dice_avg = np.mean((np.mean(dice_data_0[start_epoch:end_epoch]), np.mean(dice_data_1[start_epoch:end_epoch]), np.mean(dice_data_2[start_epoch:end_epoch]), np.mean(dice_data_3[start_epoch:end_epoch]), np.mean(dice_data_4[start_epoch:end_epoch])))
        # SET_iou_avg = np.mean((np.mean(iou_data_0[start_epoch:end_epoch]), np.mean(iou_data_1[start_epoch:end_epoch]), np.mean(iou_data_2[start_epoch:end_epoch]), np.mean(iou_data_3[start_epoch:end_epoch]), np.mean(iou_data_4[start_epoch:end_epoch])))

        # print('\n\n')
        # print(f'Param \t\t{tests[0]} \t\t{tests[1]} \t{tests[2]} \t{tests[3]} \t{tests[4]} \t\tAvg(dice, IoU) \tEpoch Range')
        # print("max(dice, iou) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t---------- \t----------".format(np.max(dice_data_0), np.max(iou_data_0), np.max(dice_data_1), np.max(iou_data_1), np.max(dice_data_2), np.max(iou_data_2), np.max(dice_data_3), np.max(iou_data_3), np.max(dice_data_4), np.max(iou_data_4)))
        # print("avg(dice, iou) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t[{}, {}]".format(np.mean(dice_data_0[start_epoch:end_epoch]), np.mean(iou_data_0[start_epoch:end_epoch]), np.mean(dice_data_1[start_epoch:end_epoch]), np.mean(iou_data_1[start_epoch:end_epoch]), np.mean(dice_data_2[start_epoch:end_epoch]), np.mean(iou_data_2[start_epoch:end_epoch]), np.mean(dice_data_3[start_epoch:end_epoch]), np.mean(iou_data_3[start_epoch:end_epoch]), np.mean(dice_data_4[start_epoch:end_epoch]), np.mean(iou_data_4[start_epoch:end_epoch]), SET_dice_avg, SET_iou_avg, start_epoch, end_epoch))
        # print('\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', type=str)
    args = parser.parse_args()

    print(f'', args.data)
    results_list = list(args.data)
    for i in range(len(results_list)):
        whipThroughTextFiles(results_list[i])

