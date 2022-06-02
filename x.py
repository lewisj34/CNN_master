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
        
        dice_combined = list()
        iou_combined = list()
        print(f'Epoch\t{tests[0]} \t\t{tests[1]} \t{tests[2]} \t{tests[3]} \t{tests[4]} \t\tAvg(dice, IoU)')
        for i in range(num_epochs):
            SET_dice_avg = np.mean((np.mean(dice_data_0[i]), np.mean(dice_data_1[i]), np.mean(dice_data_2[i]), np.mean(dice_data_3[i]), np.mean(dice_data_4[i])))
            SET_iou_avg = np.mean((np.mean(iou_data_0[i]), np.mean(iou_data_1[i]), np.mean(iou_data_2[i]), np.mean(iou_data_3[i]), np.mean(iou_data_4[i])))
            print("{}\t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f})".format(i, np.mean(dice_data_0[i]), np.mean(iou_data_0[i]), np.mean(dice_data_1[i]), np.mean(iou_data_1[i]), np.mean(dice_data_2[i]), np.mean(iou_data_2[i]), np.mean(dice_data_3[i]), np.mean(iou_data_3[i]), np.mean(dice_data_4[i]), np.mean(iou_data_4[i]), SET_dice_avg, SET_iou_avg))
            dice_combined.append(SET_dice_avg)
            iou_combined.append(SET_iou_avg)

        print('\n\n')
        print(f'#'*45, ' SUMMARY ', '#'*45)
        print('\n')
        print('Stat \t Epoch \t Max \t (dice, IoU)')
        print('IoU  \t {} \t {:.3f} \t ({:.3f}, {:.3f})'.format(np.argmax(iou_combined), np.max(iou_combined), dice_combined[np.argmax(iou_combined)], iou_combined[np.argmax(iou_combined)]))
        print('Dice \t {} \t {:.3f} \t ({:.3f}, {:.3f})'.format(np.argmax(dice_combined), np.max(dice_combined), dice_combined[np.argmax(dice_combined)], iou_combined[np.argmax(dice_combined)]))
        print('\n')
        print(f'#'*45, ' ####### ', '#'*45, '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', type=str)
    args = parser.parse_args()

    results_list = list(args.data)
    for i in range(len(results_list)):
        whipThroughTextFiles(results_list[i])

