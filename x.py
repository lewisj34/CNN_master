import os 
import shutil
import click

from regex import F 

def moveImgs(
    origin_dir: str,
    target_dir: str,
    prefix: str,
    trainDataset: bool = False,
):
    """
    moves pics from imgs and ann dir in a test dataset from its current loc
    to the master train set 
    dir should be a test dataset dir with and images and annotations directory 
    """
    origin_img_dir = origin_dir + '/images/'
    origin_ann_dir = origin_dir + '/masks/'

    target_img_dir = target_dir + '/images/'
    target_ann_dir = target_dir + '/masks/'

    if trainDataset: # just gotta put this logic in because TrainDataset has image and mask as the dir name 
        origin_img_dir = origin_dir + '/image/'
        origin_ann_dir = origin_dir + '/mask/'

    for file in os.listdir(origin_img_dir):
        path = origin_img_dir + file
        # print(path)
        assert os.path.isfile(path), f'{path} is not a file. Check.'

        origin = path
        target_name = prefix + os.path.basename(path)
        target_path = target_img_dir + target_name

        print(f'target: {target_path}')
        shutil.copy(origin, target_path)



    for file in os.listdir(origin_ann_dir):
        path = origin_ann_dir + file
        assert os.path.isfile(path), f'{path} is not a file. Check.'

        origin = path
        target_name = prefix + os.path.basename(path)
        target_path = target_ann_dir + target_name

        print(f'target: {target_path}')
        shutil.copy(origin, target_path)

    

@click.command(help='')
@click.option('--origin_dir', type=str, default='/home/lewisj34_local/Dev/Datasets/master_polyp_mixed')
@click.option('--target_dir', type=str, default='/home/lewisj34_local/Dev/Datasets/master_polyp_mixed_final')
def main(
    origin_dir,
    target_dir
):
    if not os.path.isdir(target_dir):
        print(f'target_dir: {target_dir} DNE. Creating...')
        os.mkdir(target_dir)
        if not os.path.isdir(target_dir + '/TestDataset/'):
            os.mkdir(target_dir + '/TestDataset/')
            os.mkdir(target_dir + '/TestDataset/images')
            os.mkdir(target_dir + '/TestDataset/masks')
        if not os.path.isdir(target_dir + '/TrainDataset/'):
            os.mkdir(target_dir + '/TrainDataset/')
            os.mkdir(target_dir + '/TrainDataset/images')
            os.mkdir(target_dir + '/TrainDataset/masks')
        if not os.path.isdir(target_dir + '/MergedDataset/'):
            os.mkdir(target_dir + '/MergedDataset/')
            os.mkdir(target_dir + '/MergedDataset/images')
            os.mkdir(target_dir + '/MergedDataset/masks')
    else:
        if not os.path.isdir(target_dir + '/TestDataset/'):
            os.mkdir(target_dir + '/TestDataset/')
            os.mkdir(target_dir + '/TestDataset/images')
            os.mkdir(target_dir + '/TestDataset/masks')
        if not os.path.isdir(target_dir + '/TrainDataset/'):
            os.mkdir(target_dir + '/TrainDataset/')
            os.mkdir(target_dir + '/TrainDataset/images')
            os.mkdir(target_dir + '/TrainDataset/masks')
        if not os.path.isdir(target_dir + '/MergedDataset/'):
            os.mkdir(target_dir + '/MergedDataset/')
            os.mkdir(target_dir + '/MergedDataset/images')
            os.mkdir(target_dir + '/MergedDataset/masks')
        print(f'target_dir: {target_dir} and derivatives already exists.')

    # train dataset
    train_dir = origin_dir + '/TrainDataset/'

    # test datasets
    cvc300_dir = origin_dir + '/TestDataset/CVC-300'
    cvcClinic_dir = origin_dir + '/TestDataset/CVC-ClinicDB'
    cvcColon_dir = origin_dir + '/TestDataset/CVC-ColonDB'
    ETIS_dir = origin_dir + '/TestDataset/ETIS-LaribPolypDB'
    kvasir_dir = origin_dir + '/TestDataset/Kvasir'

    # home computer stuff 
    # cvc300_dir = '/home/john/Documents/Datasets/master_polyp_mixed/TestDataset/CVC-300'
    # cvcClinic_dir = '/home/john/Documents/Datasets/master_polyp_mixed/TestDataset/CVC-ClinicDB'
    # cvcColon_dir = '/home/john/Documents/Datasets/master_polyp_mixed/TestDataset/CVC-ColonDB'
    # ETIS_dir = '/home/john/Documents/Datasets/master_polyp_mixed/TestDataset/ETIS-LaribPolypDB'
    # kvasir_dir = '/home/john/Documents/Datasets/master_polyp_mixed/TestDataset/Kvasir'

    moveImgs(
        cvc300_dir,
        target_dir + '/MergedDataset/',
        prefix = 'cvc_300_'
    )
    moveImgs(
        cvcClinic_dir,
        target_dir + '/MergedDataset/',
        prefix = 'cvc_ClinicDB_'
    )
    moveImgs(
        cvcColon_dir,
        target_dir + '/MergedDataset/',
        prefix = 'cvc_ColonDB_'
    )
    moveImgs(
        ETIS_dir,
        target_dir + '/MergedDataset/',
        prefix = 'ETIS_'
    )
    moveImgs(
        kvasir_dir,
        target_dir + '/MergedDataset/',
        prefix = 'kvasir_'
    )
    moveImgs(
        train_dir,
        target_dir + '/MergedDataset/',
        prefix = 'train_',
        trainDataset=True,
    )

if __name__ == '__main__':
    main()