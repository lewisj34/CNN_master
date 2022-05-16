import os 
import shutil

from regex import F 

def moveImgs(
    dir: str,
):
    """
    moves pics from imgs and ann dir in a test dataset from its current loc
    to the master train set 
    dir should be a test dataset dir with and images and annotations directory 
    """
    img_dir = dir + '/images/'
    ann_dir = dir + '/masks/'

    for file in os.listdir(img_dir):
        path = img_dir + file
        print(path)
        assert os.path.isfile(path), f'{path} is not a file. Check.'

    for file in os.listdir(ann_dir):
        path = ann_dir + file
        print(path)
        assert os.path.isfile(path), f'{path} is not a file. Check.'

    



def main():
    cvc300_dir = '/home/john/Documents/Datasets/master_polyp_mixed/TestDataset/CVC-300'
    cvcClinic_dir = '/home/john/Documents/Datasets/master_polyp_mixed/TestDataset/CVC-ClinicDB'
    cvcColon_dir = '/home/john/Documents/Datasets/master_polyp_mixed/TestDataset/CVC-ColonDB'
    ETIS_dir = '/home/john/Documents/Datasets/master_polyp_mixed/TestDataset/ETIS-LaribPolypDB'
    kvasir_dir = '/home/john/Documents/Datasets/master_polyp_mixed/TestDataset/Kvasir'

    dirs = [cvc300_dir, cvcClinic_dir, cvcColon_dir, ETIS_dir, kvasir_dir]

    # for i in range(len(dirs)):
    #     assert(os.path.isdir(dirs[i])), f'{dirs[i]} DNE, check.'

    #     moveImgs(dirs[i])

    moveImgs(cvc300_dir)

if __name__ == '__main__':
    main()