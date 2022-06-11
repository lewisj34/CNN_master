"""
Goal is to iterate through a directory and get create a new set of images
that is modified in terms of cropping and colour 
"""
import os
import socket
from time import sleep

import cv2
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from tqdm import tqdm


def process_dataset_augmentations(
    split_path='/home/john/Documents/Datasets/master_polyp/splits/train.txt',        
    save_aug_img_location = "/home/john/Documents/Datasets/master_polyp_t/AugmentedTrainDataset",     
    save_aug_ann_location = "/home/john/Documents/Datasets/master_polyp_t/AugmentedTrainDataset",
    save_grid_location = '/home/john/Documents/Datasets/master_polyp_t/grids'     ,
    img_dir='/home/john/Documents/Datasets/master_polyp_t/TrainDataset/image/',
    ann_dir='/home/john/Documents/Datasets/master_polyp_t/TrainDataset/mask/',
    unique_identifier='z_adj_',
):   
    '''
    Modded from process_dataset in generate_npy.py

    From a split text file, detailing dataset split, finds these files and 
    imports them with OpenCV and then dumps them to another folder with the 
    augmented verison of the image 

    Option to save original images in a dir. 
    Args:
        split_path: path to split.txt file
        save_location: directory to save augmented images (and potentially ogs)
        img_dir: location of img paths that split path points 
        ann_dir: location of ann paths that split path points
    '''
    # save_img_dir = f'{save_location}/images'
    # save_ann_dir = f'{save_location}/masks'
    # save_aug_img_dir = f'{save_location}/aug_images'
    # save_aug_ann_dir = f'{save_location}/aug_masks'
    # save_aug_ann_gray_dir = f'{save_location}/aug_masks_gray'
    # save_grid_dir = f'{save_location}/grids'

    # os.makedirs(save_img_dir, exist_ok=True)
    # os.makedirs(save_ann_dir, exist_ok=True)
    # os.makedirs(save_aug_img_dir, exist_ok=True)
    # os.makedirs(save_aug_ann_dir, exist_ok=True)
    # os.makedirs(save_aug_ann_gray_dir, exist_ok=True)
    # os.makedirs(save_grid_dir, exist_ok=True)

    assert split_path.endswith('.txt')
    size = num_lines = sum(1 for line in open(split_path))
    print(f"Dataset/file - {os.path.basename(split_path)}. Number in dataset: {size}")

    with open(split_path) as f:
        paths = f.readlines()
    paths = list(map(lambda s: s.strip(), paths))
    count = 0
    length = size

    # basic data structures holding image data - DONT NEED THESE IN THIS PROB
    # imgs = np.uint8(np.zeros([length, image_size[0], image_size[1], 3]))
    # masks = np.uint8(np.zeros([length, image_size[0], image_size[1]]))

    image_paths = paths.copy()
    mask_paths= paths.copy()

    for i in tqdm(range(len(paths))):
        sleep(0.0001)
        image_paths[i] = img_dir + image_paths[i]
        mask_paths[i] = ann_dir + mask_paths[i]

        img = cv2.imread(image_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_paths[i], 0)
        
        # save originals images -> take this out after we've debugged and compared 
        # imageio.imwrite(f'{save_img_dir}/{paths[i]}', img)
        # imageio.imwrite(f'{save_ann_dir}/{paths[i]}', mask)

        ret, mask = cv2.threshold(
                mask, 
                thresh=255.0 / 2, 
                maxval=1, 
                type=cv2.THRESH_BINARY
            )

        mask = np.expand_dims(mask, axis=2)

        segmap = SegmentationMapsOnImage(mask, shape=img.shape)

        # save augmented images 
        seq = iaa.Sequential([
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 80% of all images.
            iaa.Dropout([0.05, 0.2]),  
            iaa.Sometimes(
                0.8,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 50% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.5),
            # flip horizontally 50% prob
            iaa.Fliplr(0.5),
            # flip vertically 50% prob
            iaa.Flipud(0.5),
        ], random_order=True) # apply augmenters in random order
        
        images_aug = []
        segmaps_aug = []
        for _ in range(5):
            images_aug_i, segmaps_aug_i = seq(image=img, segmentation_maps=segmap)
            images_aug.append(images_aug_i)
            segmaps_aug.append(segmaps_aug_i)
            # print(f'images_aug.shape {images_aug_i.shape}')
            # print(f'segmaps_aug_i.shape {segmaps_aug_i.shape}')

        cells = []
        for image_aug, segmap_aug in zip(images_aug, segmaps_aug):
            cells.append(img)                                         # column 1
            cells.append(segmap.draw_on_image(img)[0])                # column 2
            cells.append(image_aug)                                     # column 3
            cells.append(segmap_aug.draw_on_image(image_aug)[0])        # column 4
            cells.append(segmap_aug.draw(size=image_aug.shape[:2])[0])  # column 5

        # # Convert cells to a grid image and save.
        if save_grid_location is not None:
            grid_image = ia.draw_grid(cells, cols=5)
            imageio.imwrite(f'{save_grid_location}/{unique_identifier}_grid_{paths[i]}', grid_image)

        imageio.imwrite(f'{save_aug_img_location}/{unique_identifier}{paths[i]}', image_aug)
        # imageio.imwrite(f'{save_aug_ann_dir}/z_adj_{paths[i]}', segmap_aug.draw(size=image_aug.shape[:2])[0])
        grayImage =  cv2.cvtColor(segmap_aug.draw(size=image_aug.shape[:2])[0], cv2.COLOR_RGB2GRAY)
        
        
        # print(f'max: {np.max(grayImage)}') # 92
        # print(f'min: {np.min(grayImage)}') # 0
        # print(f'unique_values values in numpy array: {np.unique(grayImage)}') # [0, 92] (therefore 2)

        # The thresholding below only works if 2 unique values. @ testing this was [0, 92].
        # if following test doesn't pass, see how many unique values we have here 
        # if it's more than 2, then we need to make a judgement call on what values 
        # get thresholded and the arg in cv2.threshold needs to be adjusted accordingly
        # if the number of unique values are bimodal (i.e. most are strongly 
        # towards ex: max 92, and the other are strongly towards min 0). Then I 
        # think we could just keep this as is.
        assert len(np.unique(grayImage)) == 2, \
            f'Number of unique values: {len(np.unique(grayImage))} != 2. See comment above.'
        
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, np.max(grayImage) / 2, 255, cv2.THRESH_BINARY)
        imageio.imwrite(f'{save_aug_ann_location}/{unique_identifier}{paths[i]}', blackAndWhiteImage)
    

import shutil
from tqdm import tqdm
from time import sleep 
def copyDir2Dir(
    source_folder,
    destination_folder,
):
    # fetch all files
    for file_name in tqdm(os.listdir(source_folder)):
        # construct full file path
        sleep(0.001)
        source = source_folder + file_name
        destination = destination_folder + file_name
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
        else:
            raise ValueError(f'file: {source} DNE.')

def copyFiles(
    src_imgs,
    src_anns,
    dst_imgs='/home/lewisj34_local/Dev/Datasets/master_polyp_mixed_with_augs/MergedDataset/images',
    dst_anns='/home/lewisj34_local/Dev/Datasets/master_polyp_mixed_with_augs/MergedDataset/annotations',
):
    assert os.path.isdir(src_imgs), f'src_imgs: {src_imgs} DNE.'
    assert os.path.isdir(src_anns), f'src_anns: {src_anns} DNE.'
    assert os.path.isdir(dst_imgs), f'src_imgs: {dst_imgs} DNE.'
    assert os.path.isdir(dst_anns), f'src_imgs: {dst_anns} DNE.'

    copyDir2Dir(src_imgs, dst_imgs)
    copyDir2Dir(src_anns, dst_anns)

def checkAllFilesFoundInDirectory(
    sub_dir,
    master_dir,
):
    """
    checks files in sub_dir, which should be a subset of master_dir, which 
    contains all files 
    """
    print(f'[{os.path.basename(sub_dir)} check]')
    sub_files = os.listdir(sub_dir)
    master_files = os.listdir(master_dir)

    result = all(file in sub_files for file in master_files)

    if result:
        print(f'\tSuccess: all files in sub_dir: {sub_dir} found in master_dir: {master_dir}')
    else:
        raise ValueError(f'\tError: check.')
    

def checkDirOfImagesExist(dir: str, printVals: bool = True) -> None:
    fileList = os.listdir(dir)
    print(f'Dir name: {dir}')
    print(f'Number of images in dir: {len(fileList)}') 
    if printVals:
        for i in range(len(fileList)): 
            print(f'\t{fileList[i]}')
    print(f'Dir name: {dir}')
    print(f'Number of images in dir: {len(fileList)}') 

def getNumberOfFilesinDir(dir: str) -> int:
    assert os.path.isdir(dir), f'dir: {dir} DNE'
    return len(os.listdir(dir))

def checkUniqueGetUniqueDir(parent_save_dir: str) -> str:
    """
    checks if `dir` is a unique dir, i.e. if the directory already exists. 
    If it doesn't, creates a new one. 
    """
    i = 1
    curr_save_dir = parent_save_dir + f"/{os.path.basename(parent_save_dir)}_{i}"
    if not os.path.isdir(curr_save_dir):
        print(f'curr_save_dir: {curr_save_dir} DNE. Creating.')
        os.mkdir(curr_save_dir)
        return curr_save_dir
    else:
        while os.path.isdir(curr_save_dir):
            print(f'curr_save_dir: {curr_save_dir} already exists...')
            i += 1
            curr_save_dir = parent_save_dir + f"/{os.path.basename(parent_save_dir)}_{i}"
            if not os.path.isdir(curr_save_dir):
                print(f'curr_save_dir: {curr_save_dir} DNE. Creating.')
                os.mkdir(curr_save_dir)
                return curr_save_dir
        return curr_save_dir

def createAndPopulateDirs(
    merged: bool,
    master_original_dir: str,
    parent_save_dir: str,
    split_paths: list, 
    src_img_dirs: list,
    src_ann_dirs: list,
    save_grids: bool = False,
):
    """
    Creates augmented images and then populate them into an almagomated dataset.
        @param merged: Use the merged dataset struture: {train, val, test}
        @param master_original_dir: where the original, unaugmented images are 
            going to be copied from (imgs/, anns/ if merged) (TrainDataset and 
            TestDataset if master)
        @param parent_save_dir: where all the images are going to be copied to
            both unaugmented and augmented
        @param split_paths: split paths that will be augmented
        @param src_img_dirs: img_dir holding the values present in @split_paths
        @param src_ann_dirs: ann_dir holding the values present in @split_paths
    """
    if merged:
        # defines the final locations of where the whole combined dataset will be saved 
        # directory SHOULD NOT exist
        if os.path.isdir(parent_save_dir):
            print(f'Master dir: {parent_save_dir} already exists.')
        else:
            print(f'No master directory yet. Master dir: {parent_save_dir}')
            os.mkdir(parent_save_dir)

        curr_save_dir = checkUniqueGetUniqueDir(parent_save_dir)
        

        dst_img_dir = curr_save_dir + '/images/'
        dst_ann_dir = curr_save_dir + '/annotations/'
        os.mkdir(dst_img_dir)
        os.mkdir(dst_ann_dir)
        
        src_img_dir = master_original_dir + "/images/"
        src_ann_dir = master_original_dir + "/masks/"

        assert getNumberOfFilesinDir(src_img_dir) == 2248, \
            f'Wrong number of files in img_dir: {src_img_dir}'
        assert getNumberOfFilesinDir(src_ann_dir) == 2248, \
            f'Wrong number of files in ann_dir: {src_ann_dir}'

        # now that we've established that the src_dir has the right number of files
        # copy them 
        print(f'Copying baseline images from merged dataset. Number in dataset: {len(os.listdir(src_img_dir))}')
        copyDir2Dir(src_img_dir, dst_img_dir)
        print(f'Copying baseline annotations from merged dataset. Number in dataset: {len(os.listdir(src_ann_dir))}')
        copyDir2Dir(src_img_dir, dst_ann_dir)

        # get augmented images 
        if save_grids:
            save_grid_location = curr_save_dir + '/grids'
            if not os.path.isdir(save_grid_location):
                os.mkdir(save_grid_location)
        else:
            save_grid_location = None

        assert len(split_paths) == len(src_img_dirs) == len(src_ann_dirs),\
            f'Num given: {len(split_paths), len(src_img_dirs), len(src_ann_dirs)}'
        num_datasets_to_add = len(split_paths)

        for i in range((num_datasets_to_add)):
            process_dataset_augmentations(
                split_path = split_paths[i],
                save_aug_img_location=dst_img_dir,
                save_aug_ann_location=dst_ann_dir,
                save_grid_location=save_grid_location,
                img_dir=src_img_dirs[i],
                ann_dir=src_ann_dirs[i],
                unique_identifier=f'{os.path.splitext(os.path.basename(split_paths[i]))[0]}_{i}_'
            )


        exit(1)
    else: # master set, so two locations: {TrainDataset and TestDataset}, we want to copy every `augmented` image to TrainDataset
        # defines the final locations of where the whole combined dataset will be saved 
        # directory SHOULD NOT exist
        if os.path.isdir(parent_save_dir):
            print(f'Master dir: {parent_save_dir} already exists.')
        else:
            print(f'No master directory yet. Master dir: {parent_save_dir}')
            os.mkdir(parent_save_dir)
        
        # assume master_original_dir is: /home/john/Documents/Datasets/master_polyp/TrainDataset
        # assume parent_save_dir is: /home/john/Documents/Datasets/arbitrary_master_polyp_comb
        curr_save_dir = checkUniqueGetUniqueDir(parent_save_dir)

        dst_img_dir = curr_save_dir + '/TrainDataset/image/'
        dst_ann_dir = curr_save_dir + '/TrainDataset/mask/'
        os.makedirs(dst_img_dir, exist_ok=False)
        os.makedirs(dst_ann_dir, exist_ok=False)
        # os.mkdir(dst_img_dir)
        # os.mkdir(dst_ann_dir)

        src_img_dir = master_original_dir + "/image/"
        src_ann_dir = master_original_dir + "/mask/"

        assert getNumberOfFilesinDir(src_img_dir) == 1450, \
            f'Wrong number of files in img_dir: {src_img_dir}'
        assert getNumberOfFilesinDir(src_ann_dir) == 1450, \
            f'Wrong number of files in ann_dir: {src_ann_dir}'

        # now that we've established that the src_dir has the right number of files
        # copy them 
        print(f'Copying baseline images from merged dataset. Number in dataset: {len(os.listdir(src_img_dir))}')
        copyDir2Dir(src_img_dir, dst_img_dir)
        print(f'Copying baseline annotations from merged dataset. Number in dataset: {len(os.listdir(src_ann_dir))}')
        copyDir2Dir(src_img_dir, dst_ann_dir)

        # need to copy TestDataset too 
        src_test_dataset_dir = f'{master_original_dir}/../TestDataset'
        dst_test_dataset_dir = curr_save_dir + '/TestDataset/'
        # os.mkdir(dst_test_dataset_dir)
        shutil.copytree(src_test_dataset_dir, dst_test_dataset_dir)

        # get augmented images 
        if save_grids:
            save_grid_location = curr_save_dir + '/grids'
            if not os.path.isdir(save_grid_location):
                os.mkdir(save_grid_location)
        else:
            save_grid_location = None

        assert len(split_paths) == len(src_img_dirs) == len(src_ann_dirs),\
            f'Num given: {len(split_paths), len(src_img_dirs), len(src_ann_dirs)}'
        num_datasets_to_add = len(split_paths)

        for i in range((num_datasets_to_add)):
            process_dataset_augmentations(
                split_path = split_paths[i],
                save_aug_img_location=dst_img_dir,
                save_aug_ann_location=dst_ann_dir,
                save_grid_location=save_grid_location,
                img_dir=src_img_dirs[i],
                ann_dir=src_ann_dirs[i],
                unique_identifier=f'{os.path.splitext(os.path.basename(split_paths[i]))[0]}_{i}_'
            )
            print(f'Num in img_dir, {dst_img_dir}: {len(dst_img_dir)}')
            print(f'Num in ann_dir, {dst_ann_dir}: {len(dst_ann_dir)}')


if __name__ == '__main__':
    merged = False
    if socket.gethostname() == "john-linux":
        # imgs that aren't going to be augmented, but used in training 
        merge_src_dir = "/home/john/Documents/Datasets/master_polyp_mixed_final/MergedDataset"
        
        # where to save the images and where the new superdataset will be  
        merge_parent_save_dir = '/home/john/Documents/Datasets/master_merged_with_augs_out'

        # imgs that aren't going to be augmented, but used in training 
        master_train_src_dir = "/home/john/Documents/Datasets/master_polyp/TrainDataset"

        # where to save the new superdataset of train images 
        master_parent_save_dir = '/home/john/Documents/Datasets/arbitrary_master_polyp_comb'

        split_paths = [
            '/home/john/Documents/Datasets/master_polyp/splits/CVC_300_test.txt',
            '/home/john/Documents/Datasets/master_polyp/splits/CVC_ColonDB_test.txt',  
            '/home/john/Documents/Datasets/master_polyp/splits/ETIS_test.txt',  
            '/home/john/Documents/Datasets/master_polyp/splits/CVC_ClinicDB_test.txt',  
            '/home/john/Documents/Datasets/master_polyp/splits/Kvasir_test.txt', 
        ]
        src_img_dirs = [
            '/home/john/Documents/Datasets/master_polyp/TestDataset/CVC-300/images/',
            '/home/john/Documents/Datasets/master_polyp/TestDataset/CVC-ColonDB/images/',
            '/home/john/Documents/Datasets/master_polyp/TestDataset/ETIS-LaribPolypDB/images/',
            '/home/john/Documents/Datasets/master_polyp/TestDataset/CVC-ClinicDB/images/',
            '/home/john/Documents/Datasets/master_polyp/TestDataset/Kvasir/images/',
        ]
        src_ann_dirs = [
            '/home/john/Documents/Datasets/master_polyp/TestDataset/CVC-300/masks/',
            '/home/john/Documents/Datasets/master_polyp/TestDataset/CVC-ColonDB/masks/',
            '/home/john/Documents/Datasets/master_polyp/TestDataset/ETIS-LaribPolypDB/masks/',
            '/home/john/Documents/Datasets/master_polyp/TestDataset/CVC-ClinicDB/masks/',
            '/home/john/Documents/Datasets/master_polyp/TestDataset/Kvasir/masks/',
        ]

    elif socket.gethostname() == f'ce-yc-dlearn{5, 6}.eng.umanitoba.ca':\
        # imgs that aren't going to be augmented, but used in training 
        merge_src_dir = '/home/lewisj34_local/Dev/Datasets/master_polyp_mixed_final/MergedDataset'
        
        # where to save the images and where the new superdataset will be  
        merge_parent_save_dir = '/home/lewisj34_local/Dev/Datasets/master_merged_with_augs_reboot_out' 

        # imgs that aren't going to be augmented, but used in training 
        master_train_src_dir = '/home/lewisj34_local/Dev/Datasets/master_polyp/TrainDataset'

        # where to save the new superdataset of train images 
        master_parent_save_dir = '/home/lewisj34_local/Dev/Datasets/arbitrary_master_polyp_comb'

        split_paths = [
            '/home/lewisj34_local/Dev/Datasets/master_polyp/splits/CVC_300_test.txt',
            '/home/lewisj34_local/Dev/Datasets/master_polyp/splits/CVC_ColonDB_test.txt',  
            '/home/lewisj34_local/Dev/Datasets/master_polyp/splits/ETIS_test.txt',  
            '/home/lewisj34_local/Dev/Datasets/master_polyp/splits/CVC_ClinicDB_test.txt',  
            '/home/lewisj34_local/Dev/Datasets/master_polyp/splits/Kvasir_test.txt', 
        ]
        src_img_dirs = [
            '/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/CVC-300/images/',
            '/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/CVC-ColonDB/images/',
            '/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/ETIS-LaribPolypDB/images/',
            '/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/CVC-ClinicDB/images/',
            '/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/Kvasir/images/',
        ]
        src_ann_dirs = [
            '/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/CVC-300/masks/',
            '/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/CVC-ColonDB/masks/',
            '/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/ETIS-LaribPolypDB/masks/',
            '/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/CVC-ClinicDB/masks/',
            '/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/Kvasir/masks/',
        ]

    if merged == True:
        createAndPopulateDirs(
            merged=True,
            master_original_dir=merge_src_dir,
            parent_save_dir=merge_parent_save_dir,
            split_paths=split_paths,
            src_img_dirs=src_img_dirs,
            src_ann_dirs=src_ann_dirs,
        )
    else:
        createAndPopulateDirs(
            merged=False,
            master_original_dir=master_train_src_dir,
            parent_save_dir=master_parent_save_dir,
            split_paths=split_paths,
            src_img_dirs=src_img_dirs,
            src_ann_dirs=src_ann_dirs,
        )



    # elif socket.gethostname() == f'ce-yc-dlearn{5, 6}.eng.umanitoba.ca':
    #     merge_src_dir = "/home/lewisj34_local/Dev/Datasets/master_polyp_mixed_final/MergedDataset"
    #     parent_save_dir = '/home/lewisj34_local/Dev/Datasets/master_merged_with_augs_out'
    # else:
    #     raise ValueError(f'Device: {socket.gethostname()} unknown. Check.')

    # createAndPopulateDirs(
    #     merged=True, 
    #     master_original_dir=merge_src_dir,
    #     parent_save_dir=parent_save_dir
    # )
    exit(1)
    # school computer

    # where the images are going 
    mergedDataset = True
    dst_imgs = '/home/lewisj34_local/Dev/Datasets/master_polyp_aug_spec/images/'
    dst_anns = '/home/lewisj34_local/Dev/Datasets/master_polyp_aug_spec/annotations/'


    # home computer 

    if socket.gethostname() == "john-linux":
        print(f'home comp in use.')
        # code below needs checking...
        # process_dataset_augmentations(
        #     split_path='/home/john/Documents/Datasets/master_polyp/splits/CVC_300_test.txt',        
        #     save_location = "/home/john/Documents/Datasets/master_polyp/AugmentedTestDatasetsV2/CVC_300",     
        #     img_dir='/home/john/Documents/Datasets/master_polyp/TestDataset/CVC-300/images/',
        #     ann_dir='/home/john/Documents/Datasets/master_polyp/TestDataset/CVC-300/masks/',
        # )
        # process_dataset_augmentations(
        #     split_path='/home/john/Documents/Datasets/master_polyp/splits/ETIS_test.txt',        
        #     save_location = "/home/john/Documents/Datasets/master_polyp/AugmentedTestDatasets/ETIS",     
        #     img_dir='/home/john/Documents/Datasets/master_polyp/TestDataset/ETIS-LaribPolypDB/images/',
        #     ann_dir='/home/john/Documents/Datasets/master_polyp/TestDataset/ETIS-LaribPolypDB/masks/',
        # )
        # process_dataset_augmentations(
        #     split_path='/home/john/Documents/Datasets/master_polyp/splits/CVC_ColonDB_test.txt',        
        #     save_location = "/home/john/Documents/Datasets/master_polyp/AugmentedTestDatasets/CVC_ColonDB",     
        #     img_dir='/home/john/Documents/Datasets/master_polyp/TestDataset/CVC-ColonDB/images/',
        #     ann_dir='/home/john/Documents/Datasets/master_polyp/TestDataset/CVC-ColonDB/masks/',
        # )
        # process_dataset_augmentations(
        #     split_path='/home/john/Documents/Datasets/master_polyp/splits/CVC_ClinicDB_test.txt',        
        #     save_location = "/home/john/Documents/Datasets/master_polyp/AugmentedTestDatasets/CVC_ClinicDB",     
        #     img_dir='/home/john/Documents/Datasets/master_polyp/TestDataset/CVC-ClinicDB/images/',
        #     ann_dir='/home/john/Documents/Datasets/master_polyp/TestDataset/CVC-ClinicDB/masks/',
        # )
        # process_dataset_augmentations(
        #     split_path='/home/john/Documents/Datasets/master_polyp/splits/Kvasir_test.txt',        
        #     save_location = "/home/john/Documents/Datasets/master_polyp/AugmentedTestDatasets/Kvasir",     
        #     img_dir='/home/john/Documents/Datasets/master_polyp/TestDataset/Kvasir/images/',
        #     ann_dir='/home/john/Documents/Datasets/master_polyp/TestDataset/Kvasir/masks/',
        # )
    elif socket.gethostname() == "ce-yc-dlearn6.eng.umanitoba.ca" or socket.gethostname() == "ce-yc-dlearn5.eng.umanitoba.ca":
        print(f'dlearn6 or dlearn 5 in use.')
        
        # # copy all files 
        # dst_imgs = '/home/lewisj34_local/Dev/Datasets/master_polyp_mixed_with_augsV2/MergedDataset/images/'
        # dst_anns = '/home/lewisj34_local/Dev/Datasets/master_polyp_mixed_with_augsV2/MergedDataset/annotations/'

        # kvasir_imgs = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasetsV2/Kvasir/aug_images/'
        # kvasir_ans = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasetsV2/Kvasir/aug_masks_gray/'

        # CVC_300_imgs = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasetsV2/CVC_300/aug_images/'
        # CVC_300_ans = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasetsV2/CVC_300/aug_masks_gray/'

        # CVC_ClinicDB_imgs = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasetsV2/CVC_ClinicDB/aug_images/'
        # CVC_ClinicDB_ans = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasetsV2/CVC_ClinicDB/aug_masks_gray/'

        # CVC_ColonDB_imgs = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasetsV2/CVC_ColonDB/aug_images/'
        # CVC_ColonDB_ans = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasetsV2/CVC_ColonDB/aug_masks_gray/'

        # ETIS_imgs = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasetsV2/ETIS/aug_images/'
        # ETIS_ans = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasetsV2/ETIS/aug_masks_gray/'

        # process_dataset_augmentations(
        #     split_path='/home/lewisj34_local/Dev/Datasets/master_polyp/splits/CVC_300_test.txt',        
        #     save_location = "/home/lewisj34_local/Dev/Datasets/master_polyp_aug_spec/AugmentedTestSetsV2/CVC_300",     
        #     img_dir='/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/CVC-300/images/',
        #     ann_dir='/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/CVC-300/masks/',
        #     unique_identifier='z_adj_CVC300_'
        # )
        # copyFiles(
        #     src_imgs=CVC_300_imgs,
        #     src_anns=CVC_300_ans,
        #     dst_imgs=dst_imgs,
        #     dst_anns=dst_anns,
        # )
        # process_dataset_augmentations(
        #     split_path='/home/lewisj34_local/Dev/Datasets/master_polyp/splits/ETIS_test.txt',        
        #     save_location = "/home/lewisj34_local/Dev/Datasets/master_polyp_aug_spec/AugmentedTestSetsV2/ETIS",     
        #     img_dir='/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/ETIS-LaribPolypDB/images/',
        #     ann_dir='/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/ETIS-LaribPolypDB/masks/',
        #     unique_identifier='z_adj_ETIS_'
        # )
        # copyFiles(
        #     src_imgs=ETIS_imgs,
        #     src_anns=ETIS_ans,
        #     dst_imgs=dst_imgs,
        #     dst_anns=dst_anns,
        # )
        # process_dataset_augmentations(
        #     split_path='/home/lewisj34_local/Dev/Datasets/master_polyp/splits/CVC_ColonDB_test.txt',        
        #     save_location = "/home/lewisj34_local/Dev/Datasets/master_polyp_aug_spec/AugmentedTestSetsV2/CVC_ColonDB",     
        #     img_dir='/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/CVC-ColonDB/images/',
        #     ann_dir='/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/CVC-ColonDB/masks/',
        #     unique_identifier='z_adj_ColonDB_'
        # )
        # copyFiles(
        #     src_imgs=CVC_ColonDB_imgs,
        #     src_anns=CVC_ColonDB_ans,
        #     dst_imgs=dst_imgs,
        #     dst_anns=dst_anns,
        # )
        
        # process_dataset_augmentations(
        #     split_path='/home/lewisj34_local/Dev/Datasets/master_polyp/splits/CVC_ClinicDB_test.txt',        
        #     save_location = "/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasetsV2/CVC_ClinicDB",     
        #     img_dir='/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/CVC-ClinicDB/images/',
        #     ann_dir='/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/CVC-ClinicDB/masks/',
        # )

        # process_dataset_augmentations(
        #     split_path='/home/lewisj34_local/Dev/Datasets/master_polyp/splits/Kvasir_test.txt',        
        #     save_location = "/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasetsV2/Kvasir",     
        #     img_dir='/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/Kvasir/images/',
        #     ann_dir='/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/Kvasir/masks/',
        # )
        
        



    else:
        print(f'IP not input correctly.')
        raise ValueError(f'IP: {socket.gethostname()} not validated.')

    # # copy all files 
    dst_imgs = '/home/lewisj34_local/Dev/Datasets/master_polyp_aug_spec/images/'
    dst_anns = '/home/lewisj34_local/Dev/Datasets/master_polyp_aug_spec/annotations/'

    # kvasir_imgs = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasets/Kvasir/aug_images/'
    # kvasir_ans = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasets/Kvasir/aug_masks_gray/'

    CVC_300_imgs = '/home/lewisj34_local/Dev/Datasets/master_polyp_aug_spec/AugmentedTestSetsV2/CVC_300/aug_images/'
    CVC_300_ans = '/home/lewisj34_local/Dev/Datasets/master_polyp_aug_spec/AugmentedTestSetsV2/CVC_300/aug_masks_gray/'

    # CVC_ClinicDB_imgs = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasets/CVC_ClinicDB/aug_images/'
    # CVC_ClinicDB_ans = '/home/lewisj34_local/Dev/Datasets/master_polyp/AugmentedTestDatasets/CVC_ClinicDB/aug_masks_gray/'

    CVC_ColonDB_imgs = '/home/lewisj34_local/Dev/Datasets/master_polyp_aug_spec/AugmentedTestSetsV2/CVC_ColonDB/aug_images/'
    CVC_ColonDB_ans = '/home/lewisj34_local/Dev/Datasets/master_polyp_aug_spec/AugmentedTestSetsV2/CVC_ColonDB/aug_masks_gray/'

    ETIS_imgs = '/home/lewisj34_local/Dev/Datasets/master_polyp_aug_spec/AugmentedTestSetsV2/ETIS/aug_images/'
    ETIS_ans = '/home/lewisj34_local/Dev/Datasets/master_polyp_aug_spec/AugmentedTestSetsV2/ETIS/aug_masks_gray/'

    # copyFiles(
    #     src_imgs=kvasir_imgs,
    #     src_anns=kvasir_ans,
    #     dst_imgs=dst_imgs,
    #     dst_anns=dst_anns,
    # )
    copyFiles(
        src_imgs=CVC_300_imgs,
        src_anns=CVC_300_ans,
        dst_imgs=dst_imgs,
        dst_anns=dst_anns,
    )
    # copyFiles(
    #     src_imgs=CVC_ClinicDB_imgs,
    #     src_anns=CVC_ClinicDB_ans,
    #     dst_imgs=dst_imgs,
    #     dst_anns=dst_anns,
    # )
    copyFiles(
        src_imgs=CVC_ColonDB_imgs,
        src_anns=CVC_ColonDB_ans,
        dst_imgs=dst_imgs,
        dst_anns=dst_anns,
    )
    copyFiles(
        src_imgs=ETIS_imgs,
        src_anns=ETIS_ans,
        dst_imgs=dst_imgs,
        dst_anns=dst_anns,
    )
    # checkAllFilesFoundInDirectory(
    #     sub_dir='/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/Kvasir/images/',
    #     master_dir='/home/lewisj34_local/Dev/Datasets/master_polyp_mixed_final/MergedDataset/images/',
    # )
    # checkAllFilesFoundInDirectory(
    #     sub_dir='/home/lewisj34_local/Dev/Datasets/master_polyp/TestDataset/Kvasir/masks/',
    #     master_dir='/home/lewisj34_local/Dev/Datasets/master_polyp_mixed_final/MergedDataset/annotations/',
    # )