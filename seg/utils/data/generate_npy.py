"""
NEW VERSION of preprocess.py (renamed and moved from main directory to utils)! 

Takes in a directory of images with the file structure: 
    /images
    /annotations
And then moves all the images into a train/test/val split list org by generating
the following text split files as follows: 
    train.txt
    val.txt
    test.txt
Once it's done that, or checked to see that those files already exists. It 
converts those lists and generates .npy binary files containing the image data. 
Which in turn was taken from OpenCV matrices after iterating through the .txt
split files. Which are stored in location `dataset` + `save_dir`
    /dataset/save_dir/data_train.npy
    /dataset/save_dir/data_test.npy
    /dataset/save_dir/data_valid.npy
    /dataset/save_dir/mask_train.npy
    /dataset/save_dir/mask_test.npy
    /dataset/save_dir/mask_valid.npy
Therefore NOTE: What you need to know is that the incoming file structure, is 
    not split. The splits are generated with teh following code. 
"""

import os 
import sys
import cv2
from cv2 import IMREAD_GRAYSCALE
import matplotlib.pyplot as plt 
import numpy as np
import yaml
import pandas as pd 
import shutil 

from tqdm import tqdm
from time import sleep 
from pathlib import Path

from seg.utils.err_codes import crop_err3
from .utils.generate_csv import generate_csv

def list_overlap(a, b):
    return bool(set(a) & set(b))

def generate_split_txt_files_master(
    parent_dir='/home/john/Documents/Datasets/master_polyp'
):
    '''
    Used explicitly for the master polyp dataset. 

    Generates .txt files for train, valid, and test sets by creating a csv for 
    each set, and then converting that csv to a .txt file for each and then 
    populating the split_dir with each of those .txt files. 
    Args:
        @parent_dir: the master dataset containing the master set of images.
    File sturcture:
        master/
        ├─ TrainDataset/
        │  ├─ image/
        │  ├─ mask/
        ├─ TestDataset/
        │  ├─ CVC-300/
        │  ├─ CVC-ClinicDB/
        │  ├─ CVC-ColonDB/
        │  ├─ ETIS-LaribPolypDB/
        │  ├─ Kvasir/
    '''
    train_img_dir = parent_dir + '/TrainDataset/image/'
    train_ann_dir = parent_dir + '/TrainDataset/mask/'
    
    test_CVC_300_img_dir = parent_dir + '/TestDataset/CVC-300/images/'
    test_CVC_300_ann_dir = parent_dir + '/TestDataset/CVC-300/masks/'

    test_CVC_ClinicDB_img_dir = parent_dir + '/TestDataset/CVC-ClinicDB/images/'
    test_CVC_ClinicDB_ann_dir = parent_dir + '/TestDataset/CVC-ClinicDB/masks/'

    test_CVC_ColonDB_img_dir = parent_dir + '/TestDataset/CVC-ColonDB/images/'
    test_CVC_ColonDB_ann_dir = parent_dir + '/TestDataset/CVC-ColonDB/masks/'

    test_ETIS_img_dir = parent_dir + '/TestDataset/ETIS-LaribPolypDB/images/'
    test_ETIS_ann_dir = parent_dir + '/TestDataset/ETIS-LaribPolypDB/masks/'

    test_kvasir_img_dir = parent_dir + '/TestDataset/Kvasir/images/'
    test_kvasir_ann_dir = parent_dir + '/TestDataset/Kvasir/masks/'
    
    split_dir = parent_dir + '/splits/'
    train_split_path = split_dir + 'train.txt'
    valid_split_path = split_dir + 'valid.txt'
    test_split_CVC_300_path = split_dir + 'CVC_300_test.txt'
    test_split_CVC_ClinicDB_path = split_dir + 'CVC_ClinicDB_test.txt'
    test_split_CVC_ColonDB_path = split_dir + 'CVC_ColonDB_test.txt'
    test_split_ETIS_path = split_dir + 'ETIS_test.txt'
    test_split_Kvasir_path = split_dir + 'Kvasir_test.txt'



    
    if not os.path.isfile(train_split_path) \
        or not os.path.isfile(valid_split_path) \
        or not os.path.isfile(test_split_CVC_300_path) \
        or not os.path.isfile(test_split_CVC_ClinicDB_path) \
        or not os.path.isfile(test_split_CVC_ColonDB_path) \
        or not os.path.isfile(test_split_ETIS_path) \
        or not os.path.isfile(test_split_Kvasir_path):
        print("Splits don't exist. Generating new splits.")

        # split 'TrainDataset' into train and val by generating csv of file locs
        csv_location = generate_csv(train_img_dir, train_ann_dir, 'traindata.csv', 'seg/data/master/csvs/')
        print(f'csv_location: {csv_location}')
        metadata_df = pd.read_csv(csv_location)

        # shuffle 
        metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
        # metadata_df = metadata_df.sample(frac=1); print(f'Shuffling again.')
        # metadata_df = metadata_df.sample(frac=1); print(f'Shuffling again.')
        # metadata_df = metadata_df.sample(frac=1); print(f'Shuffling again.')
        # metadata_df = metadata_df.sample(frac=1); print(f'Shuffling again.')


        # we're not going to split the train data for the master dataset, like 
        # we have before. we're just going to sample 20-30% of the train dataset
        # WITH replacement. so the train dataset will automatically contain the 
        # val data and we won't need to readd it again later. in other words
        # the validation data is just going to be train data that the moodel 
        # has already seen 
        # and keep the train dataset whole 
        valid_df = metadata_df.sample(frac=0.1, random_state=42)
        train_df = metadata_df

        valid_list = valid_df["image_ids"].tolist()
        train_list = train_df["image_ids"].tolist()
        
        # for test sets: same thing, generate a csv of file locations, convert
        # to data frame, then create lists 
        tCVC300_loc = generate_csv(test_CVC_300_img_dir, test_CVC_300_ann_dir, 'tCVC300.csv', 'seg/data/master/csvs/')
        tCVC_ClinicDB_loc = generate_csv(test_CVC_ClinicDB_img_dir, test_CVC_ClinicDB_ann_dir, 'tCVC_ClinicDB.csv', 'seg/data/master/csvs/')
        tCVC_ColonDB_loc = generate_csv(test_CVC_ColonDB_img_dir, test_CVC_ColonDB_ann_dir, 'tCVC_ColonDB.csv', 'seg/data/master/csvs/')
        tETIS_loc = generate_csv(test_ETIS_img_dir, test_ETIS_ann_dir, 'tETIS.csv', 'seg/data/master/csvs/')
        tKvasir_loc = generate_csv(test_kvasir_img_dir, test_kvasir_ann_dir, 'tKvasir.csv', 'seg/data/master/csvs/')
        
        tCVC300_df = pd.read_csv(tCVC300_loc)
        tCVCClinicDB_df = pd.read_csv(tCVC_ClinicDB_loc)
        tCVCColonDB_df = pd.read_csv(tCVC_ColonDB_loc)
        tETIS_df = pd.read_csv(tETIS_loc)
        tKvasir_df = pd.read_csv(tKvasir_loc)

        # shuffle the test sets 
        tCVC300_df = tCVC300_df.sample(frac=1).reset_index(drop=True)
        tCVCClinicDB_df = tCVCClinicDB_df.sample(frac=1).reset_index(drop=True)
        tCVCColonDB_df = tCVCColonDB_df.sample(frac=1).reset_index(drop=True)
        tETIS_df = tETIS_df.sample(frac=1).reset_index(drop=True)
        tKvasir_df = tKvasir_df.sample(frac=1).reset_index(drop=True)

        tCVC300_list = tCVC300_df["image_ids"].tolist()
        tCVCClinicDB_list = tCVCClinicDB_df["image_ids"].tolist()
        tCVCColonDB_list = tCVCColonDB_df["image_ids"].tolist()
        tETIS_list = tETIS_df["image_ids"].tolist()
        tKvasir_list = tKvasir_df["image_ids"].tolist()

        print(f'len(tCVC300_list): {len(tCVC300_list)}')
        print(f'len(tCVCClinicDB_list): {len(tCVCClinicDB_list)}')
        print(f'len(tCVCColonDB_list): {len(tCVCColonDB_list)}')
        print(f'len(tETIS_list): {len(tETIS_list)}')
        print(f'len(tKvasir_list): {len(tKvasir_list)}')

        # generate split files/dir in split_dir
        if os.path.isdir(split_dir):
            print("splits/ exists. Deleting.\n")
            shutil.rmtree(split_dir)
        else:
            print("splits/ DNE. Creating new directory.\n")

        os.makedirs(split_dir,\
            exist_ok=False)

        print("Writing split files.\n")
        with open(train_split_path, "w") as outfile:
            outfile.write("\n".join(train_list))
        with open(valid_split_path, "w") as outfile:
            outfile.write("\n".join(valid_list))
        with open(test_split_CVC_300_path, "w") as outfile:
            outfile.write("\n".join(tCVC300_list))
        with open(test_split_CVC_ClinicDB_path, "w") as outfile:
            outfile.write("\n".join(tCVCClinicDB_list))
        with open(test_split_CVC_ColonDB_path, "w") as outfile:
            outfile.write("\n".join(tCVCColonDB_list))
        with open(test_split_ETIS_path, "w") as outfile:
            outfile.write("\n".join(tETIS_list))
        with open(test_split_Kvasir_path, "w") as outfile:
            outfile.write("\n".join(tKvasir_list))
        print("Splits written to: ", split_dir)
    else:
        print("Splits for master set exist already at location:", split_dir)  

def generate_split_txt_files(
    parent_dir='/home/john/Documents/Datasets/kvasir_merged'
):
    ''' 
    Generates .txt files for train, valid, and test sets by creating a csv for 
    each (train, valid, ...) and then converting that csv to a .txt file for 
    each and then populating the split_dir with each of those .txt files. 
    Args:
        @parent_dir: the master dataset of file structure:
            parent_dir/
                /images
                /annotations
                /splits (will be generated if it doesn't exist already)
    '''

    img_dir = parent_dir + '/images'
    ann_dir = parent_dir + '/annotations'
    split_dir = parent_dir + "/splits/"
    train_split_path = split_dir + 'train.txt'
    valid_split_path = split_dir + 'valid.txt'
    test_split_path = split_dir + 'test.txt'

    assert os.path.isdir(parent_dir), f'parent_dir: {parent_dir} DNE.'
    assert os.path.isdir(img_dir), f'img_dir: {img_dir} DNE.'
    assert os.path.isdir(ann_dir), f'ann_dir: {ann_dir} DNE.'

    # check if {train, ..., test}.txt files exist, if they don't, create them. 
    if not os.path.isfile(train_split_path) or not os.path.isfile(\
        valid_split_path) or not os.path.isfile(test_split_path):
        print("Splits don't exist. Generating new splits.")
        
        # generate the metadata into a csv 
        csv_location = generate_csv(img_dir, ann_dir)
        metadata_df = pd.read_csv(csv_location)

        # shuffle 
        metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

        # perform 80/20 split for train / val 
        valid_df = metadata_df.sample(frac=0.2, random_state=42)
        train_df = metadata_df.drop(valid_df.index)

        # now perform a 50/50 split for test/ val 
        test_df = valid_df.sample(frac=0.5, random_state=42)
        valid_df = valid_df.drop(test_df.index)

        valid_list = valid_df["image_ids"].tolist()
        train_list = train_df["image_ids"].tolist()
        test_list = test_df["image_ids"].tolist()

        # check to see if overlap between train and valid sets 
        if list_overlap(train_list, valid_list):
            print("ERROR: overlap in train and valid list. Debug code.")
            sys.exit(1) 
        elif list_overlap(test_list, valid_list):
            print("ERROR: overlap in test and valid list. Debug code.")
            sys.exit(1) 
        elif list_overlap(train_list, test_list):
            print("ERROR: overlap in train and test list. Debug code.")
            sys.exit(1) 
        else:
            print("No overlap between files in train and valid sets.")
            print("Proceeding with creating split files...\n")

        # generate split files/dir in split_dir
        if os.path.isdir(split_dir):
            print("splits/ exists. Deleting.\n")
            shutil.rmtree(split_dir)
        else:
            print("splits/ DNE. Creating new directory.\n")

        os.makedirs(split_dir,\
            exist_ok=False)
        
        print("Writing split files.\n")
        with open(train_split_path, "w") as outfile:
            outfile.write("\n".join(train_list))
        with open(valid_split_path, "w") as outfile:
            outfile.write("\n".join(valid_list))
        with open(test_split_path, "w") as outfile:
            outfile.write("\n".join(test_list))

        print("Splits written to: ", split_dir)
    else:
        print("Splits exist already at location:", split_dir)


def data_files_exist(save_dir):
    if os.path.isfile(save_dir + "/params.yaml") and \
        os.path.isfile(save_dir + "/data_test.npy") and \
        os.path.isfile(save_dir + "/data_train.npy") and \
        os.path.isfile(save_dir + "/data_valid.npy") and \
        os.path.isfile(save_dir + "/mask_test.npy") and \
        os.path.isfile(save_dir + "/mask_train.npy") and \
        os.path.isfile(save_dir + "/mask_valid.npy"):
            return True
    else:
        return False

def data_files_exist_master(save_dir):
    if os.path.isfile(save_dir + "/params.yaml") and \
        os.path.isfile(save_dir + "/data_CVC_300_test.npy") and \
        os.path.isfile(save_dir + "/data_CVC_ClinicDB_test.npy") and \
        os.path.isfile(save_dir + "/data_CVC_ColonDB_test.npy") and \
        os.path.isfile(save_dir + "/data_ETIS_test.npy") and \
        os.path.isfile(save_dir + "/data_Kvasir_test.npy") and \
        os.path.isfile(save_dir + "/data_train.npy") and \
        os.path.isfile(save_dir + "/data_valid.npy") and \
        os.path.isfile(save_dir + "/mask_CVC_300_test.npy") and \
        os.path.isfile(save_dir + "/mask_CVC_ClinicDB_test.npy") and \
        os.path.isfile(save_dir + "/mask_CVC_ColonDB_test.npy") and \
        os.path.isfile(save_dir + "/mask_ETIS_test.npy") and \
        os.path.isfile(save_dir + "/mask_Kvasir_test.npy") and \
        os.path.isfile(save_dir + "/mask_train.npy") and \
        os.path.isfile(save_dir + "/mask_valid.npy"):
            return True
    else:
        return False

def one_hot_encode(image_path, num_classes=2):
    """
    Takes in an input image path, turns into an OpenCV matrix... 
    """
    # read in mask as grayscale to get `num_classes` unique values 
    mask = cv2.imread(image_path, IMREAD_GRAYSCALE)

    # For whatever reason unique values are read in like this: 
    #   [  0   1   2   3   4   5   6   7   8 248 249 250 251 252 253 254 255]
    # i.e. we get a clear binary but the range is over [0, 255], 
    # so we need to adjust the matrix so that we only have a binary value for 
    # each pixel 
    ret, mask = cv2.threshold(
        mask, 
        thresh=255.0 / 2, 
        maxval=1, 
        type=cv2.THRESH_BINARY
    )

    one_hot = np.zeros([mask.shape[0], mask.shape[1], num_classes])

    for i, unique_value in enumerate(np.unique(mask)):
        one_hot[:, :, i][mask == unique_value] = 1
        print(f'i, unique_val, mask.shape', i, unique_value, one_hot.shape)

    return one_hot

def process_dataset_two_classes(
    split_path,        
    save_location = "seg/data/random_dataset",     
    img_dir = "/home/john/Documents/Datasets/kvasir_merged/images/",            
    ann_dir = "/home/john/Documents/Datasets/kvasir_merged/annotations/", 
    resize_size = (None, None),
    crop_size = (None, None),
    image_size = (None, None),
):
    """
    See description for process_dataset, pretty much the same except we import
    with two classes for the mask instead of just one 
    """
    
    # this is already called in the main function, split_and_convert_to_npyV2
    # but calling it again anyways 
    ###########################################################################
    # determine if we need to call resize or crop funcs 
    if resize_size[0] is not None or resize_size[1] is not None:
        resize_image = True
    else:
        resize_image = False
    if crop_size[0] is not None or crop_size[1] is not None:
        crop_image = True
    else:
        crop_image = False

    # check to make sure everythings been assigned a variable correctly 
    if (resize_image and crop_image) or (crop_image and not resize_image): 
        assert image_size == crop_size, \
            f'image_size: {image_size}, crop_size: {crop_size}'
    if resize_image and not crop_image:
        assert image_size == resize_size, \
            f'image_size: {image_size}, resize_size: {resize_size}'

    if resize_size == crop_size == image_size == (None, None):
        raise ValueError(f'Not suported yet.') #  where we just import the 
        # images as is and wait to crop and resize at Dataloader / runtime.
    ###########################################################################


    assert split_path.endswith('.txt')
    size = num_lines = sum(1 for line in open(split_path))
    print(f"Length of {os.path.basename(split_path)}: {size}")
    print(f"Processing {os.path.basename(split_path)} to .npy files")

    with open(split_path) as f:
        paths = f.readlines()
    paths = list(map(lambda s: s.strip(), paths))
    count = 0
    length = size

    # basic data structures holding image data
    imgs = np.uint8(np.zeros([length, image_size[0], image_size[1], 3]))
    masks = np.uint8(np.zeros([length, image_size[0], image_size[1], 2]))
    if resize_image:
        images_before_cropping = np.uint8(np.zeros([length, resize_size[0], resize_size[1], 3]))
        masks_before_cropping = np.uint8(np.zeros([length, resize_size[0], resize_size[1], 2]))
    if crop_image:
        images_after_cropping = np.uint8(np.zeros([length, crop_size[0], crop_size[1], 3]))
        masks_after_cropping = np.uint8(np.zeros([length, crop_size[0], crop_size[1], 2]))

    image_paths = paths.copy()
    mask_paths= paths.copy()

    for i in tqdm(range(len(paths))):
        sleep(0.0001)
        image_paths[i] = img_dir + image_paths[i]
        mask_paths[i] = ann_dir + mask_paths[i]

        img = cv2.imread(image_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(mask_paths[i], 0)
        mask = one_hot_encode(
            image_path = mask_paths[i],
            num_classes = 2,
        )
        
        if resize_image:
            img = cv2.resize(img, (resize_size[0], resize_size[1]))
            mask = cv2.resize(mask, (resize_size[0], resize_size[1]))

        if crop_image:
            assert crop_size[0] is not None and crop_size[1] is not None, \
                crop_err3(crop_size[0], crop_size[1])

            # to be deleted
            if resize_image:
                images_before_cropping[count] = img # to be deletd 
                masks_before_cropping[count] = mask # to be deletd   

            # KEEP THIS!!!!!!!!!!!
            img, mask = get_random_crop(img, mask, crop_size[0], crop_size[1])

            # to be deleted
            images_after_cropping[count] = img # to be deletd 
            masks_after_cropping[count] = mask # to be deletd

        imgs[count] = img
        masks[count] = mask
        print(f'masks.shape: {masks.shape}')
        count += 1 
    set_name = os.path.splitext(os.path.basename(split_path))[0]

    # modification to operate with base code, if not in generates: 
    # FileNotFoundError: [Errno 2] No such file or directory: 'seg/data//data_val.npy'
    # if set_name == "valid":
    #     set_name = "val"
    np.save('{}/data_{}.npy'.format(save_location, set_name), imgs)
    np.save('{}/mask_{}.npy'.format(save_location, set_name), masks)  


def process_dataset(
    split_path,        
    save_location = "seg/data/random_dataset",     
    img_dir = "/home/john/Documents/Datasets/kvasir_merged/images/",            
    ann_dir = "/home/john/Documents/Datasets/kvasir_merged/annotations/", 
    resize_size = (None, None),
    crop_size = (None, None),
    image_size = (None, None),
):   
    '''
    From split text files, detailing dataset split, finds these files and 
    imports them with OpenCV and then dumps them to .npy binary files in 
    save_location from corresponding img_dir and ann_dir with corr. resized 
    height & width, along with the corr. cropped height and width. 
    The image size should correspond to either the resized size (if no crop) or
    the crop size if the crop is specified. TODO: Put this last part in, as a 
    check that we've imported this stuff properly. 
    Args:
        split_path: path to split.txt file, EITHER: {train, valid, test}.txt
        save_location: directory to save .npy data files 
        img_dir: location of img paths that split path points 
        ann_dir: location of ann paths that split path points
        resize_size: the size to resize the input to
        crop_size: the size to crop the input to
        image_size: should be either the resize size or the crop size (crop size
            if the crop size is specified, resize size if no crop size is spec.)     
    '''
    
    # this is already called in the main function, split_and_convert_to_npyV2
    # but calling it again anyways 
    ###########################################################################
    # determine if we need to call resize or crop funcs 
    if resize_size[0] is not None or resize_size[1] is not None:
        resize_image = True
    else:
        resize_image = False
    if crop_size[0] is not None or crop_size[1] is not None:
        crop_image = True
    else:
        crop_image = False

    # check to make sure everythings been assigned a variable correctly 
    if (resize_image and crop_image) or (crop_image and not resize_image): 
        assert image_size == crop_size, \
            f'image_size: {image_size}, crop_size: {crop_size}'
    if resize_image and not crop_image:
        assert image_size == resize_size, \
            f'image_size: {image_size}, resize_size: {resize_size}'

    if resize_size == crop_size == image_size == (None, None):
        raise ValueError(f'Not suported yet.') #  where we just import the 
        # images as is and wait to crop and resize at Dataloader / runtime.
    ###########################################################################


    assert split_path.endswith('.txt')
    size = num_lines = sum(1 for line in open(split_path))
    print(f"Length of {os.path.basename(split_path)}: {size}")
    print(f"Processing {os.path.basename(split_path)} to .npy files")

    with open(split_path) as f:
        paths = f.readlines()
    paths = list(map(lambda s: s.strip(), paths))
    count = 0
    length = size

    # basic data structures holding image data
    imgs = np.uint8(np.zeros([length, image_size[0], image_size[1], 3]))
    masks = np.uint8(np.zeros([length, image_size[0], image_size[1]]))
    if resize_image:
        images_before_cropping = np.uint8(np.zeros([length, resize_size[0], resize_size[1], 3]))
        masks_before_cropping = np.uint8(np.zeros([length, resize_size[0], resize_size[1]]))
    if crop_image:
        images_after_cropping = np.uint8(np.zeros([length, crop_size[0], crop_size[1], 3]))
        masks_after_cropping = np.uint8(np.zeros([length, crop_size[0], crop_size[1]]))

    image_paths = paths.copy()
    mask_paths= paths.copy()

    for i in tqdm(range(len(paths))):
        sleep(0.0001)
        image_paths[i] = img_dir + image_paths[i]
        mask_paths[i] = ann_dir + mask_paths[i]

        img = cv2.imread(image_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_paths[i], 0)
        
        if resize_image:
            img = cv2.resize(img, (resize_size[0], resize_size[1]))
            mask = cv2.resize(mask, (resize_size[0], resize_size[1]))

        if crop_image:
            assert crop_size[0] is not None and crop_size[1] is not None, \
                crop_err3(crop_size[0], crop_size[1])

            # to be deleted
            if resize_image:
                images_before_cropping[count] = img # to be deletd 
                masks_before_cropping[count] = mask # to be deletd   

            # KEEP THIS!!!!!!!!!!!
            img, mask = get_random_crop(img, mask, crop_size[0], crop_size[1])

            # to be deleted
            images_after_cropping[count] = img # to be deletd 
            masks_after_cropping[count] = mask # to be deletd

        imgs[count] = img
        masks[count] = mask
        count += 1 
    set_name = os.path.splitext(os.path.basename(split_path))[0]

    # modification to operate with base code, if not in generates: 
    # FileNotFoundError: [Errno 2] No such file or directory: 'seg/data//data_val.npy'
    # if set_name == "valid":
    #     set_name = "val"
    np.save('{}/data_{}.npy'.format(save_location, set_name), imgs)
    np.save('{}/mask_{}.npy'.format(save_location, set_name), masks)  



def get_random_crop(image, mask, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    cropped_image = image[y: y + crop_height, x: x + crop_width]
    cropped_mask = mask[y: y + crop_height, x: x + crop_width]

    return cropped_image, cropped_mask

def split_and_convert_to_npyV2(
    dataset = 'kvasir',
    save_dir = 'seg/data',
    resize_size = (256, 256),
    crop_size = (None, None),
    image_size = (256, 256),
    reimport_data = False, 
    num_classes = 1,
    parent_dir = None,
):
    """
    @dataset: type of dataset
    @save_dir: location to save the npy files
    @resize_size: the size to resize the input image to 
    @crop_size: what to crop the input image or the resized image at 
    @image_size: final image size, f(resize_size, crop_size)
    @reimport_data: whether to reimport the data or not
    @num_classes: import masks with chan dim == num_classes and therefore: 
        num_chans = 2 or 1, 'new' = num_classes = 2, 'old' = num_classes = 1
    @parent_dir: the directory that holds the image data w/ images/annotations
    """

    if parent_dir is None:
        print(f'Note: dataset location not given, using default provided in code.')

    assert num_classes == 1 or num_classes == 2, \
        'new: num_classes = 2, old: num_classes = 1, must be either new or old'

    if num_classes == 2:
        print(f'Processing all datasets with masks having TWO channel dimensions')
    else:
        print(f'Processing all datasets with masks having just ONE channel dimension')

    model_params = {
        'dataset': dataset, 
        'save_dir': save_dir, 
        'resize_size': list(resize_size),
        'crop_size': list(crop_size),
        'image_size': list(image_size), 
        'process_dataset_version': num_classes
    }

    # determine if we need to call resize or crop funcs 
    if resize_size[0] is not None or resize_size[1] is not None:
        resize_image = True
    else:
        resize_image = False
    if crop_size[0] is not None or crop_size[1] is not None:
        crop_image = True
    else:
        crop_image = False

    # check to make sure everythings been assigned a variable correctly 
    if (resize_image and crop_image) or (crop_image and not resize_image): 
        assert image_size == crop_size, \
            f'image_size: {image_size}, crop_size: {crop_size}'
    if resize_image and not crop_image:
        assert image_size == resize_size, \
            f'image_size: {image_size}, resize_size: {resize_size}'

    
    if os.path.isdir(save_dir):
        print("Directory:", save_dir + " already exists.")
    else:
        print("Directory:", save_dir + " does not exist. Creating.")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    if dataset != 'master':
        if data_files_exist(save_dir):
            print("All data files exist")
            file_params = yaml.load(open(Path(save_dir + "/params.yaml")), 
                Loader=yaml.FullLoader)
            if file_params == model_params and not reimport_data:
                print("All files exist and model params are the same. \
    Not reimporting data. Exiting split_and_convert_to_npy(...)")
                return None
            elif file_params != model_params:
                print("All file exists but model params changed, must rewrite.")
                with open(save_dir + "/params.yaml", 'w') as file:
                    params = yaml.dump(model_params, file)
        else:
            print("One or more data files do not exist. Must create or rewrite.")
            with open(save_dir + "/params.yaml", 'w') as file:
                params = yaml.dump(model_params, file)
    else:
        if data_files_exist_master(save_dir):
            print("All data files exist")
            file_params = yaml.load(open(Path(save_dir + "/params.yaml")), 
                Loader=yaml.FullLoader)
            if file_params == model_params and not reimport_data:
                print("All files exist and model params are the same. \
    Not reimporting data. Exiting split_and_convert_to_npy(...)")
                return None
            elif file_params != model_params:
                print("All file exists but model params changed, must rewrite.")
                with open(save_dir + "/params.yaml", 'w') as file:
                    params = yaml.dump(model_params, file)
        else:
            print("One or more data files do not exist. Must create or rewrite.")
            with open(save_dir + "/params.yaml", 'w') as file:
                params = yaml.dump(model_params, file)

    if dataset == 'kvasir':
        if parent_dir is None:
            parent_dir = '/home/john/Documents/Datasets/kvasir_merged'
        assert os.path.isdir(parent_dir), f'directory: {parent_dir} doesnt exist adjust above, or input --dataset_file_location in commandline arg'
    elif dataset == 'CVC_ClinicDB':
        if parent_dir is None:    
            parent_dir = '/home/john/Documents/Datasets/CVC-ClinicDB/PNG'
        assert os.path.isdir(parent_dir), f'directory: {parent_dir} doesnt exist adjust above, or input --dataset_file_location in commandline arg'
    elif dataset == 'ETIS':
        if parent_dir is None:
            parent_dir = '/home/john/Documents/Datasets/ETIS'
        assert os.path.isdir(parent_dir), f'directory: {parent_dir} doesnt exist adjust above, or input --dataset_file_location in commandline arg'
    elif dataset == 'CVC_ColonDB':
        if parent_dir is None:
            parent_dir = '/home/john/Documents/Datasets/CVC-ColonDB'
        assert os.path.isdir(parent_dir), f'directory: {parent_dir} doesnt exist adjust above, or input --dataset_file_location in commandline arg'
    elif dataset == 'master':
        if parent_dir is None:
            parent_dir = '/home/john/Documents/Datasets/master_polyp'
        assert os.path.isdir(parent_dir), f'directory: {parent_dir} doesnt exist adjust above, or input --dataset_file_location in commandline arg'
    elif dataset == 'master_merged': 
        if parent_dir is None:
            parent_dir = '/home/lewisj34_local/Dev/Datasets/master_polyp_mixed_final/'
        assert os.path.isdir(parent_dir), f'directory: {parent_dir} doesnt exist adjust above, or input --dataset_file_location in commandline arg'
    else:
        raise NotImplementedError(f'Dataset: {dataset} not implemented.')    
        
    print(f'dataset location: {parent_dir}')
    
    if dataset != 'master': # i.e. any normal dataset
        TRAIN_SPLIT_PATH = parent_dir + "/splits/" + "train.txt"
        VALID_SPLIT_PATH = parent_dir + "/splits/" + "valid.txt"
        TEST_SPLIT_PATH = parent_dir + "/splits/" + "test.txt"    
        IMG_DIR = parent_dir + '/images/'
        ANN_DIR = parent_dir + '/annotations/'
        
        # generate split mapping file paths {train, valid, test}.txt
        generate_split_txt_files(parent_dir=parent_dir)

        if num_classes == 1:
            # # organize train data into {data, mask}.npy files from splits
            process_dataset(
                split_path = TRAIN_SPLIT_PATH, 
                save_location = save_dir, 
                img_dir = IMG_DIR,
                ann_dir = ANN_DIR,
                resize_size = resize_size,
                crop_size = crop_size,
                image_size = image_size,
            )

            # # organize valid data into {data, mask}.npy files from splits
            process_dataset(
                split_path = VALID_SPLIT_PATH, 
                save_location = save_dir, 
                img_dir = IMG_DIR,
                ann_dir = ANN_DIR,
                resize_size = resize_size,
                crop_size = crop_size,
                image_size = image_size,
            )

            # # organize test data into {data, mask}.npy files from splits 
            process_dataset(
                split_path = TEST_SPLIT_PATH, 
                save_location = save_dir, 
                img_dir = IMG_DIR,
                ann_dir = ANN_DIR,
                resize_size = resize_size,
                crop_size = crop_size,
                image_size = image_size,
            )
        elif num_classes == 2:
            # # organize train data into {data, mask}.npy files from splits
            process_dataset_two_classes(
                split_path = TRAIN_SPLIT_PATH, 
                save_location = save_dir, 
                img_dir = IMG_DIR,
                ann_dir = ANN_DIR,
                resize_size = resize_size,
                crop_size = crop_size,
                image_size = image_size,
            )

            # # organize valid data into {data, mask}.npy files from splits
            process_dataset_two_classes(
                split_path = VALID_SPLIT_PATH, 
                save_location = save_dir, 
                img_dir = IMG_DIR,
                ann_dir = ANN_DIR,
                resize_size = resize_size,
                crop_size = crop_size,
                image_size = image_size,
            )

            # # organize test data into {data, mask}.npy files from splits 
            process_dataset_two_classes(
                split_path = TEST_SPLIT_PATH, 
                save_location = save_dir, 
                img_dir = IMG_DIR,
                ann_dir = ANN_DIR,
                resize_size = resize_size,
                crop_size = crop_size,
                image_size = image_size,
            )

            print(f'exiting..')
            exit(1)
    else:
        # so we still want split files for train test and valid... but we're 
        # going to keep these the way they are 
        '''
        So we still want split files for train, test, and valid sets but the 
        master dataset structure is different. It follows: 
        master/
        ├─ TrainDataset/
        │  ├─ images/
        │  ├─ masks/
        ├─ TestDataset/
        │  ├─ CVC-300/
        │  ├─ CVC-ClinicDB/
        │  ├─ CVC-ColonDB/
        │  ├─ ETIS-LaribPolypDB/
        │  ├─ Kvasir/

        We're just going to assume that every iteration of this program that is
        run will have this folder structure somewhere on the system so all we 
        will specify as an argument is parent_dir which holds the location of
        the above file structure
        '''
        TRAIN_SPLIT_PATH = parent_dir + "/splits/" + "train.txt"
        VALID_SPLIT_PATH = parent_dir + "/splits/" + "valid.txt"
        TEST_CVC300_SPLIT_PATH = parent_dir + "/splits/" + "CVC_300_test.txt"  
        TEST_CLINICDB_SPLIT_PATH = parent_dir + "/splits/" + "CVC_ClinicDB_test.txt"  
        TEST_COLONDB_SPLIT_PATH = parent_dir + "/splits/" + "CVC_ColonDB_test.txt"  
        TEST_ETIS_SPLIT_PATH = parent_dir + "/splits/" + "ETIS_test.txt"  
        TEST_KVASIR_SPLIT_PATH = parent_dir + "/splits/" + "Kvasir_test.txt"  

        generate_split_txt_files_master(parent_dir)  
        IMG_DIR = parent_dir + '/TrainDataset/image/'
        ANN_DIR = parent_dir + '/TrainDataset/mask/'
        CVC300_IMG_DIR = parent_dir + '/TestDataset/CVC-300/images/'
        CVC300_ANN_DIR = parent_dir + '/TestDataset/CVC-300/masks/'
        CLINC_DB_IMG_DIR = parent_dir + '/TestDataset/CVC-ClinicDB/images/'
        CLINC_DB_ANN_DIR = parent_dir + '/TestDataset/CVC-ClinicDB/masks/'
        COLON_DB_IMG_DIR = parent_dir + '/TestDataset/CVC-ColonDB/images/'
        COLON_DB_ANN_DIR = parent_dir + '/TestDataset/CVC-ColonDB/masks/'
        ETIS_IMG_DIR = parent_dir + '/TestDataset/ETIS-LaribPolypDB/images/'
        ETIS_ANN_DIR = parent_dir + '/TestDataset/ETIS-LaribPolypDB/masks/'
        KVASIR_IMG_DIR = parent_dir + '/TestDataset/Kvasir/images/'
        KVASIR_ANN_DIR = parent_dir + '/TestDataset/Kvasir/masks/'

        if num_classes != 1: 
            raise ValueError(f'num_classes == 1 for master dataset')

        process_dataset(
            split_path = TRAIN_SPLIT_PATH, 
            save_location = save_dir, 
            img_dir = IMG_DIR,
            ann_dir = ANN_DIR,
            resize_size = resize_size,
            crop_size = crop_size,
            image_size = image_size,
        )
        process_dataset(
            split_path = VALID_SPLIT_PATH, 
            save_location = save_dir, 
            img_dir = IMG_DIR,
            ann_dir = ANN_DIR,
            resize_size = resize_size,
            crop_size = crop_size,
            image_size = image_size,
        )
        process_dataset(
            split_path = TEST_CVC300_SPLIT_PATH, 
            save_location = save_dir, 
            img_dir = CVC300_IMG_DIR,
            ann_dir = CVC300_ANN_DIR,
            resize_size = resize_size,
            crop_size = crop_size,
            image_size = image_size,
        )
        process_dataset(
            split_path = TEST_CLINICDB_SPLIT_PATH, 
            save_location = save_dir, 
            img_dir = CLINC_DB_IMG_DIR,
            ann_dir = CLINC_DB_ANN_DIR,
            resize_size = resize_size,
            crop_size = crop_size,
            image_size = image_size,
        )
        process_dataset(
            split_path = TEST_COLONDB_SPLIT_PATH, 
            save_location = save_dir, 
            img_dir = COLON_DB_IMG_DIR,
            ann_dir = COLON_DB_ANN_DIR,
            resize_size = resize_size,
            crop_size = crop_size,
            image_size = image_size,
        )
        process_dataset(
            split_path = TEST_ETIS_SPLIT_PATH, 
            save_location = save_dir, 
            img_dir = ETIS_IMG_DIR,
            ann_dir = ETIS_ANN_DIR,
            resize_size = resize_size,
            crop_size = crop_size,
            image_size = image_size,
        )
        process_dataset(
            split_path = TEST_KVASIR_SPLIT_PATH, 
            save_location = save_dir, 
            img_dir = KVASIR_IMG_DIR,
            ann_dir = KVASIR_ANN_DIR,
            resize_size = resize_size,
            crop_size = crop_size,
            image_size = image_size,
        )
    ########################################################################
    # DEFUNCT BELOW THIS LINE. JUST WAS GETTING US WHAT WE NEEDED IN TERMS OF 
    # CROP AND STUFF SO THAT WE COULD FINALIZE process_dataset(...)
    ########################################################################

    # get a test batch and a list of paths - this will be done in the process_dataset above
    # but what were trying to do is modify that right now 
    # data_dir = '/home/john/Documents/Datasets/kvasir_small'
    # img_dir = data_dir + '/images/'
    # ann_dir = data_dir + '/annotations/'

    # image_path_list = list()
    # ann_path_list = list()
    # for file in os.listdir(img_dir):
    #     filename = os.fsdecode(file)
    #     image_path_list.append(img_dir + filename)
    # for file in os.listdir(ann_dir):
    #     filename = os.fsdecode(file)
    #     ann_path_list.append(ann_dir + filename)

    # manually_input_observed_size_of_image_dir = 10
    # assert len(image_path_list) == len(ann_path_list) == manually_input_observed_size_of_image_dir
    # length = len(image_path_list)


    # imgs = np.uint8(np.zeros([length, image_size[0], image_size[1], 3]))
    # masks = np.uint8(np.zeros([length, image_size[0], image_size[1]]))



    # count = 0 
  
    # # to be deleted 
    # visualize_images = True
    # if resize_image:
    #     images_before_cropping = np.uint8(np.zeros([length, resize_size[0], resize_size[1], 3]))
    #     masks_before_cropping = np.uint8(np.zeros([length, resize_size[0], resize_size[1]]))
    # if crop_image:
    #     images_after_cropping = np.uint8(np.zeros([length, crop_size[0], crop_size[1], 3]))
    #     masks_after_cropping = np.uint8(np.zeros([length, crop_size[0], crop_size[1]]))


    # for i in tqdm(range(manually_input_observed_size_of_image_dir)):
    #     sleep(0.0001)

    #     img = cv2.imread(image_path_list[i])
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #     # to be deleted
    #     img_before_resize_and_crop = img

    #     if resize_image:
    #         img = cv2.resize(img, (resize_size[0], resize_size[1]))
    #         image_after_resize = img

    #     mask = cv2.imread(ann_path_list[i], 0)

    #     # to be deleted
    #     mask_before_resize_and_crop = mask

    #     if resize_image:
    #         mask = cv2.resize(mask, (resize_size[0], resize_size[1]))
    #         mask_after_resize = mask

    #     if crop_image:
    #         assert crop_size[0] is not None and crop_size[1] is not None, \
    #             crop_err3(crop_size[0], crop_size[1])

    #         # to be deleted
    #         if resize_image:
    #             images_before_cropping[count] = img # to be deletd 
    #             masks_before_cropping[count] = mask # to be deletd   

    #         # KEEP THIS!!!!!!!!!!!
    #         img, mask = get_random_crop(img, mask, crop_size[0], crop_size[1])

    #         # to be deleted
    #         images_after_cropping[count] = img # to be deletd 
    #         masks_after_cropping[count] = mask # to be deletd
        
    #     # to be deleted
    #     if visualize_images:
    #         if resize_image and crop_image:
    #             showSingleImageAndMask(
    #                 img_before_resize_and_crop,
    #                 mask_before_resize_and_crop,
    #                 images_before_cropping[count],
    #                 images_after_cropping[count],
    #                 masks_before_cropping[count],
    #                 masks_after_cropping[count],
    #             )
    #         if resize_image and not crop_image:
    #             showImageJUSTResizedORJUSTCropped(
    #                 img_before_resize_and_crop,
    #                 mask_before_resize_and_crop,
    #                 image_after_resize,
    #                 mask_after_resize,
    #             )
    #         if crop_image and not resize_image:
    #             showImageJUSTResizedORJUSTCropped(
    #                 img_before_resize_and_crop,
    #                 mask_before_resize_and_crop,
    #                 images_after_cropping[count],
    #                 masks_after_cropping[count]
    #             )
    #     imgs[count] = img
    #     masks[count] = mask
    #     count += 1 



    # set_name = os.path.splitext(os.path.basename(split_path))[0]


    # np.set_printoptions(threshold=sys.maxsize)
    # print(masks[5,20:100,20:100])


    # img_test_path = data_dir + '/images/' 'cju0rx1idathl0835detmsp84.jpg'
    # msk_test_path = data_dir + '/annotations/' + 'cju0rx1idathl0835detmsp84.jpg'
    

    # img = cv2.imread(img_test_path, cv2.IMREAD_COLOR)
    # mask = cv2.imread(msk_test_path, cv2.IMREAD_GRAYSCALE)

    # get a list of the paths, this would be done already in the program 



    # imgs = np.uint8(np.zeros([length, height, width, 3]))
    # plt.imshow(img, cmap='gray')
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    
    # ade20k_one_hot = one_hot_encode(random_mask_ADEK, num_classes = 150)

    # print(f'ade20k_one_hot.shape: {ade20k_one_hot.shape} ')

    # kvasir_one_hot = one_hot_encode(kvasir_path, num_classes = 2)
    # print(f'kvasir_one_hot.shape: {kvasir_one_hot.shape} ')

def showSingleImageAndMask(
    image_before_resize,
    mask_before_resize,
    image_before_crop, 
    image_after_crop, 
    mask_before_crop, 
    mask_after_crop
):
    """
    Shows a single image and mask in a 4x4 plot of the args, so image before cropping, 
    image after its been cropped. (nump matrix that's just been sliced from an OpenCV mat
    hence the dimensions will be NHWC). Args are obvious here. 
    """

    # new stuff for plotting and testing, all this stuff will go away soon enough 
    fig = plt.figure(figsize=(10, 7))
    rows = 3
    columns = 2

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    
    # showing image
    plt.imshow(image_before_crop)
    plt.axis('off')
    plt.title("Image Resized and before crop")
    
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    
    # showing image
    plt.imshow(image_after_crop)
    plt.axis('off')
    plt.title("Image Resized and after crop")
    
    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
    
    # showing image
    plt.imshow(mask_before_crop, cmap='gist_gray')
    plt.axis('off')
    plt.title("Masks Resized and before crop")
    
    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)
    
    # showing image
    plt.imshow(mask_after_crop, cmap='gist_gray')
    plt.axis('off')
    plt.title("Mask Resized and after crop")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 5)
    
    # showing image
    plt.imshow(image_before_resize)
    plt.axis('off')
    plt.title("Image before resize and crop")
    
    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 6)
    
    # showing image
    plt.imshow(mask_before_resize, cmap='gist_gray')
    plt.axis('off')
    plt.title("Mask before resize and crop")


    plt.show()

def showImageJUSTResizedORJUSTCropped(
    image_before_resize_or_crop,
    mask_before_resize_or_crop, 
    image_after_resize_or_crop, 
    mask_after_resize_or_crop,
):
    """
    For use only if the image was only resized or only cropped (so its just 
    two images to show really, one before the crop and one after the crop or
    one before the resize or one after the resize )
    """

    # new stuff for plotting and testing, all this stuff will go away soon enough 
    fig = plt.figure(figsize=(10, 7))
    rows = 2
    columns = 2
    
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    
    # showing image
    plt.imshow(image_before_resize_or_crop)
    plt.axis('off')
    plt.title("Image before resize or crop (unaltered)")
    
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    
    # showing image
    plt.imshow(mask_before_resize_or_crop, cmap='gist_gray')
    plt.axis('off')
    plt.title("Mask before resize or crop (unaltered)")
    
    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
    
    # showing image
    plt.imshow(image_after_resize_or_crop)
    plt.axis('off')
    plt.title("Image after resize or crop")
    
    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)
    
    # showing image
    plt.imshow(mask_after_resize_or_crop, cmap='gist_gray')
    plt.axis('off')
    plt.title("Mask after resize or crop")

    plt.show()