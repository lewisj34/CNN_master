'''
Takes in a directory with file structure:
    /images
    /annoations
And then moves all the images into a train/test/split organization by generating 
text split files as follows:
    train.txt
    val.txt
    test.txt
Therefore NOTE: What you need to know is that the incoming file structure, is 
    not split. The splits are generated with this code. 
'''

import os, os.path 
import sys
import cv2
import yaml
import numpy as np 
import pandas as pd 
import os 
import shutil 

from time import sleep
from tqdm import tqdm
from pathlib import Path 

# from seg.utils.preprocess.generate_splits import generate_splits
from seg.utils.preprocess.generate_csv import generate_csv 

def list_overlap(a, b):
    return bool(set(a) & set(b))

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

    assert(os.path.isdir(parent_dir) 
        and os.path.isdir(img_dir) 
        and os.path.isdir(ann_dir))

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



def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop



example_image = np.random.randint(0, 256, (1024, 1024, 3))
random_crop = get_random_crop(example_image, 64, 64)

def process_dataset(
    split_path,        
    save_location = "data/",     
    img_dir = "/home/john/Documents/Datasets/kvasir_merged/images/",            
    ann_dir = "/home/john/Documents/Datasets/kvasir_merged/annotations/", 
    height = 192,            
    width = 256,
    crop_size = None,
    ):            
    '''
    From split text files, detailing dataset split, finds these files and 
    imports them with OpenCV and then dumps them to .npy binary files in 
    save_location from corresponding img_dir and ann_dir with corr. height and
    width 
    Args:
        split_path: path to split.txt file, EITHER: {train, valid, test}.txt
        save_location: directory to save .npy data files 
        img_dir: location of img paths that split path points 
        ann_dir: location of ann paths that split path points
        height: resized height
        width: resized width
        crop_size: the size to crop the resized image to (square)      
    '''
    assert split_path.endswith('.txt')
    size = num_lines = sum(1 for line in open(split_path))
    print(f"Length of {os.path.basename(split_path)}: {size}")
    print(f"Processing {os.path.basename(split_path)} to .npy files")

    with open(split_path) as f:
        paths = f.readlines()
    paths = list(map(lambda s: s.strip(), paths))
    count = 0
    length = size
    imgs = np.uint8(np.zeros([length, height, width, 3]))
    masks = np.uint8(np.zeros([length, height, width]))

    image_paths = paths.copy()
    mask_paths= paths.copy()

    for i in tqdm(range(len(paths))):
        sleep(0.0001)
        image_paths[i] = img_dir + image_paths[i]
        mask_paths[i] = ann_dir + mask_paths[i]

        img = cv2.imread(image_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height))

        mask = cv2.imread(mask_paths[i], 0)
        mask = cv2.resize(mask, (width, height))

        if crop_size is not None:
            img = get_random_crop(img, crop_size, crop_size)
            mask = get_random_crop(img, crop_size, crop_size)

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

def split_and_convert_to_npy_OLD(
    dataset = "kvasir", 
    save_dir = "seg/data",
    image_height = 192, 
    image_width = 256,
    crop_size = None,
    reimport=False,
    ):
    model_params = {
        'dataset': dataset, 
        'save_dir': save_dir, 
        'image_height': image_height, 
        'image_width': image_width,
        'crop_size': crop_size,
        }
    # check if the directory with parents exist, if not - create it
    if os.path.isdir(save_dir):
        print("Directory:", save_dir + " already exists.")
    else:
        print("Directory:", save_dir + " does not exist. Creating.")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    if data_files_exist(save_dir):
        print("All data files exist")
        file_params = yaml.load(open(Path(save_dir + "/params.yaml")), 
            Loader=yaml.FullLoader)
        if file_params == model_params and not reimport:
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
        parent_dir = '/home/john/Documents/Datasets/kvasir_merged'
    elif dataset == 'CVC_ClinicDB':
        parent_dir = '/home/john/Documents/Datasets/CVC-ClinicDB/PNG'

    TRAIN_SPLIT_PATH = parent_dir + "/splits/" + "train.txt"
    VALID_SPLIT_PATH = parent_dir + "/splits/" + "valid.txt"
    TEST_SPLIT_PATH = parent_dir + "/splits/" + "test.txt"    
    IMG_DIR = parent_dir + '/images/'
    ANN_DIR = parent_dir + '/annotations/'
    
    # generate split mapping file paths {train, valid, test}.txt
    generate_split_txt_files(parent_dir=parent_dir)

    # organize train data into {data, mask}.npy files from splits
    process_dataset(
        split_path = TRAIN_SPLIT_PATH, 
        save_location = save_dir, 
        img_dir = IMG_DIR,
        ann_dir = ANN_DIR,
        height = image_height, 
        width = image_width)

    # organize valid data into {data, mask}.npy files from splits
    process_dataset(
        split_path = VALID_SPLIT_PATH, 
        save_location = save_dir, 
        img_dir = IMG_DIR,
        ann_dir = ANN_DIR,
        height = image_height, 
        width = image_width)

    # organize test data into {data, mask}.npy files from splits 
    process_dataset(
        split_path = TEST_SPLIT_PATH, 
        save_location = save_dir, 
        img_dir = IMG_DIR,
        ann_dir = ANN_DIR,
        height = image_height, 
        width = image_width)



        
if __name__ == "__main__":
    split_and_convert_to_npy_OLD()