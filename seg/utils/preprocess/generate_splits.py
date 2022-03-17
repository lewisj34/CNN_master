'''
DEFUNCT
DEFUNCT
DEFUNCT
DEFUNCT
DEFUNCT
DEFUNCT
DEFUNCT
DEFUNCT
DEFUNCT
DEFUNCT
DEFUNCT
DEFUNCT
DEFUNCT
DEFUNCT
DEFUNCT
(still works tho)
'''


import pandas as pd 
from seg.utils.preprocess.generate_csv import generate_csv 
import numpy as np 
import os 
import shutil 
import sys 
import json 

def list_overlap(a, b):
    return bool(set(a) & set(b))

def generate_splits(parent_dir):
    '''
    Generates .txt files for train, valid, and test sets by creating a csv for 
    each (train, valid, ...) and then converting that csv to a .txt file for 
    each and then populating the split_dir with each of those .txt files.     
    Args:
        parent_dir: contains img_dir, ann_dir, and split_dir. File structure:
            parent_dir/
                /images
                /annotations
                /splits
            NOTE: If /splits doesn't exist, it will generate it. 
    '''
    split_dir = parent_dir + '/splits/'
    img_dir = parent_dir + '/images', 
    ann_dir = parent_dir + '/annotations'

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
    train_path = split_dir + "/train.txt"
    valid_path = split_dir + "/valid.txt"
    test_path = split_dir + "/test.txt"
    
    with open(train_path, "w") as outfile:
        outfile.write("\n".join(train_list))
    with open(valid_path, "w") as outfile:
        outfile.write("\n".join(valid_list))
    with open(test_path, "w") as outfile:
        outfile.write("\n".join(test_list))

    print("Complete: split files written.\n")

def main():
    # do something
    generate_splits()

if __name__ == "__main__":
    main() 