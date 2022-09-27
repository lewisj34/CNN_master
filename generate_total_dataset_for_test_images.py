"""
This script and set of files basically just generates a txt file corresponding
to the full dataset (UNSPLIT), and then generates npy files corresponding to 
that "split" (even though thats a misnomer, it generates npy files corresponding)
to (hopefully) the full ORDERED datset so that hopefully we can generate gif files
without having to reorganize the whole dataset

WHY DID WE DO THIS: 
1. We wanted to run a test_images on an ENTIRE dataset so that we could hopefully get
as many consistent frames that could then make a gif for seamless presentation IN ORDER
so that we weren't going through the dataset over and over again 
"""
import os
import click
import shutil

from seg.utils.data.generate_npy import process_dataset

def generate_splits_in_order(
    parent_dir='/home/john/Documents/Datasets/CVC-ClinicDB/PNG'
):
    """
    Generates a .txt file that holds the txt files to be used as "splits"
    even though this doesn't ACTUALLY SPLIT ANY DATA!
    Assumes that the splits being input are only numbers: ex:
        1.jpg, 2.jpg, 3.jpg, ...
    And are IN ORDER 

    And then generates a txt file IN ORDER (as os.listdir messes it up) 
        parent_dir: holds the images and annotations dir (assumes same file type)
    returns:
        split_path: holds split_path for imgs and anns 

    Sample usage: python generate_total_dataset_for_test_images.py --dataset_save_name CVC_ColonDB --parent_dir /home/john/Documents/Datasets/CVC-ColonDB
    """
    img_dir = parent_dir + '/images/'
    ann_dir = parent_dir + '/annotations/'
    
    split_path = parent_dir + '/splits_ordered_no_division/'
    os.makedirs(split_path, exist_ok=True)

    img_list = os.listdir(img_dir)
    ann_list = os.listdir(ann_dir)

    img_list = [x[:-4] for x in img_list]
    ann_list = [x[:-4] for x in ann_list]

    img_list = sorted([int(x) for x in img_list])
    ann_list = sorted([int(x) for x in ann_list])
    
    img_list = [str(x) + '.png' for x in img_list]
    ann_list = [str(x) + '.png' for x in ann_list]
    
    assert img_list == ann_list, f'different names or file endings in list of images and annotations, run a for loop to see which file names are different'

    with open(split_path + 'dataset_list_ordered.txt', 'w') as f:
        for line in img_list:
            f.write(f"{line}\n")
    return split_path + 'dataset_list_ordered.txt'

@click.command(help='')
@click.option('--parent_dir', type=str, default='/home/john/Documents/Datasets/CVC-ClinicDB/PNG')
@click.option('--dataset_save_name', type=str, help='what to name the npy save location as') # ex: CVC_ClinicDB
@click.option('--resize_size', type=int, default=512)
def main(
    parent_dir,
    dataset_save_name,
    resize_size,
):
    split_path = generate_splits_in_order(parent_dir=parent_dir)
    save_location = 'seg/data/totals/' + dataset_save_name
    os.makedirs(save_location, exist_ok=True)
    # just for posterity we're going to copy the "split" .txt file detailing
    # the full ordered dataset to where we store the numpy files (as right now 
    # the npy file is in seg/data/... and the split.txt file saves to where the 
    # dataset is saved)
    shutil.copy(split_path, save_location)

    process_dataset(
        split_path=split_path,
        save_location=save_location,
        img_dir = parent_dir + '/images/',
        ann_dir = parent_dir + '/annotations/',
        resize_size = (resize_size, resize_size), 
        image_size = (resize_size, resize_size), 
    )

if __name__ == '__main__':
    main()