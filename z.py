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
    save_location = "/home/john/Documents/Datasets/master_polyp_t/AugmentedTrainDataset",     
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
    save_img_dir = f'{save_location}/images'
    save_ann_dir = f'{save_location}/masks'
    save_aug_img_dir = f'{save_location}/aug_images'
    save_aug_ann_dir = f'{save_location}/aug_masks'
    save_aug_ann_gray_dir = f'{save_location}/aug_masks_gray'
    save_grid_dir = f'{save_location}/grids'

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_ann_dir, exist_ok=True)
    os.makedirs(save_aug_img_dir, exist_ok=True)
    os.makedirs(save_aug_ann_dir, exist_ok=True)
    os.makedirs(save_aug_ann_gray_dir, exist_ok=True)
    os.makedirs(save_grid_dir, exist_ok=True)

    assert split_path.endswith('.txt')
    size = num_lines = sum(1 for line in open(split_path))
    print(f"Length of {os.path.basename(split_path)}: {size}")

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
        grid_image = ia.draw_grid(cells, cols=5)
        imageio.imwrite(f'{save_grid_dir}/{unique_identifier}_grid_{paths[i]}', grid_image)

        imageio.imwrite(f'{save_aug_img_dir}/{unique_identifier}{paths[i]}', image_aug)
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
        imageio.imwrite(f'{save_aug_ann_gray_dir}/{unique_identifier}{paths[i]}', blackAndWhiteImage)
    

import shutil

def copyDir2Dir(
    source_folder,
    destination_folder,
):
    # fetch all files
    for file_name in os.listdir(source_folder):
        # construct full file path
        source = source_folder + file_name
        destination = destination_folder + file_name
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print('copied', file_name)
        else:
            print(f'ERROR. {source}')

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
    


if __name__ == '__main__':
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