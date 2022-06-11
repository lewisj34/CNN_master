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

from seg.utils.data.generate_npy import generate_split_txt_files_master

def _load_augmentation_aug_non_geometric():
    """
    
    """
    return iaa.Sequential([
        iaa.Sometimes(0.3, iaa.Multiply((0.5, 1.5), per_channel=0.5)),
        iaa.Sometimes(0.2, iaa.JpegCompression(compression=(70, 99))),
        iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
        iaa.Sometimes(0.2, iaa.MotionBlur(k=15, angle=[-45, 45])),
        iaa.Sometimes(0.2, iaa.MultiplyHue((0.5, 1.5))),
        iaa.Sometimes(0.2, iaa.MultiplySaturation((0.5, 1.5))),
        iaa.Sometimes(0.34, iaa.MultiplyHueAndSaturation((0.5, 1.5),
                                                         per_channel=True)),
        iaa.Sometimes(0.34, iaa.Grayscale(alpha=(0.0, 1.0))),
        iaa.Sometimes(0.2, iaa.ChangeColorTemperature((1100, 10000))),
        iaa.Sometimes(0.1, iaa.GammaContrast((0.5, 2.0))),
        iaa.Sometimes(0.2, iaa.SigmoidContrast(gain=(3, 10),
                                               cutoff=(0.4, 0.6))),
        iaa.Sometimes(0.1, iaa.CLAHE()),
        iaa.Sometimes(0.1, iaa.HistogramEqualization()),
        iaa.Sometimes(0.2, iaa.LinearContrast((0.5, 2.0), per_channel=0.5)),
        iaa.Sometimes(0.1, iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)))
    ]) 
def _load_aug_ex1_img_aug():
    return iaa.Sequential([
            iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
            iaa.Sharpen((0.0, 1.0)),       # sharpen the image
            iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
            # iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
            iaa.GaussianBlur(sigma=(0, 3.0))
        ], random_order=True)

def _load_aug_ex2_img_aug():
    return iaa.Sequential([
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

def process_dataset_augmentations(
    aug,
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
        if aug == '1':
            seq = _load_aug_ex1_img_aug()
        elif aug == '2':
            seq = _load_aug_ex2_img_aug()
        elif aug == '3':
            seq = _load_augmentation_aug_non_geometric()
        
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
    
if __name__ == '__main__':
    parent_dir='/home/john/Documents/Datasets/master_polyp_mini'
    split_dir = parent_dir + '/splits/'
    generate_split_txt_files_master(
        parent_dir='/home/john/Documents/Datasets/master_polyp_mini'
    )

    splits = [
        '/home/john/Documents/Datasets/master_polyp_mini/splits/CVC_300_test.txt',
        # '/home/john/Documents/Datasets/master_polyp_mini/splits/CVC_ClinicDB_test.txt',
        # '/home/john/Documents/Datasets/master_polyp_mini/splits/CVC_ColonDB_test.txt',
        # '/home/john/Documents/Datasets/master_polyp_mini/splits/ETIS_test.txt',
        # '/home/john/Documents/Datasets/master_polyp_mini/splits/Kvasir_test.txt',
        # '/home/john/Documents/Datasets/master_polyp_mini/splits/train.txt',
        # '/home/john/Documents/Datasets/master_polyp_mini/splits/valid.txt',
    ]
    img_dirs = [
        '/home/john/Documents/Datasets/master_polyp_mini/TestDataset/CVC-300/images/',
        # '/home/john/Documents/Datasets/master_polyp_mini/TestDataset/CVC-ClinicDB/images/',
        # '/home/john/Documents/Datasets/master_polyp_mini/TestDataset/CVC-ColonDB/images/',
        # '/home/john/Documents/Datasets/master_polyp_mini/TestDataset/ETIS-LaribPolypDB/images/',
        # '/home/john/Documents/Datasets/master_polyp_mini/TestDataset/Kvasir/images/',
        # '/home/john/Documents/Datasets/master_polyp_mini/TrainDataset/image/',
        # '/home/john/Documents/Datasets/master_polyp_mini/TrainDataset/mask/',
    ]
    ann_dirs = [
        '/home/john/Documents/Datasets/master_polyp_mini/TestDataset/CVC-300/masks/',
        # '/home/john/Documents/Datasets/master_polyp_mini/TestDataset/CVC-ClinicDB/masks/',
        # '/home/john/Documents/Datasets/master_polyp_mini/TestDataset/CVC-ColonDB/masks/',
        # '/home/john/Documents/Datasets/master_polyp_mini/TestDataset/ETIS-LaribPolypDB/masks/',
        # '/home/john/Documents/Datasets/master_polyp_mini/TestDataset/Kvasir/masks/',
        # '/home/john/Documents/Datasets/master_polyp_mini/TrainDataset/image/',
        # '/home/john/Documents/Datasets/master_polyp_mini/TrainDataset/mask/',
    ]

    # what we're going to do, 
    paths = os.listdir(split_dir)
    for i in range(len(paths)):
        process_dataset_augmentations(
            aug=3,
            split_path=splits[i],
            save_aug_img_location=parent_dir + '/saves/augd_images/',
            save_aug_ann_location=parent_dir + '/saves/augd_masks/',
            save_grid_location=parent_dir+'/saves/grids/',
            img_dir=img_dirs[i],
            ann_dir=ann_dirs[i],
            unique_identifier=f'z_{os.path.basename(os.path.splitext(splits[i])[0])}_{i}',
        )


