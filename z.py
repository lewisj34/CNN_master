"""
So far, 3 new params imported to main 


Modified data processer that converts the directory of images and masks to a 
.npy file, while also provides options for transforming the image. I'm not sure 
if this will be just adding a paremter which is a list of transforms or a 
transform.Compose object but anyways yeah 

first priority is to develop a program that can take in the images and have the 
tensor or the numpy array be of format (N, C, H, W) with C = 2 
"""

backbone = 'vit'

import click
import random 

from seg.utils.dataset import get_dataset
from seg.utils.t_dataset import get_tDataset
from seg.utils.err_codes import crop_err1, crop_err2, crop_err3,resize_err1, resize_err2, resize_err3
from seg.utils.data.generate_npy import split_and_convert_to_npyV2

@click.command(help='')
@click.option('--dataset', type=str, default='kvasir')
@click.option('--save_dir', type=str, default='ztest/seg/data')
@click.option('--image_height', type=int, default=256)
@click.option('--image_width', type=int, default=256)
@click.option('--crop_height', type=int, default=None)
@click.option('--crop_width', type=int, default=None)
@click.option('--no_transforms_before_runtime', type=bool, default=False)
@click.option('--reimport_data', type=bool, default=False)
@click.option('--n_cls', type=int, default=2)
@click.option('--batch_size', type=int, default=10)
def main(
    dataset,
    save_dir,
    image_height,
    image_width,
    crop_height,
    crop_width,
    no_transforms_before_runtime,
    reimport_data,
    n_cls,
    batch_size,
):
    """
    ****************************** USAGE ****************************** 
    4 Situations we need to cover:
        1. Just resize, no crop. 
            -> given by crop_size == None. 
        2. Resize and then crop. 
            -> both image_size and crop_size given. 
        3. No resize, just crop. (Needs an additional check here potentially 
            when iterating through the dir and openCVing the images, lets run
            a test on this. - shouldnt be hard) 
            -> given by image_size, image_width = 0. 
            -> crop_size must not be None 
    """
    if crop_height == 0 or crop_width == 0:
        assert crop_height == 0 and crop_width == 0, crop_err3(crop_height, crop_width)
        print(f'Crop_height or crop_width was manually input in command line as 0. Resetting thes values to None type.')
        crop_height = None
        crop_width = None 

    if crop_height is not None or crop_width is not None:
        """
        2 situations apply when crop_size is given. 
            1. Resize and then crop. 
            2. No resize, just crop. 
        """
        # following check applies to both situations
        assert crop_height is not None and crop_width is not None, crop_err1(crop_height, crop_width)
        
        # situation 1: resize and then crop
        if image_height != 0 or image_width != 0:
            print(f'Resizing and cropping.')
            assert image_height != 0 and image_width != 0, resize_err1(image_height, image_width)
            
            if image_height < crop_height or image_width < crop_width:
                raise ValueError(crop_err2(crop_height, crop_width, image_height, image_width))
            resize_size = (image_height, image_width)
            crop_size = (crop_height, crop_width)
            image_size = crop_size
        
        # situation 2: just crop (because image_height and width are given as 0) 
        else:
            print(f'Cropping but not resizing the input data.')
            assert image_height == 0 and image_width == 0, resize_err2(image_height, image_width)
            resize_size = (None, None)
            crop_size = (crop_height, crop_width)
            image_size = (crop_height, crop_width)

    else:
        print(f'Reszing but not cropping the input data.')
        assert image_height != 0 and image_width != 0, resize_err3(image_height, image_width)
        resize_size = (image_height, image_width)
        crop_size = (None, None)
        image_size = resize_size

    if no_transforms_before_runtime:
        resize_size = (None, None)
        crop_size = (None, None)
        image_size = (None, None)
        print(f'********* Warning: no resize or cropping, just importing as is to binaries... *********')
        raise NotImplementedError(f'No resize or cropping val assigned, just importing as is to binaries -> Exiting for now...')
    print(f'resize_size: {resize_size}')
    print(f'crop_size: {crop_size}')
    print(f'image_size: {image_size}')

    if resize_size[0] != resize_size[1] or crop_size[0] != crop_size[1] \
        or image_size[0] != image_size[1]:
        print(f'********* Warning: input image dims are not square. *********')
    
    print(f'dir: {save_dir}')
    split_and_convert_to_npyV2(
        dataset = dataset,
        save_dir = save_dir, 
        resize_size = resize_size,
        crop_size = crop_size,
        image_size = image_size,
        reimport_data=False, 
    )
    print('Exiting...')
    exit(1)


    # following code can be removed all i want to do is just test if we've 
    # imported the data properly 

    data_train_loc = save_dir + '/data_train.npy'
    data_valid_loc = save_dir + '/data_valid.npy'
    data_test_loc = save_dir + '/data_test.npy'

    mask_train_loc = save_dir + '/mask_train.npy'
    mask_valid_loc = save_dir + '/mask_valid.npy'
    mask_test_loc = save_dir + '/mask_test.npy'

    train_loader = get_dataset(
        dataset, 
        save_dir + "/data_train.npy", # location of train images.npy file (str)
        save_dir + "/mask_train.npy" , # location of train masks.npy file (str)
        batchsize=batch_size, 
        normalization="deit" if "deit" in backbone else "vit"
    )

    valid_loader = get_tDataset(
        image_root = save_dir + "/data_valid.npy",
        gt_root = save_dir + "/mask_valid.npy",
        normalize_gt = False,
        batch_size = 1,
        normalization = 'vit',
        num_workers = 4, 
        pin_memory=True,
    )
    test_loader = get_tDataset(
        image_root = save_dir + "/data_test.npy",
        gt_root = save_dir + "/mask_test.npy",
        normalize_gt = False,
        batch_size = 1,
        normalization = 'vit',
        num_workers = 4, 
        pin_memory=True,
    )

    visualizeImagesAndGtsFromDataLoadder(
        dl = train_loader,
        num_images_to_randomly_sample_and_show=5,
    )
    
import matplotlib.pyplot as plt 
def visualizeImagesAndGtsFromDataLoadder(
    dl,
    num_images_to_randomly_sample_and_show=5,
):
    image_list = list()
    gt_list = list()

    # iterate through each batch
    for i, image_gt in enumerate(dl):
        images, gts = image_gt
        images = images.permute(0, 2, 3, 1)
        gts = gts.permute(0, 2, 3, 1)

        assert images.size()[0] == gts.size()[0]
        bs = images.size()[0]
        for img_in_batch in range(bs):
            image_list.append(images[img_in_batch, :, :, :])
            gt_list.append(gts[img_in_batch, :, :, :])

    # get random sample of images and gts from image_list and gt_list
    assert len(image_list) == len(gt_list), f'number of images and gts not same'
    indices = random.sample(image_list, num_images_to_randomly_sample_and_show)
    
    fig, ax = plt.subplots(num_images_to_randomly_sample_and_show, 2, sharex='col', sharey='row')
    for i in range(len(indices)):
        ax[i, 0].imshow(image_list[i])
        ax[0, 0].set_title(f'Input Image')
        ax[i, 1].imshow(gt_list[i], cmap='gist_gray')
        ax[0, 1].set_title(f'Ground Truth')
    fig.suptitle('Model Output and Visualization') 
    plt.show()




if __name__ == '__main__':
    main()
