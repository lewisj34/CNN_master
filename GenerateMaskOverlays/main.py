"""
script to generate image overlays of black and white masks along with a corresponding 
multi plot for the overlays of the original image and the annotation along with 
the original image and the model prediction, along with the original image
"""
import os 

import cv2
import click
import numpy as np
import time 
import progressbar
import matplotlib.pyplot as plt 

def justGetBoundarySingleImage(
    img_path: str,
    output_path: str = None,
):
    """
    Takes a binary mask as input, @img_path, (just black + white) and returns 
    the boundary as a numpy array. Also saves it to the @output_path. 

    Taken from: https://medium.com/@rootaccess/how-to-detect-edges-of-a-mask-in-python-with-opencv-4bcdb3049682
    """
    img_data = cv2.imread(img_path)
    img_data = img_data > 128

    img_data = np.asarray(img_data[:, :, 0], dtype=np.double)
    gx, gy = np.gradient(img_data)
    temp_edge = gy * gy + gx * gx
    temp_edge[temp_edge != 0.0] = 255.0
    temp_edge = np.asarray(temp_edge, dtype=np.uint8)
    
    if output_path != None:
        cv2.imwrite(output_path, temp_edge)
    return temp_edge

def overlayOriginalImage(
    img_path: str,
    msk_path: str,
    maskColor: str = 'white', # white, red, purple
    output_path: str = None,
):
    msk_edge = justGetBoundarySingleImage(msk_path)

    og_image = cv2.imread(img_path)
    msk_image = cv2.imread(msk_path, cv2.IMREAD_COLOR)

    # uncomment this line 
    if maskColor == 'white':
        pass
    elif maskColor == 'red':
        msk_image[np.where((msk_image==[255, 255, 255]).all(axis=2))] = [0, 0, 255]
    elif maskColor == 'green':
        msk_image[np.where((msk_image==[255, 255, 255]).all(axis=2))] = [255, 255, 0]
    elif maskColor == 'lighter_pink':
        msk_image[np.where((msk_image==[255, 255, 255]).all(axis=2))] = [255, 0, 0]
    elif maskColor == 'pink':
        msk_image[np.where((msk_image==[255, 255, 255]).all(axis=2))] = [255, 0, 255]
    elif maskColor == 'grey':
        msk_image[np.where((msk_image==[255, 255, 255]).all(axis=2))] = [255,165,0]
    else:
        raise ValueError(f'maskColor: {maskColor} not available.')


    # this puts on the annotation or prediction with some transparency
    merged_image = cv2.addWeighted(og_image,1.0,msk_image,0.2,1.0)
    
    
    # now put on the border with no transparency / full opacity 
    merged_image = cv2.addWeighted(merged_image,1.0, cv2.cvtColor(msk_edge, cv2.COLOR_GRAY2RGB),1.0,1.0)

    if output_path != None:
        cv2.imwrite(output_path, merged_image)

def getSortedListOfFilePaths(dir: str):
    """
    returns list of files in @dir from os.listdir + the whole path to that file
    """
    list = os.listdir(dir)
    list = [x[:-4] for x in list]
    list = sorted([int(x) for x in list])
    list = [dir + '/' + str(x) + '.png' for x in list]
    return list

def getSortedListofFileNames(dir: str):
    """
    returns list of files from os.listdir just sorted (inc order) assuming ALL 
    numbers 
    same as above but just doesnt return the total path to the file it just 
    returns the file name 
    """
    list = os.listdir(dir)
    list = [x[:-4] for x in list]
    list = sorted([int(x) for x in list])
    list = [str(x) + '.png' for x in list]
    return list

def generate3Plot(
    img_path: str = "1_img.png",
    imgann_path: str = "1_imgann.png",
    imgprd_path: str = "1_img_prd.png",
    save_name: str = '1.png',
):
    """
    taking the images, image+pred masks overlays, image+ann masks overlays, 
    generates a 3 plot of each and saves them each to a directory 
    """

    img = cv2.imread(img_path, cv2.IMREAD_COLOR); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ann = cv2.imread(imgann_path, cv2.IMREAD_COLOR); ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
    prd = cv2.imread(imgprd_path, cv2.IMREAD_COLOR); prd = cv2.cvtColor(prd, cv2.COLOR_BGR2RGB)

    plt.style.use('seaborn-white')

    # note super titles (Original, Annotaiton, Predicted) DO NOT WORK (they cut off)
    # will have to add these manually if we want them though I think its easy
    # to work around this 
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        if i == 0:
            plt.imshow(img, cmap=plt.cm.jet)
            # plt.title('Original')
        elif i == 1:
            plt.imshow(ann, cmap=plt.cm.jet)
            # plt.title('Annotation')
        elif i == 2:
            plt.imshow(prd, cmap=plt.cm.jet)
            # plt.title('Predicted')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_name)

@click.command(help='')
@click.option('--save_dir', type=str, default='CVC_ClinicDB_total/', help='where to save the combined images (image + mask')
@click.option('--src_imgs', type=str, default='CVC_ClinicDB_total/images')
@click.option('--src_anns', type=str, default='CVC_ClinicDB_total/gts')
@click.option('--src_prds', type=str, default='CVC_ClinicDB_total/outputs')
def main(
    save_dir,
    src_imgs,
    src_anns,
    src_prds,
):
    overlayOriginalImage(
        img_path='556_img.png',
        msk_path='556_prd.png',
        maskColor='pink',
        output_path='img_plus_ann.png'
    )

    os.makedirs(save_dir + '/img_ann/', exist_ok=True)
    os.makedirs(save_dir + '/img_prd/', exist_ok=True)
    os.makedirs(save_dir + '/MultiPlot/', exist_ok=True)

    img_names = getSortedListofFileNames(src_imgs) # all file names are the same {1, 2, ... x.png}
    img_paths = getSortedListOfFilePaths(src_imgs)
    ann_paths = getSortedListOfFilePaths(src_anns)
    prd_paths = getSortedListOfFilePaths(src_prds)

    with progressbar.ProgressBar(max_value=len(img_names)) as bar:
        for i in range(len(img_paths)):
            time.sleep(0.1)
            bar.update(i)

            # do it for the img + anns
            overlayOriginalImage(
                img_path=img_paths[i],
                msk_path=ann_paths[i],
                maskColor='green',
                output_path=save_dir + '/img_ann/' + img_names[i],
            )
            # now for the img + prds
            overlayOriginalImage(
                img_path=img_paths[i],
                msk_path=prd_paths[i],
                maskColor='pink',
                output_path=save_dir + '/img_prd/' + img_names[i],
            )
            # now that we have img_anns + img_preds images saved, now MultiPlot
            generate3Plot(
                img_path=img_paths[i],
                imgann_path=save_dir + '/img_ann/' + img_names[i],
                imgprd_path=save_dir + '/img_prd/' + img_names[i],
                save_name=save_dir + '/MultiPlot/' + img_names[i]
            )

if __name__ == '__main__':
    main()