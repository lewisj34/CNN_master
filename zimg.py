import cv2
from cv2 import IMREAD_GRAYSCALE 
import numpy as np 
import matplotlib.pyplot as plt

from seg.utils.data.generate_npy import one_hot_encode 

"""
Problem:

unique_vals (kvasir mask): [  0   1   2   3   4   5   6   7   8 248 249 250 251 252 253 254 255]
unique_vals (ade20k mask): [  0   1   4   6   9  11  15  18  28  38  48  59  66 146]

See one_hot_encode() in generate_npy.py for how we modify this (we use 
cv2.threshold) AFTER making all numbers below the average in the unique_vals
and then theresholding those values to be be 0 if theyr'e below the average or 1 if they're above the average 
So for example the above average would be 156.78

Now what we want to do, so we take in an individual mask_path 
and from one_hot_encode we return a 3D numpy matrix or a cv2 matrix
(H, W, num_classes) 

So now what we gotta do is incporate this per batch or whatever and can save it
and stuff 
"""

def main():
    kva_path = '/home/john/Documents/Datasets/kvasir_small/annotations/cju6vifjlv55z0987un6y4zdo.jpg'
    ade_path = '/home/john/Documents/Datasets/ade20k_5Sample/annotations/ADE_train_00000117.png'
    mask = cv2.imread(kva_path, IMREAD_GRAYSCALE)
    one_hot = one_hot_encode(
        image_path = kva_path,
        num_classes = 2,
    )
    print(one_hot.shape)

    # rows, cols, chans = mask.shape
    # print(f'rows, cols, chans: {rows, cols, chans}')
    
    # unique_vals = np.unique(mask)
    # print(f'unique_vals: {unique_vals}')



if __name__ == '__main__':
    main()