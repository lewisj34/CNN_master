import sys 
import torch
import torch.nn.functional as F
import numpy as np 
from skimage.segmentation import find_boundaries
from preprocess import split_and_convert_to_npy
from seg.utils.dataset import get_dataset

def weighted_loss(seg_map, ground_truth):
    '''
    Calculates the loss of the network by weighting BCE and IoU
    Args:
        @seg_map (N = Batch Size, C = 1, H = Input Height, W = Input Width) 
        output segmentation map from network
        @ground_truth (N = Batch Size, C = 1, H = Input Height, W = Input Width) 
        mask or ground truth map with pixel by pixel labelling of class 
    '''
    # weight to output pixels that focuses on boundary pixels 
    weit = 1 + 5*torch.abs(
        F.avg_pool2d(
            ground_truth, kernel_size=31, stride=1, padding=15) - ground_truth) 
            # output dim of this is same as seg_map and ground_truth (NCHW)

    # print(weit)
    wbce = F.binary_cross_entropy_with_logits(seg_map, ground_truth, reduction='none') # reduction ensures the function will return a loss value for each element 
    # output will be of dimension: torch.Size([16, 1, 192, 256])
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)) 
    # output will be of dim: torch.Size([16, 1])

    pred = torch.sigmoid(seg_map)
    inter = ((pred * ground_truth)*weit).sum(dim=(2, 3)) # out_dim = [16, 1]
    union = ((pred + ground_truth)*weit).sum(dim=(2, 3)) # out dim = [16, 1]
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()



if __name__ == "__main__":
    a = torch.Tensor(np.random.uniform(0,1, (2, 1, 192, 256)))
    b = torch.Tensor(np.random.uniform(0,1, (2, 1, 192, 256)))
    print("a.shape: ", a.shape)
    print("b.shape", b.shape)
    print("weighted_loss = ", weighted_loss(a,b))

    dataset = "kvasir"
    save_dir = "seg/data"
    image_height = 192
    image_width = 256 
    train_dataset_path = "seg/data/"
    test_dataset_path = "seg/data/"
    batch_size = 16

    # import data and generate dataset  
    if dataset == 'kvasir':
        split_and_convert_to_npy(dataset, save_dir, image_height, image_width)
    else:
        print("Error: only Kvasir dataset supported at this time")
        sys.exit(1)

    # generate model and optimizer 

    image_data_file = train_dataset_path + "/data_train.npy"
    masks_data_file = test_dataset_path + "/mask_train.npy" 

    train_loader = get_dataset(
        dataset, 
        image_data_file, 
        masks_data_file, 
        batchsize=batch_size, 
        normalization="deit")

    # criterion = BCEIoUWeightedLoss()
    # print(criterion.weights)