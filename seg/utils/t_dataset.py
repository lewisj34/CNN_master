import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2
from torchvision.transforms.transforms import RandomCrop

class tDataset(data.Dataset):
    """
    TESTING AND VALIDATION DATASET. FINAL VERSION. 
    
    as there are no image transforms performed on either beyond normalization. 
    There is also an option to include normalization of the ground truth in the
    event a loss function can permit it. However, inputting this arg as True,
    and thus normalizing the ground truth is not recommended for loss functions
    that use BCE / cross entropy.  
    """
    def __init__(
        self, 
        image_root, 
        gt_root, 
        normalization="vit",
        normalize_gt=False,
    ):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        self.normalize_gt = normalize_gt

        assert len(self.images) == len(self.gts) 
        self.size = len(self.images)

        print(f'Using normalization: {normalization}')

        if normalization == "vit":
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5])
                ]) 
        elif normalization == "deit":
            self.img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
                ])
        else:
            print("Error: Normalization used: ", normalization)
            raise ValueError("Normalization can only be vit or deit")

        if normalize_gt:
            if normalization == "vit":
                self.gt_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5])
                    ]) 
            elif normalization == "deit":
                self.gt_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                    ])
            else:
                print("Error: Normalization used: ", normalization)
                raise ValueError("Normalization can only be vit or deit")
        else:
            self.gt_transform = transforms.Compose([
                transforms.ToTensor()])
        

    def __getitem__(self, index):
        image = self.images[index]
        gt = self.gts[index]
        gt = gt / 255.0

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    def __len__(self):
        return self.size

def get_tDataset(
    image_root,
    gt_root, 
    normalize_gt=False,
    batch_size=1,
    normalization='vit',
    shuffle=True,
    num_workers=4,
    pin_memory=True,
):
    """
    GETTER FUNCTION FOR TEST AND VALIDATION DATASETS 
        @data_dir: the location holding the .npy files. 
        @batch_size: size of batch
        @normalization: type of normalization {vit, deit}
        @batch_size: number of batches
        @shuffle: shuffle the input data
        @num_workers: number of processes to generate the batches with
        @pin_memory: set to True if loading samples on CPU and are going to 
        push the samples to GPU for training, as this will speed up the process.
        Dataloader allocates the samples in page-locked memory which speeds up 
        the transfer between the CPU and GPU later. 
    """
    dataset = tDataset(
        image_root = image_root,
        gt_root = gt_root,
        normalization = normalization,
        normalize_gt = normalize_gt,
    )
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = pin_memory,
    )
    return data_loader         
