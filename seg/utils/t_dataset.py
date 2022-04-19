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

        # print(f'Using normalization: {normalization}')

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

def get_tDatasets_master(
    save_dir,
    normalize_gt=False,
    batch_size=1, 
    normalization='vit',
    num_workers=4, 
    pin_memory=True,
):
    """
    Getter function for all the test loaders required for the metric evaluation 
    using the general master dataset. 

    @save_dir: General location of where the data files are located. Should be 
        seg/data/master/{data_mask}_{ETIS, Kvasir, CVC_{...}}_test.npy
    @normalize_gt: T/F value to determine if we normalize the ground truth
    @batch_size: batch size for testing. should be set to 1
    @normalization: normalization method, should be vit given what we've been 
        running.
    @num_workers: see description above
    @pin_memory: see description above
    """
    test_cls = ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir']
    test_loaders = list()
    for i in range(len(test_cls)):
        print(f'Creating test dataloader: {test_cls[i]}')
        test_loaders.append(
            get_tDataset(
                image_root=save_dir + '/data_' + test_cls[i] + '_test.npy',
                gt_root=save_dir + '/mask_' + test_cls[i] + '_test.npy',
                normalize_gt=normalize_gt,
                batch_size=batch_size,
                normalization=normalization,
                num_workers=num_workers, 
                pin_memory=pin_memory,
            )
        )
    return test_loaders

if __name__ == '__main__':
    test_loaders = get_tDatasets_master(
       save_dir='seg/data/master',
       normalize_gt=False,
       batch_size=1, 
       normalization='vit', 
       num_workers=4,
       pin_memory=True,
    )
