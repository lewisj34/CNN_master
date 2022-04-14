import sys
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2
from torchvision.transforms.transforms import RandomCrop


class KvasirDataset(data.Dataset):
    """
    Dataset for kvasir polyp data
    """
    def __init__(
        self, 
        image_root, 
        gt_root, 
        normalization="vit"
    ):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

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



        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, \
                    rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ]
        )

    def __getitem__(self, index):
        image = self.images[index]
        gt = self.gts[index]
        gt = gt/255.0

        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        gt = self.gt_transform(transformed['mask'])
        return image, gt

    def __len__(self):
        return self.size

class CVC_ClinicDB_Dataset(data.Dataset):
    """
    Dataset for the CVC_ClinicDB polyp data
    """
    def __init__(self, image_root, gt_root, normalization="deit"):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        assert len(self.images) == len(self.gts) 
        self.size = len(self.images)

        # print("Using normalization: ", normalization)

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



        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, \
                    rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ]
        )

    def __getitem__(self, index):
        image = self.images[index]
        gt = self.gts[index]
        gt = gt/255.0

        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        gt = self.gt_transform(transformed['mask'])
        return image, gt

    def __len__(self):
        return self.size

class CVC_ColonDB_Dataset(data.Dataset):
    """
    Dataset for the CVC_ClinicDB polyp data
    """
    def __init__(self, image_root, gt_root, normalization="deit"):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        assert len(self.images) == len(self.gts) 
        self.size = len(self.images)

        # print("Using normalization: ", normalization)

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



        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, \
                    rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ]
        )

    def __getitem__(self, index):
        image = self.images[index]
        gt = self.gts[index]
        gt = gt/255.0

        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        gt = self.gt_transform(transformed['mask'])
        return image, gt

    def __len__(self):
        return self.size

class ETIS_dataset(data.Dataset):
    """
    Dataset for the ETIS polyp data
    """
    def __init__(self, image_root, gt_root, normalization="deit"):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        assert len(self.images) == len(self.gts) 
        self.size = len(self.images)

        # print("Using normalization: ", normalization)

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



        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, \
                    rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ]
        )

    def __getitem__(self, index):
        image = self.images[index]
        gt = self.gts[index]
        gt = gt/255.0

        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        gt = self.gt_transform(transformed['mask'])
        return image, gt

    def __len__(self):
        return self.size

def get_dataset(
    dataset,
    image_root, 
    gt_root, 
    batchsize, 
    normalization,
    shuffle=True, 
    num_workers=4, 
    pin_memory=True):
    if dataset == "kvasir":
        dataset = KvasirDataset(image_root, gt_root, 
            normalization=normalization)
        data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=batchsize,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)
    elif dataset == "CVC_ClinicDB":
        dataset = CVC_ClinicDB_Dataset(image_root, gt_root, 
            normalization=normalization)
        data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=batchsize,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)    
    elif dataset == "ETIS":
        dataset = ETIS_dataset(image_root, gt_root, 
            normalization=normalization)
        data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=batchsize,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory)  
    elif dataset == "CVC_ColonDB":
        dataset = CVC_ColonDB_Dataset(image_root, gt_root, 
            normalization=normalization)
        data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=batchsize,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory) 
    else:
        print("Error: Only Kvasir dataset supported at this time")
        sys.exit(1)

    return data_loader

class test_dataset:
    def __init__(self, image_root, gt_root, normalization="deit"):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        # print("Using normalization: ", normalization)

        if normalization == "vit":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5])
                ]) 
        elif normalization == "deit":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
                ])
        else:
            print("Error: Normalization used: ", normalization)
            raise ValueError("Normalization can only be vit or deit")

        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        gt = self.gts[self.index]
        gt = gt/255.0
        self.index += 1

        return image, gt

class TestDataset:
    def __init__(self, image_root, gt_root, normalization="deit"):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        # print("Using normalization: ", normalization)

        if normalization == "vit":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5])
                ]) 
        elif normalization == "deit":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
                ])
        else:
            raise ValueError("Normalization can only be vit or deit")

        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        gt = self.gts[self.index]
        gt = gt/255.0
        self.index += 1

        return image, gt

class ValidDataset:
    def __init__(self, image_root, gt_root, normalization="deit"):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        # print("Using normalization: ", normalization)

        if normalization == "vit":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5])
                ]) 
        elif normalization == "deit":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
                ])
        else:
            print("Error: Normalization used: ", normalization)
            raise ValueError("Normalization can only be vit or deit")

        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        gt = self.gts[self.index]
        gt = gt/255.0
        self.index += 1

        return image, gt





class TestDatasetV2(data.Dataset):
    """
    Dataset for testing. 
    """
    def __init__(self, image_root, gt_root, normalization="deit"):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        assert len(self.images) == len(self.gts)
        self.size = len(self.images)

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
            raise ValueError("Normalization can only be vit or deit")

        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
    

    def __getitem__(self, index):
        image = self.images[index]
        image = self.img_transform(image) # .unsqueeze(0)???? dunno... (see load_data()) in TestDataset
        gt = self.gts[index]
        gt = self.gt_transform(gt)
        gt = gt/255.0

        # transformed = self.transform(image=image, mask=gt)
        # image = self.img_transform(transformed['image'])
        # gt = self.gt_transform(transformed['mask'])
        return image, gt

    def __len__(self):
        return self.size
        
def get_TestDatasetV2(
    image_root, 
    gt_root, 
    normalization='deit', 
    batch_size = 1, 
    shuffle=True,
):
    dataset = TestDatasetV2(image_root, gt_root, normalization)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader 