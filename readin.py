import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import os
from torch.utils.data import DataLoader,Dataset,random_split
import numpy as np
from augmentation import augmentatonTransforms


class Brain_dataset(Dataset):
    def __init__(self,path,transform=None):
        self.transform=transform
        self.path=path
        self.patients=[file for file in os.listdir(path) if file not in ['data.csv','README.md']]
        self.mask,self.image=[],[] #to store the paths to mask and image

        for patient in self.patients:
            for file in sorted(os.listdir(os.path.join(self.path,patient)),key=lambda x:int(x.split(".")[-2].split("_")[4])):
                if 'mask' in file.split('.')[0].split('_'):
                    self.mask.append(os.path.join(self.path,patient,file))
                else:
                    self.image.append(os.path.join(self.path,patient,file))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path=self.image[idx]
        mask_path=self.mask[idx]
        image=Image.open(image_path)
        mask=Image.open(mask_path)
        if self.transform:
            image,mask=self.transform((image,mask))
        mask=transforms.ToTensor()(mask)
        image=transforms.ToTensor()(image)
        return image,mask




