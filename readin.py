import torch
from skimage.io import imread,imshow
import PIL.Image as Image
import os
from torch.utils.data import DataLoader,Dataset,random_split
import numpy as np

class Brain_dataset(Dataset):
    def __init__(self,path):
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
        image=imread(image_path)
        mask=imread(mask_path,as_gray=True)

        mask=np.expand_dims(mask,axis=-1)
        mask=mask.transpose((2,0,1))
        mask=mask/255

        image=image/255
        image=image.transpose((2,0,1))
        image=torch.from_numpy(image)
        mask=torch.from_numpy(mask)
        return image,mask




