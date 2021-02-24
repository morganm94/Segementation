import torchvision.transforms as transforms
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt


class HorizationFlip(object):
    def __init__(self,prob):
        self.prob=prob
    def __call__(self, sample):
        image,mask=sample
        if self.prob>=np.random.uniform(low=0,high=1):
            image=image.transpose(Image.FLIP_LEFT_RIGHT)
            mask=mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image,mask

class VerticalFlip(object):
    def __init__(self,prob):
        self.prob=prob
    def __call__(self, sample):
        image,mask=sample
        if self.prob>=np.random.uniform(low=0,high=1):
            image=image.transpose(Image.FLIP_TOP_BOTTOM)
            mask=mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image,mask

class Rotate(object):
    def __init__(self,angle):
        self.angle=angle
    def __call__(self,sample):
        image,mask=sample
        angle=np.random.uniform(low=-self.angle,high=self.angle)
        image=image.rotate(angle)
        mask =mask.rotate(angle)
        return image,mask

class Scale(object):
    def __init__(self,scale):
        self.scale=scale
    def __call__(self,sample):
        image, mask = sample
        scale=np.random.uniform(low=-self.scale,high=0)
        if scale==0:
            return image,mask
        w, h = image.size
        new_w =int((1+scale)*w)
        new_h = int((1 + scale) * h)
        transform1 = transforms.Resize((new_w, new_h))
        image = transform1(image)
        mask = transform1(mask)
        if scale>0:
            transform2=transforms.CenterCrop((w,h))
        else:
            pad_l=(w-new_w)//2
            pad_r=w-new_w-pad_l
            transform2=transforms.Pad((pad_l,pad_l,pad_r,pad_r))
        image = transform2(image)
        mask = transform2(mask)
        return image, mask

def augmentatonTransforms(scale=None,angle=None,h_flip_prob=None,v_flip_prob=None):
    transform_list=[]
    if scale: transform_list.append(Scale(scale))
    if angle: transform_list.append(Rotate(angle))
    if h_flip_prob: transform_list.append(HorizationFlip(h_flip_prob))
    if v_flip_prob: transform_list.append(VerticalFlip(v_flip_prob))
    return transforms.Compose(transform_list)
