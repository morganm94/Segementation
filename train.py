import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset,random_split
from TwoD_Unet_parts import *
from evaluate import *
from readin import Brain_dataset


device=torch.device('cudn:0' if torch.cuda.is_available() else 'cpu')

EPOCH=25
BATCHSIZE=10
PERCENTAGE=0.8
path='.\dataset\kaggle_3m'
filter=[64,128,256,512,1024]
Learning_rate=1e-3

data=Brain_dataset(path)
train_number=int(data.__len__()*PERCENTAGE)
test_number=data.__len__()-train_number
train_set,test_set=random_split(data,[train_number,test_number])
train_loader=DataLoader(dataset=train_set,batch_size=BATCHSIZE,shuffle=True)
test_loader=DataLoader(dataset=test_set,batch_size=BATCHSIZE)

unet=Unet(3,1,filter)
unet.to(device)
Loss_dsc=Diceloss()
iou=Iou()
dsc=Dicecoeff()
optimizer=torch.optim.Adam(unet.parameters(),lr=Learning_rate)


for epoch in range(EPOCH):
    print('Epoch {}/{}'.format(epoch+1,EPOCH))
    running_loss=[]

    for image,mask in train_loader:
        image=image.to(device,dtype=torch.float)
        mask=mask.to(device,dtype=torch.float)
        prediction=unet(image)
        loss=Loss_dsc(prediction,mask)+CE(prediction,mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())

dsc_list=[]
iou_list=[]
unet.eval()
with torch.no_grad():
    for img,msk in test_loader:
        img=img.to(device,dtype=torch.float)
        msk=msk.to(device,dtype=torch.float)
        pred=unet(img)
        iou_val=iou(img,msk)
        dsc_val=dsc(img,msk)
        dsc_list.append(dsc_val)
        iou_list.append(iou_val)

plt.plot(dsc_list,label='dsc')