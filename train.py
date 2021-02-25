import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset,random_split
from TwoD_Unet_parts import *
from evaluate import *
from readin import Brain_dataset


device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

EPOCH=300
BATCHSIZE=64
PERCENTAGE=0.8
NUM_WORKERS=4
path='.\dataset\kaggle_3m'
filter=[32,64,128,256,512]
Learning_rate=1e-3

data=Brain_dataset(path)
train_number=int(data.__len__()*PERCENTAGE)
test_number=data.__len__()-train_number
train_set,test_set=random_split(data,[train_number,test_number])
train_loader=DataLoader(dataset=train_set,batch_size=BATCHSIZE,shuffle=True,num_workers=NUM_WORKERS)
test_loader=DataLoader(dataset=test_set,batch_size=BATCHSIZE)

unet=Unet(3,1,filter)
unet.to(device,dtype=torch.float)
Loss_dsc=Diceloss()
iou=Iou()
dsc=Dicecoeff()
optimizer=torch.optim.Adam(unet.parameters(),lr=Learning_rate)


for epoch in range(EPOCH):
    print('EPOCH:{}/{}'.format(epoch+1,EPOCH))
    history_loss={'train':[],'test':[]}
    history_dsc={'train':[],'test':[]}
    for phase in ['train','test']:
        running_loss = 0.0
        running_dsc = 0.0
        if phase=='train':
            unet.train()
            now_loader=train_loader
            number=train_number
        else:
            unet.eval()
            now_loader = test_loader
            number = test_number
        for image, mask in now_loader:
            image = image.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float)
            with torch.set_grad_enabled(phase=='train'):
                prediction = unet(image)
                loss = Loss_dsc(prediction, mask)#try one loss firstly,
                                            #in the seconde stage,i will attempt to
                                            #add one more loss:CE
            if phase=='train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss+=loss.item()*image.size(0)
            running_dsc+=dsc(prediction,mask)*image.size(0)

        epoch_loss=running_loss/number
        epoch_dsc=running_dsc/number
        print('{} loss:{}'.format(phase,epoch_loss))
        print('{} dsc:{}'.format(phase, epoch_dsc))
        history_loss[phase].append(epoch_loss)
        history_dsc[phase].append(epoch_dsc)
