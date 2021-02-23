import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self,in_chan,ou_chan,ker_size=3,padding=1,numgroup=1):
        super().__init__()
        # self.in_chan=in_chan
        # self.ou_chan=ou_chan
        # self.ker_size=ker_size
        # self.padding=padding
        self.block=nn.Sequential(
            nn.GroupNorm(numgroup,in_chan),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=in_chan,out_channels=ou_chan,kernel_size=ker_size,padding=padding),
            nn.GroupNorm(numgroup, in_chan),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=in_chan, out_channels=ou_chan, kernel_size=ker_size, padding=padding)
        )#在2018年的论文中，一个block里的channel数量是不变的
    def forward(self,x):
        result1=self.block(x)
        output=result1+x
        return output

class conv_block2(nn.Module):
    def __init__(self,in_chan,ou_chan,ker_size=3,padding=1,numgroup=1):
        super().__init__()
        self.block=nn.Sequential(
            conv_block(in_chan=in_chan,ou_chan=ou_chan),
            conv_block(in_chan=in_chan, ou_chan=ou_chan),
        )
    def forward(self,x):
        x=self.block(x)
        return x

class conv_block4(nn.Module):
    def __init__(self,in_chan,ou_chan,ker_size=3,padding=1,numgroup=1):
        super().__init__()
        self.block=nn.Sequential(
            conv_block2(in_chan=in_chan, ou_chan=ou_chan),
            conv_block2(in_chan=in_chan, ou_chan=ou_chan),
        )
    def forward(self,x):
        x=self.block(x)
        return x

class up(nn.Module):
    def __init__(self,in_can,factor=2):
        super().__init__()
        self.conv1=nn.Conv3d(in_channels=in_can,out_channels=in_can//factor,kernel_size=1,padding=0,stride=1)#降维
        self.up=nn.Upsample(scale_factor=factor,mode='trilinear')
    def forward(self,x):
        x=self.conv1(x)
        x=self.up(x)
        return x



class Unet(nn.Module):
    def __init__(self,filter:list,in_chan,padding=1,ker_size=3):
        super(Unet,self).__init__()
        #encoder部分
        self.conv1=nn.Conv3d(in_chan,filter[0],padding=padding,kernel_size=ker_size)
        self.block1=conv_block(filter[0],filter[0])
        self.down1=nn.Conv3d(filter[0],filter[1],kernel_size=3,stride=2,padding=1)
        self.block2=conv_block2(filter[1],filter[1])
        self.down2=nn.Conv3d(filter[1],filter[2],kernel_size=3,stride=2,padding=1)
        self.block3=conv_block2(filter[2],filter[2])
        self.down3 = nn.Conv3d(filter[2], filter[3], kernel_size=3, stride=2,padding=1)
        self.block4 = conv_block4(filter[3], filter[3])

    #decoder部分
        self.up1=up(filter[3])
        self.block5=conv_block(filter[2],filter[2])
        self.up2= up(filter[2])
        self.block6 = conv_block(filter[1], filter[1])
        self.up3= up(filter[1])
        self.block7 = conv_block(filter[0], filter[0])
        self.conv2=nn.Conv3d(filter[0],3,kernel_size=1)

    def forward(self,x):
    #encoder部分
        x=self.conv1(x)
        print(x.size())
        x1=self.block1(x)
        x2=self.down1(x1)
        print(x2.size())
        x3 = self.block2(x2)
        x4 = self.down2(x3)
        print(x4.size())
        x5 = self.block3(x4)
        x6 = self.down3(x5)
        print(x6.size())
        x7 = self.block4(x6)
    #decoder部分
        x8=self.up1(x7)
        x8+=x5
        print(x8.size())
        x9=self.up2(x8)
        x9+=x3
        print(x9.size())
        x10=self.up3(x9)
        x10+=x1
        print(x10.size())
        result=self.conv2(x10)
        print(result.size())
        return result




filter=[32,64,128,256]
input=torch.randn((1,1,96,88,24))
unet=Unet(filter,1)
unet(input)


