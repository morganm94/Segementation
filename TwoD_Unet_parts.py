import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_block_2D(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.block(x)

class Enconder_block(nn.Module):
    def __init__(self,in_channel,out_channel,mid_channel=None):
        super().__init__()
        if not mid_channel:
            mid_channel=out_channel
        self.block=nn.Sequential(
            Conv_block_2D(in_channel,mid_channel),
            Conv_block_2D(mid_channel,out_channel)
        )
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        out1=self.block(x) # for cantaneing
        result=self.maxpool(out1)
        return  result,out1

class Decoder_block(nn.Module):
    def __init__(self,in_channel,down_channel,out_channel):
        super().__init__()
        assert in_channel>1,'in_channel//2 is 0 /decoder_block'
        self.up=nn.ConvTranspose2d(in_channel,in_channel//2,kernel_size=2,stride=2)
        self.block=nn.Sequential(
            Conv_block_2D(in_channel//2+down_channel,out_channel),
            Conv_block_2D(out_channel,out_channel),
        )
    def forward(self,x,down_featuremap):
        _,channels,height,width=down_featuremap.size()
        x=self.up(x)
        diff_y=x.size()[2]-down_featuremap.size()[2]
        diff_x=x.size()[3]-down_featuremap.size()[3]
        x=F.pad(x,[diff_x//2,diff_x-diff_x//2,diff_y//2,diff_y-diff_y//2])
        x_mid=torch.cat([x,down_featuremap],dim=1)
        return self.block(x_mid)

class Unet(nn.Module):
    def __init__(self,in_channel,out_channel,filter:list):
        super().__init__()
        self.encoder1=Enconder_block(in_channel,out_channel=filter[0])
        self.encoder2=Enconder_block(filter[0],out_channel=filter[1])
        self.encoder3=Enconder_block(filter[1],filter[2])
        self.encoder4 = Enconder_block(filter[2], filter[3])
        self.encoder5 = nn.Sequential(
            Conv_block_2D(filter[3],filter[4]),
            Conv_block_2D(filter[4], filter[4])#filter[4]是1024
        )

        self.decoder1=Decoder_block(filter[4],filter[3],filter[3])
        self.decoder2 = Decoder_block(filter[3], filter[2], filter[2])
        self.decoder3 = Decoder_block(filter[2], filter[1], filter[1])
        self.decoder4 = Decoder_block(filter[1], filter[0], filter[0])
        self.out=Conv_block_2D(filter[0],out_channel)

    def forward(self,x):
        x,out1=self.encoder1(x)     #x: the dimension/2 , out1: the same dimension
        x, out2 = self.encoder2(x)  #out:the same dimension as the last x
        x, out3 = self.encoder3(x)
        x, out4 = self.encoder4(x)
        x=self.encoder5(x)
        x=self.decoder1(x,out4)
        x = self.decoder2(x, out3)
        x = self.decoder3(x, out2)
        x = self.decoder4(x, out1)
        x=self.out(x)
        x = torch.sigmoid(x)
        return x


filter=[64,128,256,512,1024]
unet=Unet(3,1,filter)
input=torch.randn(2,3,256,256)
output=unet(input)
print(output.size())


