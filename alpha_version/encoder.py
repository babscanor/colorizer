import torch.nn as nn
import torch.nn.functional as F
import torch

from fusion import Fusion

class Network(nn.Module):
    def __init__(self,batch_size):
        super(Network,self).__init__()
        self.fusion = Fusion(batch_size)
        #encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        #decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size= 1, padding=0),
            nn.Upsample(scale_factor=1, mode='bilinear'),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=1, mode='bilinear'),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32,out_channels=2,kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=1, mode='bilinear')
        )

    def forward(self, gray, color):
        gray = self.encoder(gray)
        color = self.fusion.reshape(color)  
        t = torch.cat((gray, color),dim=1)  
        t = self.decoder(t)
        return t


#net = Network(10)
#t = torch.rand((10,1,256,256))
#t = net.forward(t)
