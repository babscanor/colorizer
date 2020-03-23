import torch.nn as nn
import torch.nn.functional as F
import torch

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=2, padding=129)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=2, padding=129)
        self.conv5 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2, padding=129)
        self.conv7 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=1, mode='bilinear')
        self.conv10 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=1, mode='bilinear')
        self.conv11 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=32,out_channels=2,kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=1, mode='bilinear')

    def forward(self, t):
        t = t 
        # hidden layer
        t = self.conv1(t)
        t = F.relu(t)
        # hidden layer
        t = self.conv2(t)
        t = F.relu(t)
        # hidden linear layer
        t = self.conv3(t)
        t = F.relu(t)
        # hidden linear layer
        t = self.conv4(t)
        t = F.relu(t)
        # hidden linear layer
        t = self.conv5(t)
        t = F.relu(t)
        # hidden linear layer
        t = self.conv6(t)
        t = F.relu(t)
        # hidden linear layer
        t = self.conv7(t)
        t = F.relu(t)
        # hidden linear layer
        t = self.conv8(t)
        t = F.relu(t)
        # hidden linear layer
        t = self.conv9(t)
        t = F.relu(t)
        #upsample
        t = self.upsample1(t)
        # hidden linear layer
        t = self.conv10(t)
        t = F.relu(t)
        #upsample
        t = self.upsample2(t)
        # hidden linear layer
        t = self.conv11(t)
        t = F.relu(t)
        # hidden linear layer
        t = self.conv12(t)
        t = torch.tanh(t)  
        #upsample
        t = self.upsample3(t)
        return t


# net = Network()
# t = torch.rand((3,1,256,256))
# t = net(t)
# print(t)