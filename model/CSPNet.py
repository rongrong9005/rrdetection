import torch
import torch.nn as nn
from BaseConv import BaseConv

class Resblock(nn.Module):
    def __init__(self,channels,inner_channels):
        super(Resblock, self).__init__()
        if inner_channels is None:
            inner_channels = channels
        self.conv1 = BaseConv(channels,inner_channels,1,1)
        self.conv2 = nn.Conv2d(inner_channels,channels,3,1,1)
        self.bn = nn.BatchNorm2d(channels)
        self.activate = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        x = self.bn(out) + x
        return self.activate(x)


class CSPNet(nn.Module):
    def __init__(self,channels,num_block):
        super(CSPNet, self).__init__()
        self.conv1 = BaseConv(channels,channels,1,1)
        self.resblock = nn.Sequential(*[Resblock(channels,channels//2) for _ in range(num_block)])
        self.conv_cat = BaseConv(channels*2,channels,1,1)

    def forward(self,x):
        out1 = self.conv1(x)
        out2 = self.conv1(x)
        out2 = self.resblock(out2)
        out = torch.cat([out1,out2],1)
        out = self.conv_cat(out)
        return out


if __name__ == '__main__':
    input = torch.randn(1,32,320,320)
    model = CSPNet(32,8)
    print(model(input).size())