import torch
import torch.nn as nn
import numpy as np
from Focus import Focus
from BaseConv import BaseConv
from CSPNet import CSPNet

class Model(nn.Module):
    def __init__(self,anchors,number_class,channel):
        super(Model, self).__init__()
        self.number_class = number_class
        self.number_output = number_class+5 #weight height center_x center_y type
        self.anchor_length = anchors.shape[0]
        self.number_anchor = anchors.shape[1]
        self.channel = channel

        self.focus = Focus(self.channel,2)
        self.conv1 = BaseConv(32,64,kernel_size=3,stride=2)
        self.cspnet1 = CSPNet(64,2)
        self.conv_down1 = BaseConv(64,128,kernel_size=3,stride=2)
        self.cspnet2 = CSPNet(128,4)
        self.conv_down2 = BaseConv(128,256,kernel_size=3,stride=2)
        self.cspnet3 = CSPNet(256, 8)
        self.conv_down3 = BaseConv(256, 512, kernel_size=3, stride=2)

    def forward(self,x):
        x = self.focus(x)
        x = self.conv1(x)
        x = self.cspnet1(x)
        x = self.conv_down1(x)
        x = self.cspnet2(x)
        x = self.conv_down2(x)
        x = self.cspnet3(x)
        x = self.conv_down3(x)

        return x

if __name__ == '__main__':

    anchors = torch.tensor([[4,6,5,12,8,8],[13,12,8,20,13,31],[32,20,18,42,28,59]])
    # print(anchors.shape)
    number_class = 22
    channel = 3
    input = torch.randn(1,3,640,640)
    model = Model(anchors = anchors,number_class = number_class,channel=3)
    # print(model)
    print(model(input).size())