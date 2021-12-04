import torch
import torch.nn as nn
import numpy as np

from Focus import Focus
from BaseConv import BaseConv
from CSPNet import CSPNet
from SPPNet import SPPNet

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
        self.sppnet = SPPNet(512)
        self.conv_down4 = BaseConv(512*4,512,kernel_size=1,stride=1)

        self.conv_down_l = BaseConv(512,256,kernel_size=1,stride=1)
        self.upsample_l = nn.Upsample(scale_factor=2) # 7 * 2
        self.cspnet_m = CSPNet(512,4)
        self.conv_down_m = BaseConv(512,256,kernel_size=1,stride=1)

        self.conv_down_m2s = BaseConv(256, 128, kernel_size=1, stride=1)
        self.upsample_s = nn.Upsample(scale_factor=2)  # 7 * 2
        self.cspnet_s = CSPNet(256, 2)
        self.conv_down_s = BaseConv(256, 128, kernel_size=1, stride=1)

        self.output_l = nn.Conv2d(512, self.number_output * self.anchor_length, 1)
        self.output_m = nn.Conv2d(256, self.number_output * self.anchor_length, 1)
        self.output_s = nn.Conv2d(128, self.number_output * self.anchor_length, 1)



    def forward(self,x):

        ##Build Backbone
        x = self.focus(x)
        x = self.conv1(x)
        x = self.cspnet1(x)
        out_s = self.conv_down1(x)
        x = self.cspnet2(out_s)
        out_m = self.conv_down2(x)
        x = self.cspnet3(out_m)
        out_l = self.conv_down3(x)
        x = self.sppnet(out_l)
        out_l = self.conv_down4(x)

        ##Build head
        #out_l: (b,512,20,20) -> (b,num_ouput*num_anchor,20,20)
        output_l = self.output_l(out_l)

        #out_m: out_l 20*20 -> 40*40 -> cat[256,256] -> cspnet -> con1*1 [256]
        out_l_up = self.conv_down_l(out_l)
        out_l_up = self.upsample_l(out_l_up)
        out_m = torch.cat([out_l_up,out_m],dim=1)
        out_m = self.cspnet_m(out_m)
        out_m = self.conv_down_m(out_m)
        output_m = self.output_m(out_m)

        # out_s: out_m 40*40 -> 80*80 -> cat[128,128] -> cspnet -> con1*1 [128]
        out_s_up = self.conv_down_m2s(out_m)
        out_s_up = self.upsample_s(out_s_up)
        out_s = torch.cat([out_s_up, out_s], dim=1)
        out_s = self.cspnet_s(out_s)
        out_s = self.conv_down_s(out_s)
        output_s = self.output_s(out_s)

        return output_l,output_m,output_s

if __name__ == '__main__':

    anchors = torch.tensor([[4,6,5,12,8,8],[13,12,8,20,13,31],[32,20,18,42,28,59]])
    # print(anchors.shape)
    number_class = 22
    channel = 3
    input = torch.randn(1,3,640,640)
    model = Model(anchors = anchors,number_class = number_class,channel=3)
    total_params = sum(p.numel() for p in model.parameters())
    print("model total parameters: {}".format(total_params))
    out_s,out_m,out_l = model(input)
    print(out_s.size(),out_m.size(),out_l.size())