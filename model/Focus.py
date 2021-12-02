import torch
import torch.nn as nn


class Focus(nn.Module):
    def __init__(self,channel,width):
        super(Focus, self).__init__()
        self.ch_in = channel*4
        self.ch_out = 64 // width

        self.conv = nn.Conv2d(in_channels = self.ch_in, out_channels = self.ch_out,kernel_size = (1,1),stride = (1,1))
        self.bn = nn.BatchNorm2d(self.ch_out)
        self.activate = nn.ReLU()

    def forward(self,x):
        return self.activate(self.bn(self.conv(torch.cat([x[...,::2,::2],x[...,1::2,::2],x[...,::2,1::2],x[...,1::2,1::2]],1))))


if __name__ == '__main__':
    x = torch.randn(1,3,640,640)
    print(torch.cat([x[...,::2,::2],x[...,1::2,::2],x[...,::2,1::2],x[...,1::2,1::2]],1).size())
    focus = Focus(channel=3,width=2)
    out = focus(x)
    print(out.size())
