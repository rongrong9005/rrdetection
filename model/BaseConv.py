import torch
import torch.nn as nn


class BaseConv(nn.Module):
    def __init__(self,channel_in,channel_out,kernel_size,stride=1):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(channel_in,channel_out,kernel_size,stride,kernel_size//2,bias=False)
        self.bn = nn.BatchNorm2d(channel_out)
        self.activate = nn.ReLU()

    def forward(self,x):
        return self.activate(self.bn(self.conv(x)))


if __name__ == '__main__':
    input = torch.randn(1,3,640,640)
    model = BaseConv(3,3,3,2)
    print(model(input).size())