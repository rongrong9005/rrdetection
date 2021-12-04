import torch
import torch.nn as nn
from BaseConv import BaseConv


class SPPNet(nn.Module):
    def __init__(self,channels,pooling_list=(5,9,13)):
        super(SPPNet, self).__init__()
        self.pooling_list = pooling_list
        self.maxpooling = nn.ModuleList([nn.MaxPool2d(pooling,1,pooling//2) for pooling in pooling_list])

    def forward(self,x):
        return torch.cat([x]+[self.maxpooling[0](x),self.maxpooling[1](x),self.maxpooling[2](x)],dim=1)



if __name__ == '__main__':
    input = torch.randn(1,1024,20,20)
    model = SPPNet(1024)
    print(model(input).size())