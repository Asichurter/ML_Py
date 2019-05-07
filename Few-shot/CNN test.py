# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:45:52 2019

@author: 10904
"""

import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataUtils import CNNTestDataset
from torch.nn.modules import Conv2d, BatchNorm2d, ReLU

class ResBlock(t.nn.Module):
    def __init__(self, in_size, out_size, stride=1, kernel_size=3, shortcut=None):
        super(ResBlock, self)
        self.Left = t.nn.Sequential([
                Conv2d(in_size, out_size, kernel_size, stride=stride, padding=(kernel_size-1)/2, bias=False),
                BatchNorm2d(out_size),
                ReLU(),
                Conv2d(out_size, out_size, kernel_size, 1, padding=(kernel_size-1)/2, bias=False),
                BatchNorm2d])
        
        self.Right = shortcut
        
    def forward(self, x):
        left = self.Left(x)
        right = self.Right(x) if self.Right is not None else x
        #由于left在整个过程中没有经过池化而且padding均为same，因此形状理应该一样
        assert left.shape==right.shape, '残差块内左右两部分的形状不一样无法相加'
        return F.relu(left+right)
    
#class ResLayer(t.nn.Module):
    
        