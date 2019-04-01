# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:51:41 2019

@author: 10904
"""

import torch as t
from torch.autograd import Variable
import os
import visdom
import numpy as np
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

a = t.Tensor([[1,2],[3,4]])
b = t.Tensor([0,0,0,0]).byte()
a = a.view(1, -1).squeeze()
print(a)
print(b)

x = t.tensor([[1,2,3],[4,5,6],[7,8,9]])
