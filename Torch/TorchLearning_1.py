# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:51:41 2019

@author: 10904
"""

import torch as t
from torch.autograd import Variable
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

a,b,c = 3,4,5
x = Variable(t.randn(a,b))
w11111 = Variable(t.randn(b,c))
w2 = Variable(t.randn(b,c))

z = 10
y = None
if z > 0:
    y = x.mm(w11111)
else:
    y = x.mm(w2)
print(t.mm(x,w11111))
print(y)
device = t.cuda.device(0)
#print(os.environ.keys)
t.cuda.set_device(0)

print(t.cuda.is_available())