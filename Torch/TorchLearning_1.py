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

vis = visdom.Visdom()
xx = t.Tensor(np.linspace(0,10,30))
yy = t.cos(xx)
vis.line(X=xx, Y=yy, win='cosx', opts={'title':'cos(x)'})
vis.images(t.randn((2,3,128,128)), win='image')

a = [1,2,3]
print(list(map(lambda x: x+1, a)))