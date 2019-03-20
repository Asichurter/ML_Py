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

a = t.tensor([1.,1.], requires_grad=True)
for i in range(2):
    b = (a**2).sum()
    b.backward()
    print('第', i+1, '次a的梯度: ', a.grad)
    a.grad.zero_()

inputs = t.randn(3, 5, requires_grad=True)
target = t.randint(5, (3,), dtype=t.int64)
loss = F.cross_entropy(inputs, target)
print(inputs)
print(target)

aa = t.Tensor([[0.1, -0.1]])
bb = t.LongTensor([0])
print(t.nn.CrossEntropyLoss()(aa, bb))


