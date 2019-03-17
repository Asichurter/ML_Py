# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:52:23 2019

@author: 10904
"""

import random
import torch as t
import matplotlib.pyplot as plt
from torch.optim import SGD

def generate_data(k, b, num, ran=None):
    if ran == None or len(ran) != 2:
        ran = [-20,20]
    data = []
    label = []
    for i in range(num):
        x = random.uniform(ran[0], ran[1])
        y = random.uniform(ran[0], ran[1])
        data.append([x,y])
        label.append(1. if k*x+b >= y else 0.)
    return data,label

def sigmoid(x):
    return 1./(1+t.exp(-x))

def main():
    k = random.uniform(-5,5)
    b = random.uniform(-5,5)
    num = 100
    test_num = 50
    lr = 0.02
    epoches = 100
    
    train_data,train_label = generate_data(k, b, num)
    test_data,test_label = generate_data(k, b, test_num)
    
    W = t.randn((1,2), requires_grad=True)
    B = t.randn((1,1), requires_grad=True)
    
    #预先定义好优化器和损失函数
    opt = SGD([W,B], lr=lr)
    loss_func = t.nn.MSELoss()
    
    for epoch in range(epoches):
        print('epoch ', epoch)
        for data,label in zip(train_data,train_label):
            #建立动态图
            data = t.tensor(data).unsqueeze(0)
            label = t.tensor(label)
            
            linear = W.mm(data.transpose(0,1))+B
            #使用内建函数
            out = t.sigmoid(linear)
            loss = loss_func(out, label)
            loss.backward()
            
            #使用优化器更新，不使用手动更新
            opt.step()
            
            W.grad.zero_()
            B.grad.zero_()
    
    w = W.data.numpy().squeeze()
    bias = B.data.numpy()
    
    bias = -1.*bias[0]/w[1]
    w = -1.*w[0]/w[1]
    
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.plot([x[0] for x,l in zip(train_data,train_label) if l == 1],[x[1] for x,l in zip(train_data,train_label) if l == 1], 'ro')
    plt.plot([x[0] for x,l in zip(train_data,train_label) if l == 0], [x[1] for x,l in zip(train_data,train_label) if l == 0], 'go')
    plt.plot([x[0] for x,l in zip(test_data,test_label) if l == 1],[x[1] for x,l in zip(test_data,test_label) if l == 1], 'rx')
    plt.plot([x[0] for x,l in zip(test_data,test_label) if l == 0], [x[1] for x,l in zip(test_data,test_label) if l == 0], 'gx')
    plt.plot([-20,20], [-20*w+bias, 20*w+bias], 'k')
    
      
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    