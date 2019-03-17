# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:07:53 2019

@author: 10904
"""

import torch as t
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

class Net(t.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.con1 = nn.Conv2d(3, 6, 5)   #3通道，6输出通道，卷积核尺寸为5x5
        self.con2 = nn.Conv2d(6, 16, 5)  
        #32的尺寸在第一次卷积时尺寸减少4为28，然后池化时以2为池宽变为14
        #第二次卷积变为10，最后以此池化变为5
        #即((32-4)/2-4)/2=5
        self.dense1 = nn.Linear(16*5*5, 128)    
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.con1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.con2(x)), 2)
        x = x.view(x.size()[0], -1)      #将2d向量展平为1d向量输入到全连接层中
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        #x = F.softmax(self.dense3(x), dim=0)
        return x
    
epoches = 50
show = ToPILImage()

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse',
           'ship', 'truck']

data_path = r'D:\TSBrowserDownloads'

trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_set = tv.datasets.CIFAR10(data_path, train=True, download=True, transform=trans)
train_loader = t.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)

test_set = tv.datasets.CIFAR10(data_path, train=False, download=True, transform=trans)
test_loader = t.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

net = Net()
net = net.cuda()
loss_func = nn.CrossEntropyLoss().cuda()
opt = t.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

for name,par in net.named_parameters():
    print(name, par.size())

'''
for epoch in range(epoches):
    print('epoch', epoch, ': ')
    running_loss = 0.
    for i,data in enumerate(train_loader):
        if i % 500 == 0:
            print('Training, ',i,' epoch:')
        inputs,labels = data
        inputs,labels = Variable(inputs).cuda(),Variable(labels).cuda()
        
        #因为最终的函数定义为优化器的损失函数，因此使用优化器来进行梯度清空的操作
        opt.zero_grad()
        
        #forward
        out = net(inputs).cuda()
        loss = loss_func(out, labels).cuda()
        #backward
        loss.backward()
        
        opt.step()
        running_loss += loss.data.item()        
        
    correct = 0.
    total = 0.
    print('loss: ', running_loss/len(train_loader))
    for ii,data in enumerate(test_loader):
        if ii % 500 == 0:
            print('Testing, ', ii, 'epoches')
        images,labels = data
        if t.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        output = net(Variable(images))
        _, predict = t.max(output.data, 1)
        total += labels.size()[0]
        correct += (predict==labels).sum().item()
    print(' after ',epoch+1,'epoches, acc: ',correct/total, '\n')
'''
        
        
        
        
        
        
        
        
        
        
        

