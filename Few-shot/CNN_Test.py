# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:04:41 2019

@author: 10904
"""
import numpy as np
from utils import check_continuing_decrease
import torch as t
#import torch.nn.functional as F
from ResNetForMalwareImage import ResNet
from dataUtils import CNNTestDataset, DirDataset#, PretrainedResnetDataset
from torch.utils.data import DataLoader
import torchvision as tv
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

early_stop = False
early_stop_window = 3
save_path = 'D:/ML_Py/Few-shot/doc/基于resnet18的非缺省windows实验/'

'''
def get_pretrained_resnet():
    resnet = tv.models.resnet18(pretrained=True)
    #for name,layer in resnet.named_modules():
     #   print(name, ' ', layer)
    for par in resnet.parameters():
        par.requires_grad = False
    resnet.fc = t.nn.Linear(512, 2)
    parameters = []
    for name,par in resnet.named_parameters():
        if par.requires_grad:
            print(name, par)
            parameters.append(par)
    return resnet,parameters'''

def validate(model, dataloader, Criteria):
    '''
    使用指定的dataloader验证模型\n
    model:训练的模型\n
    dataloader:验证的数据加载器\n
    criteria:损失函数\n
    '''
    val_a = 0
    val_c = 0
    val_loss = 0.
    #将模型调整为测试状态
    model.eval()
    for data,label in dataloader:
        data = data.cuda()
        out = model(data)
        
        #同训练阶段一样
        labels = [[1,0] if L==0 else [0,1] for L in label]
        labels = t.FloatTensor(labels).cuda()
        
        loss = Criteria(out, labels)
        val_loss += loss.data.item()
        pre_label = t.LongTensor([0 if x[0]>=x[1] else 1 for x in out])
        val_a += pre_label.shape[0]
        val_c += (pre_label==label).sum().item()
    return val_c/val_a,val_loss

#最大迭代次数    
MAX_ITER = 30
#记录历史的训练和验证数据
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

#测试集数据集
dataset = DirDataset(r'D:/peimages/test for cnn/no padding/')
#dataset = t.load('datas/train_dataset.tds')
#dataset = PretrainedResnetDataset(r'D:/peimages/test for cnn/no padding/')
#验证集数据集
val_set = DirDataset(r'D:/peimages/validate/')
#val_set = t.load('datas/val_dataset.tds')
#t.save(val_set, 'val_dataset.tds')
#val_set = PretrainedResnetDataset(r'D:/peimages/validate/')
#训练集数据加载器
train_loader = DataLoader(dataset, batch_size=48, shuffle=True)
#验证集数据加载器
test_loader = DataLoader(val_set, batch_size=16, shuffle=False)
resnet = ResNet(1)
#resnet,pars = get_pretrained_resnet()
resnet = resnet.cuda()
#for par in pars:
#    par = par.cuda()
#opt = t.optim.SGD(pars, lr=1e-2, momentum=0.9, weight_decay=0.2, nesterov=True)
#根据resnet的论文，使用1e-4的权重衰竭
opt = t.optim.Adam(resnet.parameters(), lr=1e-3, weight_decay=1e-4)
#使用二元交叉熵为损失函数（可以替换为交叉熵损失函数）
criteria = t.nn.BCELoss()
#学习率调整器，使用的是按照指标的变化进行调整的调整器
scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=0, verbose=True, min_lr=1e-5)
#criteria = t.nn.CrossEntropyLoss()
num = 0
best_val_loss = 0.
print('training...')
for i in range(MAX_ITER):
    print(i, ' th')
    a = 0
    c = 0
    Loss = 0.
    
    #将模型调整为学习状态
    resnet.train()
    for datas,l in train_loader:
        opt.zero_grad()
        datas = datas.cuda()
        
        #创建可以输入到损失函数的float类型标签batch
        labels = [[1,0] if L==0 else [0,1] for L in l]
        labels = t.FloatTensor(labels).cuda()

        out = resnet(datas).squeeze()
        loss = criteria(out, labels).cuda()
        loss.backward()
        opt.step()
        
        #计算损失和准确率
        Loss += loss.data.item()
        #进行与实际标签的比较时，由于标签是LongTensor类型，因此转化
        #选用值高的一个作为预测结果
        predict = t.LongTensor([0 if x[0]>=x[1] else 1 for x in out])
        a += predict.shape[0]
        c += (predict==l).sum().item()
    print('train loss: ', Loss)
    train_loss_history.append(Loss)
    print('train acc: ', c/a)
    train_acc_history.append(c/a)
        
    val_acc,val_loss = validate(resnet, test_loader, criteria)
    print('val loss: ', val_loss)
    val_loss_history.append(val_loss)
    print('val accL: ', val_acc)
    val_acc_history.append(val_acc)
    
    if len(val_loss_history)==1 or val_loss < best_val_loss:
        best_val_loss = val_loss
        t.save(resnet, save_path+'best_loss_model.h5')
        print('save model at epoch %d'%i)
    
    num += 1
    #使用学习率调节器来随验证损失来调整学习率
    scheduler.step(val_loss)
    #检测是否可以提前终止学习
    if early_stop and check_continuing_decrease(val_acc_history, early_stop_window):
        break

#根据历史值画出准确率和损失值曲线    
x = [i for i in range(num)]
plt.title('Accuracy')
plt.plot(x, train_acc_history, linestyle='--', label='train')
plt.plot(x, val_acc_history, linestyle='-', label='validate')
plt.legend()
plt.savefig(save_path+'acc.png')
plt.show()


plt.title('Loss')
plt.plot(x, train_loss_history, linestyle='--', label='train')
plt.plot(x, val_loss_history, linestyle='-', label='validate')
plt.legend()
plt.savefig(save_path+'loss.png')
plt.show()

ACC = np.array(val_acc_history)
LOSS = np.array(val_loss_history)
np.save(save_path+'acc.npy', ACC)
np.save(save_path+'loss.npy', LOSS)

