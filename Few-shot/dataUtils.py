# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:20:11 2019

@author: 10904
"""

import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

class CNNTestDataset(Dataset):
    '''
    用于测试CNN结构的数据集
    '''
    def __init__(self, base=r'D:/peimages/test for cnn/', using_padding=False, transforms=None):
        data = []
        label = []
        if using_padding:
            assert 'padding' in os.listdir(base), 'padding文件夹不在指定的目录下!'
            dir_name = 'padding/'
        else:
            assert 'no padding' in os.listdir(base), 'no padding文件夹不在指定的目录下!'
            dir_name = 'no padding/'
        #该目录下只有两个文件夹:malware和benign
        for child_dir in os.listdir(base+dir_name):
            path = base+dir_name+child_dir+'/'
            columns = os.listdir(path)
            data += [os.path.join(path,column) for column in columns]
            #添加样本数量个对应的标签
            label += [1 if child_dir=='malware' else 0 for i in range(len(columns))]
            #print(child_dir,':',len(columns))
        assert len(data)==len(label),'数据与标签的数量不一致!'
        self.Datas = data
        self.Labels = label
        #假设图像是单通道的
        #归一化到[-1,1]之间
        if transforms:
            self.Transform = transforms
        else:
            self.Transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    
    def __getitem__(self, index):
        path = self.Datas[index]
        image = Image.open(path)
        image = self.Transform(image)
        return image,self.Labels[index]
    
    def __len__(self):
        return len(self.Datas)

if __name__ =='__main__':
    dataset = CNNTestDataset()