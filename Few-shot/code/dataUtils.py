# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:20:11 2019

@author: 10904
"""

import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import torch as t

#文件夹数据集
#目录下有benign和malware两个文件夹
class DirDataset(Dataset):
    def __init__(self, base, transforms=None):
        data = []
        label = []
        for child_dir in os.listdir(base):
            path = base+child_dir+'/'
            columns = os.listdir(path)
            data += [os.path.join(path,column) for column in columns]
            #添加样本数量个对应的标签
            label += [1 if child_dir=='malware' else 0 for i in range(len(columns))]
            assert len(data)==len(label),'数据与标签的数量不一致!'
            #print(child_dir,':',len(columns))
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
        if self.Transform is not None:
            image = self.Transform(image)
        return image,self.Labels[index]
    
    def __len__(self):
        return len(self.Datas)

#默认定位到test for cnn，文件夹下有是否padding两个文件夹，之下才是benign和malware
class CNNTestDataset(DirDataset):
    '''
    用于测试CNN结构的数据集
    '''
    def __init__(self, base=r'D:/peimages/test for cnn/', using_padding=False, transforms=None):
        if using_padding:
            assert 'padding' in os.listdir(base), 'padding文件夹不在指定的目录下!'
            dir_name = 'padding/'
        else:
            assert 'no padding' in os.listdir(base), 'no padding文件夹不在指定的目录下!'
            dir_name = 'no padding/'
        path = base + dir_name
        super().__init__(path, transforms)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        return super().__len__()

#torchvision自带的resnet适用的数据集
#需要将图像转为224x224，同时还需要将单通道转为3通道
class PretrainedResnetDataset(DirDataset):
    def __init__(self, base):
        #将图片尺寸调整为resnet的224x224尺寸，同时转化为三通道的图像，再标准化
        transforms = T.Compose([T.Grayscale(num_output_channels=3), 
                                     T.Resize(224), 
                                     T.ToTensor(), 
                                     T.Normalize([0.5], [0.5])])
        super().__init__(base, transforms)
        
    def __getitem__(self, index):
        #先得到的图像不能转变为向量，因为需要
        im,label = super().__getitem__(index)
        return im,label
        #return im,label
    
    def __len__(self):
        return super().__len__()

if __name__ =='__main__':
    pass
    #dataset = CNNTestDataset()
    #dataset = PretrainedResnetDataset(r'D:/peimages/validate/')