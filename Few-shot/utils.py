# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:09:04 2019

@author: 10904
"""
import numpy as np
import PIL.Image as Image 
import os
from extract import extract_infos
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

HOME = r'C:/Users/10904/Desktop/images2/'
BASE = r'D:/pe/trojan0/'
PATH = [r'/DoS.Win32.Adonai.01', r'/DoS.Win32.Agent.m']
EXES = ['exe', 'dll', 'ocx', 'sys', 'com']
BENIGN_BASE = r'C:/Windows/'
MALWARE_BASE = r'D:/pe/'
TEST_CHILD_DIR = ['backdoor1/', 'net-worm/']
TEST_NUM = 1000
SIZE_RANGE = [15, 3000]
DATA_SAVE_NAME = r'data_0504.npy'
LABEL_SAVE_NAME = r'label_0504.npy'
DATA_SPLIT = 0.8

WIDTH = 256
WIDTH_SIZE = 10
UNIT = 1/25

#base:目标文件或者目标所在的文件夹
#destination:转换后存储的文件夹
#mode:转换的模式：单个文件还是该文件夹下所有的文件
def convert_to_images(base, destination=HOME, mode='file', num_constrain=200):
    if destination[-1] != '/':
        destination += '/'
    if mode == 'dir':
        if not os.path.isdir(base):
            raise Exception(base + ' is not a director!\n')
        files = os.listdir(base)
        for i,one in enumerate(files):
            if i > num_constrain:
                break
            else:
                print(i)
            file = open(base+one, "rb")
            image = np.fromfile(file, dtype=np.byte)
            #将不足宽度大小的剩余长度的像素点都过滤掉
            if image.shape[0]%WIDTH != 0:
                image = image[:-(image.shape[0]%WIDTH)]
            #print(image.shape)
            image = image.reshape((-1, WIDTH))
            image = np.uint8(image)
            im = Image.fromarray(image)
            im.save(destination+one+'.jpg', 'JPEG')
            file.close()
            
    elif mode == 'file':
        if os.path.isdir(base):
            raise Exception(base + ' is indeed a directory!\n')
        file = open(base, 'rb')
        image = np.fromfile(file, dtype=np.byte)
        if image.shape[0]%WIDTH != 0:
            image = image[:-(image.shape[0]%WIDTH)]
        #print(image.shape)
        image = image.reshape((-1, WIDTH))
        image = np.uint8(image)
        im = Image.fromarray(image)
        im.save(destination+one+'.jpg', 'JPEG')
        file.close()
        
        
#检查一个地址的文件扩展名是否是可执行文件
def check_if_executable(path, size_thre=SIZE_RANGE):
    #需要去掉最后一个斜杠/
    extension_name = path[:-1].split('.')[-1]
    #除以1024单位为千字节KB
    size = int(os.path.getsize(path[:-1])/1024)
    #只有是pe文件且大小在范围之内的文件的绝对路径才会被返回
    return extension_name in EXES and size >= size_thre[0] and size <= size_thre[1]

#在windows目录下查找所有可执行文件的目录
#本函数必须在有管理员权限下才能使用      
def get_benign_exe_abspath(base=BENIGN_BASE):
    if os.path.isdir(base):
        for dirs in os.listdir(base):
            #加上斜杠保证以后的递归能继续在文件夹中进行
            for ele in get_benign_exe_abspath(base+dirs+'/'):
                if check_if_executable(ele):
                    yield ele
    else:
        if check_if_executable(base):
            yield base

def mix_samples(mal_base=MALWARE_BASE, num=500, split=0.5, seed=1):
    my_num = 0
    data = []
    label = []
    benign =  get_benign_exe_abspath()
    for mal_name in os.listdir(str(mal_base+TEST_CHILD_DIR[0])):
        print('A: ', my_num)
        pe_data = extract_infos(mal_base+TEST_CHILD_DIR[0]+mal_name)
        if pe_data is None:
            continue
        data.append(pe_data)
        label.append(1)
        my_num += 1
        if my_num == num:
            break
    my_num = 0
    for mal_name in os.listdir(mal_base+TEST_CHILD_DIR[1]):
        print('B: ', my_num)
        pe_data = extract_infos(mal_base+TEST_CHILD_DIR[1]+mal_name)
        if pe_data is None:
            continue
        data.append(pe_data)
        label.append(1)
        my_num += 1
        if my_num == num:
            break
    for i in range(my_num):
        try:
            print('C: ', i)
            benign_base = next(benign)[:-1]
            data.append(extract_infos(benign_base))
            label.append(0)
        except StopIteration:
            raise Exception('良性pe文件的数量不足')
            
    data = np.array(data)
    label = np.array(label)
    
    np.random.seed(seed)
    data = np.random.permutation(data)
    np.random.seed(seed)
    label = np.random.permutation(label)
    
    return data,label     

def centralize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalize_func = lambda x: (x-mean)/std
    data = np.apply_along_axis(normalize_func, axis=1, arr=data)
        
if __name__ == '__main__':
    path = get_benign_exe_abspath()
    for i,p in enumerate(path):
        if i >= 10:
            break
        #print(os.path.getsize(p[:-1])/1024)
        #print(p+'\n')
    #print(check_if_executable(r'C:/Windows/System32/1029/VsGraphicsResources.dll/'))
    
    #data,label = mix_samples()
    #np.save('data_0504.npy', data)
    #np.save('label_0504.npy', label)
    
    data = centralize_data(np.load(DATA_SAVE_NAME))
    label = np.load(LABEL_SAVE_NAME)
    
    pca = PCA(n_components=2)
    data_trans = pca.fit_transform(data)
    
    plt.plot([x[0] for x,l in zip(data_trans,label) if l==1], [x[1] for x,l in zip(data_trans,label) if l==1], 'bo', label='malware')
    plt.plot([x[0] for x,l in zip(data_trans,label) if l==0], [x[1] for x,l in zip(data_trans,label) if l==0], 'ro', label='benign')
    
    plt.legend()
    plt.show()
    
    '''
    a = np.array([[1,30,6],[2,30,2],[1,90,6]])
    mean = np.mean(a, axis=0)
    std = np.std(a, axis=0)
    m = np.max(a, axis=0)
    
    func = lambda x: (x-mean)/std
    b = np.apply_along_axis(func, axis=1, arr=a)
    '''
    
    '''
    train_data = data[:TEST_NUM*DATA_SPLIT]
    test_data = data[TEST_NUM*DATA_SPLIT]
    
    train_label = label[:TEST_NUM*DATA_SPLIT]
    test_label = label[TEST_NUM*DATA_SPLIT]
    '''
    
    
'''
hight_size = image.shape[0]/WIDTH
#plt.figure(figsize=(image.shape[0]*UNIT,256*UNIT))
plt.gray()
plt.imshow(image)

plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig(r'C:/Users/10904/Desktop/111111.png', dpi=100)
'''


