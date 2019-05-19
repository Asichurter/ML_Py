# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:11:19 2019

@author: 10904
"""
#利用pefile提取特征后，利用pca降维后输入到分类模型中进行分类

from utils import collect_save_data
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#import sklearn 
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC


DATA_SAVE_NAME = r'datas/data_0504.npy'
LABEL_SAVE_NAME = r'datas/label_0504.npy'
DATA_SPLIT = 0.8
TEST_NUM = 1000


if __name__ == '__main__':
    #collect_save_data(seed=3)
    
    #raw_data = np.load('raw_'+DATA_SAVE_NAME)
    data = np.load(DATA_SAVE_NAME)
    label = np.load(LABEL_SAVE_NAME)
    
    pca = PCA(n_components=0.9)
    data_trans = pca.fit_transform(data)
    
    '''
    '''
    train_data = data_trans[:int(len(data)*DATA_SPLIT)]
    test_data = data_trans[int(len(data)*DATA_SPLIT):]
    
    train_label = label[:int(len(data)*DATA_SPLIT)]
    test_label = label[int(len(data)*DATA_SPLIT):]
    
    knn = KNN(n_neighbors=1)
    knn.fit(train_data, train_label)
    
    svm = SVC(gamma='auto')
    svm.fit(train_data, train_label)
    
    predict = svm.predict(test_data)
    
    
    #predict = knn.predict(test_data)
    
    
    acc = np.sum(predict==test_label)/len(predict)
    print(acc)
    
    plt.figure(figsize=(40,40))
    plt.title('Using SVM with 2 components, acc=%.2f' % acc)
    plt.plot([x[0] for x,l in zip(train_data,train_label) if l==1], [x[1] for x,l in zip(train_data,train_label) if l==1], 'ro', label='train_malware')
    plt.plot([x[0] for x,l in zip(train_data,train_label) if l==0], [x[1] for x,l in zip(train_data,train_label) if l==0], 'bo', label='train_benign')
    plt.plot([x[0] for x,l,pl in zip(test_data,test_label,predict) if l==1 and pl==1], [x[1] for x,l,pl in zip(test_data,test_label,predict) if l==1 and pl==1], 'rx', label='malware_right')
    plt.plot([x[0] for x,l,pl in zip(test_data,test_label,predict) if l==0 and pl==0], [x[1] for x,l,pl in zip(test_data,test_label,predict) if l==0 and pl==0], 'bx', label='benign_right')
    plt.plot([x[0] for x,l,pl in zip(test_data,test_label,predict) if l==1 and pl==0], [x[1] for x,l,pl in zip(test_data,test_label,predict) if l==1 and pl==0], 'kx', label='malware_wrong')
    plt.plot([x[0] for x,l,pl in zip(test_data,test_label,predict) if l==0 and pl==1], [x[1] for x,l,pl in zip(test_data,test_label,predict) if l==0 and pl==1], 'gx', label='benign_wrong')
    
    plt.legend()
    plt.savefig('D:/1.png')
    plt.show()

    
    
    
    
    