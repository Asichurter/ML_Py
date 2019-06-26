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
import random as rd


DATA_SAVE_NAME = r'datas/data_0504.npy'
LABEL_SAVE_NAME = r'datas/label_0504.npy'
DATA_SPLIT = 0.8
TEST_NUM = 1000
CHANNEL = 5
TEST_ITERS = 10


if __name__ == '__main__':
    #collect_save_data(seed=3)
    
    #raw_data = np.load('raw_'+DATA_SAVE_NAME)

    data = np.load(DATA_SAVE_NAME)
    label = np.load(LABEL_SAVE_NAME)
    
    pca = PCA(n_components=2)
    data_trans = pca.fit_transform(data)
    
    '''
    '''
    train_data = data_trans[:int(len(data)*DATA_SPLIT)]
    test_data = data_trans[int(len(data)*DATA_SPLIT):]

    
    train_label = label[:int(len(data)*DATA_SPLIT)]
    test_label = label[int(len(data)*DATA_SPLIT):]
    
    knn_accs = []
    svm_accs = []
    
    for i in range(TEST_ITERS):
        sample_index = rd.sample([i for i in range(len(train_data))], CHANNEL)
    
        train_data_few = [train_data[sample_index[i]] for i in range(CHANNEL)]
        train_label_few = [train_label[sample_index[i]] for i in range(CHANNEL)]
        
        knn = KNN(n_neighbors=1)
        knn.fit(train_data_few, train_label_few)
        
        svm = SVC(gamma='auto')
        svm.fit(train_data_few, train_label_few)
        
        knn_predict = knn.predict(test_data)
        svm_predict = svm.predict(test_data)
    
        knn_acc = np.sum(knn_predict==test_label)/len(knn_predict)
        svm_acc = np.sum(svm_predict==test_label)/len(svm_predict)
        print(knn_acc)
        print(svm_acc)
        
        knn_accs.append(knn_acc)
        svm_accs.append(svm_acc)
    
    knn_acc_mean = np.mean(knn_accs)
    knn_acc_std = np.std(knn_accs)
    
    svm_acc_mean = np.mean(svm_accs)
    svm_acc_std = np.std(svm_accs)

    plt.figure(figsize=(15,15))
    plt.title('Using KNN with k=1, acc=%.2f' % knn_accs[len(knn_accs)-1])
    plt.plot([x[0] for x,l in zip(train_data_few,train_label_few) if l==1], [x[1] for x,l in zip(train_data_few,train_label_few) if l==1], 'ro', label='train_malware')
    plt.plot([x[0] for x,l in zip(train_data_few,train_label_few) if l==0], [x[1] for x,l in zip(train_data_few,train_label_few) if l==0], 'bo', label='train_benign')
    plt.plot([x[0] for x,l,pl in zip(test_data,test_label,knn_predict) if l==1 and pl==1], [x[1] for x,l,pl in zip(test_data,test_label,knn_predict) if l==1 and pl==1], 'rx', label='malware_right')
    plt.plot([x[0] for x,l,pl in zip(test_data,test_label,knn_predict) if l==0 and pl==0], [x[1] for x,l,pl in zip(test_data,test_label,knn_predict) if l==0 and pl==0], 'bx', label='benign_right')
    plt.plot([x[0] for x,l,pl in zip(test_data,test_label,knn_predict) if l==1 and pl==0], [x[1] for x,l,pl in zip(test_data,test_label,knn_predict) if l==1 and pl==0], 'kx', label='malware_wrong')
    plt.plot([x[0] for x,l,pl in zip(test_data,test_label,knn_predict) if l==0 and pl==1], [x[1] for x,l,pl in zip(test_data,test_label,knn_predict) if l==0 and pl==1], 'gx', label='benign_wrong')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(15,15))
    plt.title('Using SVM, acc=%.2f' % svm_accs[len(svm_accs)-1])
    plt.plot([x[0] for x,l in zip(train_data_few,train_label_few) if l==1], [x[1] for x,l in zip(train_data_few,train_label_few) if l==1], 'ro', label='train_malware')
    plt.plot([x[0] for x,l in zip(train_data_few,train_label_few) if l==0], [x[1] for x,l in zip(train_data_few,train_label_few) if l==0], 'bo', label='train_benign')
    plt.plot([x[0] for x,l,pl in zip(test_data,test_label,svm_predict) if l==1 and pl==1], [x[1] for x,l,pl in zip(test_data,test_label,svm_predict) if l==1 and pl==1], 'rx', label='malware_right')
    plt.plot([x[0] for x,l,pl in zip(test_data,test_label,svm_predict) if l==0 and pl==0], [x[1] for x,l,pl in zip(test_data,test_label,svm_predict) if l==0 and pl==0], 'bx', label='benign_right')
    plt.plot([x[0] for x,l,pl in zip(test_data,test_label,svm_predict) if l==1 and pl==0], [x[1] for x,l,pl in zip(test_data,test_label,svm_predict) if l==1 and pl==0], 'kx', label='malware_wrong')
    plt.plot([x[0] for x,l,pl in zip(test_data,test_label,svm_predict) if l==0 and pl==1], [x[1] for x,l,pl in zip(test_data,test_label,svm_predict) if l==0 and pl==1], 'gx', label='benign_wrong')
    plt.legend()
    plt.show()

    xx = [1, 2]
    width = 0.4
    plt.title('5-shot accuracy using knn and svm')
    plt.xlim(0,3)
    plt.ylim(0,1)
    plt.yticks(np.arange(0,1,0.1))
    plt.xticks(xx, ['KNN','SVM'])
    plt.bar(xx, [knn_acc_mean, svm_acc_mean], width, yerr=[knn_acc_std, svm_acc_std], color='orange')
    plt.show()



    
    
    
    
    