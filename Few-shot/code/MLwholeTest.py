# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:11:19 2019

@author: 10904
"""
# 利用pefile提取特征后，利用pca降维后输入到分类模型中进行分类

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
import random as rd

DATA_SAVE_NAME = r'../datas/data_0504.npy'
LABEL_SAVE_NAME = r'../datas/label_0504.npy'
DATA_SPLIT = 0.8
TEST_NUM = 1000
CHANNEL = 5
TEST_ITERS = 10

if __name__ == '__main__':

    data = np.load(DATA_SAVE_NAME)
    label = np.load(LABEL_SAVE_NAME)

    pca = PCA(n_components=2)
    data_trans = pca.fit_transform(data)
    
    knn = KNN(n_neighbors=1)
    svm = SVC(gamma='auto')

    train_data = data_trans[:int(len(data) * DATA_SPLIT)]
    test_data = data_trans[int(len(data) * DATA_SPLIT):]

    train_label = label[:int(len(data) * DATA_SPLIT)]
    test_label = label[int(len(data) * DATA_SPLIT):]
    
    knn.fit(train_data, train_label)
    svm.fit(train_data, train_label)
    
    knn_predict = knn.predict(test_data)
    svm_predict = svm.predict(test_data)

    knn_acc = np.sum(knn_predict==test_label)/len(test_label)
    svm_acc = np.sum(svm_predict==test_label)/len(test_label)

    plt.figure(figsize=(15, 15))
    plt.title('Using KNN with k=1, acc=%.2f' % knn_acc)
    plt.plot([x[0] for x, l in zip(train_data, train_label) if l == 1],
             [x[1] for x, l in zip(train_data, train_label) if l == 1], 'ro', label='train_malware')
    plt.plot([x[0] for x, l in zip(train_data, train_label) if l == 0],
             [x[1] for x, l in zip(train_data, train_label) if l == 0], 'bo', label='train_benign')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 1 and pl == 1],
             [x[1] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 1 and pl == 1], 'rx',
             label='malware_right')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 0 and pl == 0],
             [x[1] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 0 and pl == 0], 'bx',
             label='benign_right')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 1 and pl == 0],
             [x[1] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 1 and pl == 0], 'kx',
             label='malware_wrong')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 0 and pl == 1],
             [x[1] for x, l, pl in zip(test_data, test_label, knn_predict) if l == 0 and pl == 1], 'gx',
             label='benign_wrong')
    plt.legend()
    plt.show()

    plt.figure(figsize=(15, 15))
    plt.title('Using SVM, acc=%.2f' % svm_acc)
    plt.plot([x[0] for x, l in zip(train_data, train_label) if l == 1],
             [x[1] for x, l in zip(train_data, train_label) if l == 1], 'ro', label='train_malware')
    plt.plot([x[0] for x, l in zip(train_data, train_label) if l == 0],
             [x[1] for x, l in zip(train_data, train_label) if l == 0], 'bo', label='train_benign')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 1 and pl == 1],
             [x[1] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 1 and pl == 1], 'rx',
             label='malware_right')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 0 and pl == 0],
             [x[1] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 0 and pl == 0], 'bx',
             label='benign_right')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 1 and pl == 0],
             [x[1] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 1 and pl == 0], 'kx',
             label='malware_wrong')
    plt.plot([x[0] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 0 and pl == 1],
             [x[1] for x, l, pl in zip(test_data, test_label, svm_predict) if l == 0 and pl == 1], 'gx',
             label='benign_wrong')
    plt.legend()
    plt.show()







