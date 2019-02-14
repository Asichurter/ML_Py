# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:37:07 2019

@author: Asichurter
"""

from scipy.stats import multivariate_normal
import numpy as np
import random
import matplotlib.pyplot as plt
from KMeans import K_Means
 
#高斯混合模型聚类   
class GMM_Cluster:
    def __init__(self):
        self.X = None
        self.Ave = None
        self.Cov = None
        self.Alpha = None
        self.K = None
        self.PostPrior = None
        self.Clusters = None
        
    def train(self, data, k, cov_para=200, max_iter=150):
        #初始化
        #初始化均值向量，这里默认为二维的
        #x范围为-20到20之间平均点，y始终为0
        self.Ave = np.zeros([k,len(data[0])])
        for i,ave in enumerate(self.Ave):
            ave += -20+40/(k+1)*(i+1)
        #初始化协方差矩阵
        #初始值为对角矩阵，值由参数给出
        #由于协方差矩阵代表了多维高斯分布的值的紧密程度，所以初始值调整较大有利于迅速收敛
        self.Cov = np.array([np.eye(len(data[0]), len(data[0]))*cov_para for i in range(k)])
        #初始化混合系数，以平均值为初始值
        self.Alpha = np.array([1./k for i in range(k)])
        self.K = k
        self.X = np.array(data)
        #初始化后验概率矩阵
        self.PostPrior = [0 for i in range(k)]
        loop = 0
        while loop < max_iter:
            print(loop)
            #计算先验概率
            #第一个坐标指示样本，第二个坐标指示模型
            post_prior = [[] for i in range(len(self.X))]
            for x_i in range(len(self.X)):
                for m_i in range(k):
                    post_prior[x_i].append(self.cal_postprior(m_i, x_i))
            
            #更新参数
            #算法依据EM算法，先求隐变量的期望，实质上就是后验概率
            #再求Q函数的极大，这里给出的是解析解
            for i in range(k):
                #更新均值
                mean_head = np.zeros([self.X.shape[1], 1])
                tail = 0
                cov_head = np.zeros([self.X.shape[1],self.X.shape[1]])
                for j in range(len(self.X)):
                    mean_head += post_prior[j][i]*self.X[j].reshape(self.X.shape[1],1)
                    tail += post_prior[j][i]
                    
                self.Ave[i] = mean_head.reshape(self.X.shape[1])/tail
                
                #更新协方差矩阵
                for j in range(len(self.X)):
                    e = (self.X[j]-self.Ave[i]).reshape(self.X.shape[1], 1)
                    #每一项为γji*(xj-μi)(xj-μi)T
                    cov_head += post_prior[j][i]*(np.dot(e, e.T))
                #为了防止计算过程中，协方差矩阵非满秩，每轮计算都在对角线上增加一个极小的正则项来防止计算中止
                self.Cov[i] = cov_head/tail + np.eye(self.X.shape[1],self.X.shape[1])/1e6
                
                #更新混合系数
                self.Alpha[i] = tail/self.X.shape[0]   
            
            delta = np.array(post_prior)-np.array(self.PostPrior)
            #如果后验概率的改变值过小，则直接跳出循环
            if np.all(np.absolute(delta) < 1e-10):
                break
            #更新后验概率
            self.PostPrior = post_prior 
            #print(post_prior[0])
            #print(self.Alpha)
            loop += 1
                                
        self.Clusters = [[] for i in range(k)]
        for i,x in enumerate(self.X):
            #将每一个x添加到其后验概率最大的模型的簇中去
            self.Clusters[list.index(self.PostPrior[i], max(self.PostPrior[i]))].append(x)
        
    #i:模型的下标
    #j:样本的下标
    #即：Xj来自于Gi的后验概率   
    #公式的依据是：贝叶斯公式
    def cal_postprior(self, i, j):
        prob_total = 0
        for k in range(self.K):
            prob_total += self.Alpha[k]*multivariate_normal.pdf(self.X[j], mean=self.Ave[k], cov=self.Cov[k])
        return self.Alpha[i]*multivariate_normal.pdf(self.X[j], mean=self.Ave[i], cov=self.Cov[i])/prob_total
    
def get_data(x, y, r, num):
    data = []
    for i in range(num):
        theta = random.uniform(-2*np.pi, 2*np.pi)
        R = random.uniform(0,r)
        data.append([x+R*np.cos(theta), y+R*np.sin(theta)])
    return data
    
if __name__ == '__main__':
    model = GMM_Cluster()
    data = []
    data += get_data(-10,15,3,30)
    data += (get_data(-10,-10,3,30))
    data += (get_data(10,12,3,30))
    data += get_data(5,-5,3,30)
    model.train(data, 4)
    
    KM = K_Means()
    KM.train(data, 4)
    plt.title('GMM Clustering')
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.plot([x[0] for x in model.Clusters[0]], [x[1] for x in model.Clusters[0]], 'o', color='red')
    plt.plot([x[0] for x in model.Clusters[1]], [x[1] for x in model.Clusters[1]], 'o', color='green')
    plt.plot([x[0] for x in model.Clusters[2]], [x[1] for x in model.Clusters[2]], 'o', color='blue')
    plt.plot([x[0] for x in model.Clusters[3]], [x[1] for x in model.Clusters[3]], 'o', color='yellow')
    plt.show()
    
    clusters = []
    for c in KM.C:
        clusters.append(c)
    ave = model.Ave
    #print(clusters)
    plt.title('K-Means Clustering')
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.plot([x[0] for x in clusters[0]], [x[1] for x in clusters[0]], 'o', color='red')
    plt.plot([x[0] for x in clusters[1]], [x[1] for x in clusters[1]], 'o', color='green')
    plt.plot([x[0] for x in clusters[2]], [x[1] for x in clusters[2]], 'o', color='blue')
    plt.plot([x[0] for x in clusters[3]], [x[1] for x in clusters[3]], 'o', color='yellow')
    plt.plot([x[0] for x in ave], [x[1] for x in ave], 'x', color='black')
    #plt.plot([x[0] for x in clusters[3]], [x[1] for x in clusters[3]], 'o', color='yellow')
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    