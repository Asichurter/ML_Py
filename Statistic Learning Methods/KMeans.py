# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:39:37 2019

@author: 10904
"""

import numpy as np
import random
import matplotlib.pyplot as plt
#from GMM_Cluster import GMM_Cluster

class K_Means:
    def __init__(self):
        self.X = None
        self.Ave = None
        self.C = None
        self.Metric = None
        
    def train(self, data, k, metric='Euc', max_iter=1000):
        #input('K-Means Starting')
        if k > len(data):
            raise Exception('k均值聚类时，k的值超过了样本数量！'+'\n样本数: ' + str(len(data)) +
                            '\nk值: ' + str(k))
        #初始化模型
        #均值初始化为随机k个样本的值
        self.X = np.array(data)
        index = random.sample([i for i in range(self.X.shape[0])], k)
        self.Ave = np.array([self.X[i] for i in index])
        #indexes = [random.sample([i for i in range(self.X.shape[0])], 2*k) for i in range(k)]
        #self.Ave = np.array([np.sum(np.array([self.X[j] for j in indexes[i]]), axis=0)/2*k for i in range(k)
        #print(self.Ave.shape)
        loop = 0
        loop_ctrl = True
        while loop_ctrl and loop < max_iter:
            print(loop)
            loop_ctrl = True
            self.set_cluster(metric)
            for i,cluster in enumerate(self.C):
                total_x = np.zeros(self.X[0].shape)
                for c in cluster:
                    total_x += c
                if len(cluster) != 0:
                    new_ave = total_x/len(cluster)
                    #用于提前停止循环的条件，待优化
                    #loop_ctrl = loop_ctrl or np.sum(np.absolute(new_ave-self.Ave[i])) > 1e-20
                    self.Ave[i] = new_ave
            loop += 1
        self.set_cluster(metric)
    
    #将所有样本归到对应的簇中
    def set_cluster(self, metric):
        #print(self.Ave.shape)
        self.C = [[] for i in range(len(self.Ave))]
        for x in self.X: 
            dist = []
            for i,ave in enumerate(self.Ave):
                dist.append(self.cal_dist(x, ave, metric))
            self.C[list.index(dist, min(dist))].append(x)
                    
    def cal_dist(self, x1, x2, metric):
        if metric == 'Euc':
            return (np.sum((x1-x2)**2)**0.5)
        elif metric == 'Man':
            return np.sum(np.absolute(x1-x2))


if __name__ == '__main__':    
    def get_data(x, y, r, num):
        data = []
        for i in range(num):
            theta = random.uniform(-2*np.pi, 2*np.pi)
            R = random.uniform(0,r)
            data.append([x+R*np.cos(theta), y+R*np.sin(theta)])
        return data

    model = K_Means()
    data = []
    data += get_data(-10,10,5,50)
    data += (get_data(0,-5,5,50))
    data += (get_data(10,10,5,50))
    #for i in range(num):
    #    x = random.uniform(-20,20)   
    #    y = random.uniform(-20,20)
    #    data.append([x,y])
    
    model.train(data, k=3, metric='Man')
    clusters = []
    for c in model.C:
        clusters.append(c)
    ave = model.Ave
    #print(clusters)
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.plot([x[0] for x in clusters[0]], [x[1] for x in clusters[0]], 'o', color='red')
    plt.plot([x[0] for x in clusters[1]], [x[1] for x in clusters[1]], 'o', color='green')
    plt.plot([x[0] for x in clusters[2]], [x[1] for x in clusters[2]], 'o', color='blue')
    plt.plot([x[0] for x in ave], [x[1] for x in ave], 'x', color='black')
    print(ave)
    #plt.plot([x[0] for x in clusters[3]], [x[1] for x in clusters[3]], 'o', color='yellow')
    plt.show()

            
            
            
            
            
            
            
            
            