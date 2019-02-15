# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:48:32 2019

@author: 10904
"""

from Heap import Heap 
import numpy as np
import random
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, data, k=3):
        self.Data = data
        self.K = k
        
    def predict(self, x, metric='Euc'):
        dists = []
        for d1 in self.Data:
            dists.append([d1[1],self.get_dist(d1[0],x,metric)])
        data_heap = Heap(dists, False, lambda x,y: x[1] > y[1])
        neighbors = {}
        #最远的一个点的距离作为半径，仅欧氏距离适用
        radius = 0
        for i in range(self.K):
            l,d = data_heap.top(True)
            if l in neighbors:
                neighbors[l] += 1
            else:
                neighbors[l] = 1
            if i == self.K-1:
                radius = d
        return max(neighbors, key=neighbors.get),radius         
        
    def get_dist(self, x, y, metric):
        if metric == 'Euc':
            return (np.sum((np.array(x)-np.array(y))**2)**0.5)
        elif metric == 'Man':
            return np.sum(np.absolute(np.array(x)-np.array(y)))
        else:
            raise Exception('无法识别的距离度量: ' + str(metric))

  
          
if __name__ == '__main__':
    def get_data(x, y, r, num):
        data = []
        for i in range(num):
            theta = random.uniform(-2*np.pi, 2*np.pi)
            R = random.uniform(0,r)
            data.append([x+R*np.cos(theta), y+R*np.sin(theta)])
        return data
    
    def circle_drawpoints(x,y,r, num=100):
        X = []
        Y = []
        P = np.pi*2
        for i in range(num):
            X.append(x+r*np.cos(i/num*P))
            Y.append(y+r*np.sin(i/num*P))
        return X,Y
    
    data = []
    data_1 = get_data(-9,-7,10,30)
    data_1 = [[x,1] for x in data_1]
    data += data_1
    data_2 = get_data(-1,2,10,30)
    data_2 = [[x,2] for x in data_2]
    data += data_2
    data_3 = get_data(7,9,10,30)
    data_3 = [[x,3] for x in data_3]
    data += data_3
    
    model = KNN(data,20)
    
    X = random.uniform(-20,20)
    Y = random.uniform(-20,20)
    
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.plot([x[0][0] for x in data_1], [x[0][1] for x in data_1], 'o', color='red')
    plt.plot([x[0][0] for x in data_2], [x[0][1] for x in data_2], 'o', color='green')
    plt.plot([x[0][0] for x in data_3], [x[0][1] for x in data_3], 'o', color='blue')
    l,r = model.predict([X,Y])
    XX,YY = circle_drawpoints(X,Y,r,200)
    if l == 1:
        plt.plot([X], [Y], 'o', color='red')
    elif l == 2:
        plt.plot([X], [Y], 'o', color='green')
    elif l == 3:
        plt.plot([X], [Y], 'o', color='blue')
    else:
        print('出错!')
    plt.plot([X], [Y], 'kx')
    plt.plot(XX, YY, 'k')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    