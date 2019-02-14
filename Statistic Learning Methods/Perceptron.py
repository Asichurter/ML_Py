# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 18:25:10 2018

@author: Asichurter
"""

import numpy as np
import random

#感知器
class Perceptron:
    def __init__(self, lr):
        self.l_r = lr
        self.alpha = []
        self.beta = 0
        self.data = []
        self.gram = []
        self.x = []
        self.times = 0
        
    #计算对偶式学习中的βi 
    def cal_beta(self):
        total = 0
        for i,y in enumerate(self.data):
            total += self.alpha[i]*self.data[i]
        self.beta = total
        
    #计算对偶式学习中的αi    
    def cal_alpha(self,row):
        total = 0
        self.cal_beta()
        for i in range(self.alpha.__len__()):
            total += self.alpha[i]*self.data[i]*self.gram[i][row]
        return self.data[row]*(total + self.beta)
    
    #计算学习结果中的w
    def cal_alpha_all(self):
        total = np.zeros(np.shape(self.x[0][0]))
        self.cal_beta()
        for i,d in enumerate(self.alpha):
            total = np.add(total, np.array(self.x[i])*self.alpha[i]*self.data[i])
        return total
    
    def print_(self):
        print('alpha_list:', self.alpha, sep='\n')
        print('y:', self.data, sep='\n')
        print('gram matrix:', self.gram, sep='\n')
        print('alpha:',self.cal_alpha_all(),sep='\n')
        print('beta:', self.beta, sep='\n')
        print('times:', self.times, sep='\n')
        
    #训练感知器
    #输入的格式是一个列表，列表的每一个元素也是一个列表，第一个位置是x，第二个位置是y(+-1)
    #如:[[[1, 1, 1], 1], [[2, 1, 3], 1], [[0, -1, -2], -1]]
    #使用的是感知器学习的对偶形式，即记录每个x的累积次数
    def train(self, data):
        self.alpha = [0 for i in range(data.__len__())]
        
        for dat in data:
            self.data.append(dat[1])
            self.x.append(dat[0])
            
        for i,d1 in enumerate(data):
            self.gram.append([])
            for d2 in data:
                self.gram[i].append(np.dot(d1[0], d2[0]))
        
        
        #self.print_()
        all_fit = False
        #使用随机梯度下降更新
        while not all_fit:
            if self.times >= 100000:
                break
            all_fit = True
            for i,y in enumerate(data):
                #print('fit的第%d次：' % i, self.cal_alpha(i))
                if self.cal_alpha(i) <= 0:
                    self.alpha[i] += self.l_r
                    self.beta += self.l_r*self.data[i]
                    self.times += 1
                    all_fit = False
    
    #对数据进行预测             
    def predict(self, x):
        self.cal_beta()
        if np.dot(self.cal_alpha_all(), x)+self.beta > 0:
            return True
        else:
            return False
        
if __name__ == '__main__':
    trainer = Perceptron(0.1)
    trainer.train([[[3,3,2],1], [[4,3,1],1], [[1,1,1],-1]])
    trainer.print_()
    print(trainer.predict([1.3, 1.71, 1.05]))
    a = random.uniform(-5,5)
    b = random.uniform(-10,10)
    epoch = 5
    datas = []
    tests = []
    for i in range(epoch):
        x = random.uniform(-20,20)
        y = random.uniform(-20,20)
        l = 0 if y-a*x-b >= 0 else 1
        datas.append([[x,y],l])
    model = Perceptron(0.1)
    print(datas)
    model.train(datas)
    for i in range(epoch):
        x = random.uniform(-20,20)
        y = random.uniform(-20,20)
        l = 0 if y-a*x-b >= 0 else 1
        tests.append([[x,y],l])    
    cor_num = 0.
    print(tests)
    for dat in tests:
        print(dat[0],dat[1])
        if model.predict(dat[0]) == dat[1]:
            cor_num += 1
    print('测试正确率: ' + str(cor_num/epoch))
    a = [1,3,8,4,5,2,3,4,5]
    print(np.argmax(a))
       
            