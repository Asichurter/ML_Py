# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 20:58:46 2018

@author: Asichurter
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from Perceptron import Perceptron
from SVM import SVM

#基于梯度下降法，目前只适用于连续输入变量的逻辑斯蒂回归模型
class Logistic_Regression:
    def __init__(self, l_r=1e-3, threshold=0.5):
        #W
        self.Weights = []
        #学习速率
        self.Learning_Rate = l_r
        #截断点，一般都为0.5
        self.Threshold = threshold
        #记录输入的标签的字典和列表
        self.Labels = {}
        self.Labels_list = []
        
    def train(self, datas, epoches=20, strategy='BGD'):
        index = 0
        #初始化权重向量，所有位置都为0
        self.Weights = np.array([0. for i in range(datas[0][0].__len__()+1)])
        correct_all_time = 0
        #检查数据的合法性，顺便记录标签，将对应的标签对应到0和1上
        for data in datas:
            if not data[0].__len__() == datas[0][0].__len__():
                raise Exception('\n某个输入向量的维度与第一个输入向量的维度不一致，非法！'+
                                '\n输入向量的下标: ' + str(datas.index(data)) + 
                                '\n合法的维度: ' +str(datas[0][0].__len__()) +
                                '\n该向量的维度: ' +str(data[0].__len__()))            
            if not data[1] in self.Labels:
                if index == 2:
                    raise Exception('\n给定的数据中，不止有两个标签！' + 
                                    '\n已有标签: ' + self.Labels.keys + 
                                    '\n第三个标签: ' + data[1] + 
                                    '\n该条数据: ' + data)
                else:
                    self.Labels[data[1]] = index
                    index += 1
                    self.Labels_list.append(data[1])
        epoch = 0
        all_num = 0.
        correct_num = 0.
        #计算初始数据的正确率
        for data in datas:
            all_num += 1
            predict = self.predict(data[0])
            if self.predict(data[0]) == data[1]:
                correct_num += 1
        correct_now_time_keep = correct_num/all_num      
        while True:
            #初始化梯度向量
            gradient = np.array([0. for i in range(datas[0][0].__len__()+1)], dtype=float)
            #计算本轮的梯度，计算公式为: g=Σ(yi-sigmoid(-w·xi))·xi，这里的xi是n+1维的，因为有一个截距的存在，计算时要在原基础上加一个1
            for data in datas:
                gra_delta = (self.Labels[data[1]]-1./(1.+math.exp(-1.*self.cal_dot(data[0]))))*np.array(data[0]+[1.0])
                gradient += gra_delta
            #采用的是全梯度上升策略，可考虑使用SGD随机梯度上升代替
            #rand_list = random.sample([i for i in range(datas.__len__())], max([1, int(datas.__len__()/10)]))
            #for i,data in enumerate(datas): 
             #   if i in rand_list:
              #      self.Weights += self.Learning_Rate*(self.forward(data[0])-self.Labels[data[1]])*gradient
            for data in datas: 
                try:
                    # 使用梯度下降法更新权重
                    # 注意：梯度下降法的公式只含有梯度和补步长，即W'=W-▽·r，▽是梯度，r是步长，不含有误差项
                    weight_delta = self.Learning_Rate*gradient
                    self.Weights += weight_delta
                except OverflowError:
                    print('出界!!!!!!!!!!!!')
                    a = -1.*self.Weights[0]/self.Weights[1]
                    b = -1.*self.Weights[2]/self.Weights[1]
                    print('拟合直线: ', 'y=%f*x+%f'%(a, b))
                    return a,b
            all_num = 0.
            correct_num = 0.
            # 计算本轮的正确率
            for data in datas:
                all_num += 1
                try:
                    predict = self.predict(data[0])
                    if predict == data[1]:
                        correct_num += 1
                except OverflowError:
                    print('出界!!!!!!!!!!!!')
                    a = -1.*self.Weights[0]/self.Weights[1]
                    b = -1.*self.Weights[2]/self.Weights[1]
                    print('拟合直线: ', 'y=%f*x+%f'%(a, b))
                    return a,b
            correct_now_time = correct_num/all_num
            #当正确率符合一定条件的时候，跳出循环
            if (not correct_now_time > correct_all_time and epoch > epoches) or (correct_now_time >= 0.99):
                print('最终正确率: ' + str(correct_now_time))
                print('初始正确率: ' + str(correct_now_time_keep))
                a = -1.*self.Weights[0]/self.Weights[1]
                b = -1.*self.Weights[2]/self.Weights[1]
                #print('拟合直线: ', 'y=%f*x+%f'%(a, b))
                return a,b
        #返回值是生成的斜率和截距，这个是二维的测试时用的
        return -1.*self.Weights[0]/self.Weights[1],-1.*self.Weights[2]/self.Weights[1]
            
    #计算w·xi
    def cal_dot(self, data):
        if not data.__len__() == self.Weights.__len__()-1:
            raise Exception('\n在正向传播过程中，输入的向量的维度非法!'+
                            '\n合法维度: ' + str(self.Weights.__len__()-1) +
                            '\n输入维度: ' + str(data.__len__()) +
                            '\n输入的向量: ' + str(data))  
        else:
            return np.dot(self.Weights, data+[1])
    
    #f(x)=sigmoid(w·x)，这是模型的正向传播方法
    def forward(self, data):
        try:
            dot = self.cal_dot(data)
            return 1/(1+math.exp(-dot))
        except OverflowError:
            print('dot值: ',dot)
            print('x: ', data)
            print('w: ', self.Weights)
            raise OverflowError('出界！')  
    
    #预测标签
    def predict(self, data):
        if self.forward(data) >= self.Threshold:
            return self.Labels_list[1]
        else:
            return self.Labels_list[0]
        
if __name__ == '__main__':
    #生成线性可分的测试数据
    a = random.uniform(-5,5)
    b = random.uniform(-10,10)
    epoch = 100
    datas = []
    tests = []
    print('生成的直线: ','y=%f*x+%f'%(a,b))
    for i in range(epoch):
        x = random.uniform(-20,20)
        y = random.uniform(-20,20)
        l = 1 if y-a*x-b >= 0 else 0
        datas.append([[x,y],l])
    model = Logistic_Regression(5e-5)
    aa, bb = model.train(datas, 1000)  
    print('拟合的直线: ', 'y=%f*x+%f'%(aa,bb))
    for i in range(epoch):
        x = random.uniform(-20, 20)
        y = random.uniform(-20, 20)
        l = 1 if y-a*x-b >= 0 else 0
        tests.append([[x, y], l])
    cor_num = 0.
    for dat in tests:
        if model.predict(dat[0]) == dat[1]:
            cor_num += 1
    fig = plt.figure()
    '''
    plt.title('Logistic Regression for binary classifies')御医
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.plot([data[0][0] for data in datas if data[1] == 1], [data[0][1] for data in datas if data[1] == 1], 'o', color='red', label='train positive')
    plt.plot([data[0][0] for data in datas if data[1] == 0], [data[0][1] for data in datas if data[1] == 0], 'o', color='black', label='train negative')
    plt.plot([data[0][0] for data in tests if data[1] == 1], [data[0][1] for data in tests if data[1] == 1], 'x', color='red', label='test positive')
    plt.plot([data[0][0] for data in tests if data[1] == 0], [data[0][1] for data in tests if data[1] == 0], 'x', color='black', label='test negative')
    plt.plot([-20,20], [-20*a+b,20*a+b], '-', color='blue', label='real line')
    
    plt.legend()
    plt.show()
    '''
    
    print('测试正确率: ' + str(cor_num/epoch))
    
    datas_per = [[data[0],1 if data[1] == 1 else -1] for data in datas]
    perceptron = Perceptron(0.01)
    perceptron.train(datas)
    cor_num = 0
    tests_per = [[data[0],1 if data[1] == 1 else -1] for data in tests]
    for data in tests:
        if perceptron.predict(data[0]) == data[1]:
            cor_num += 1
    plt.title('Perceptron VS Log_Reg for binary classifies')
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.plot([data[0][0] for data in datas if data[1] == 1], [data[0][1] for data in datas if data[1] == 1], 'o', color='red', label='train positive')
    plt.plot([data[0][0] for data in datas if data[1] == 0], [data[0][1] for data in datas if data[1] == 0], 'o', color='black', label='train negative')
    plt.plot([data[0][0] for data in tests if data[1] == 1], [data[0][1] for data in tests if data[1] == 1], 'x', color='red', label='test positive')
    plt.plot([data[0][0] for data in tests if data[1] == 0], [data[0][1] for data in tests if data[1] == 0], 'x', color='black', label='test negative')
    w1,w2 = perceptron.cal_alpha_all()
    perceptron.cal_beta()
    B = perceptron.beta
    k = -1.*w1/w2
    BB = -1.*B/w2
    plt.plot([-20,20], [-20*k+BB, 20*k+BB], '-', color='orange', label='pertr fitting line')
    plt.plot([-20,20], [-20*a+b,20*a+b], '-', color='blue', label='real line')
    plt.plot([-20,20], [-20*aa+bb,20*aa+bb], '-', color='green', label='log fitting line')
    plt.legend()
    plt.show()
    print('拟合的直线: ', 'y=%f*x+%f'%(k,BB))
    print('测试正确率: ' + str(cor_num/epoch))
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        