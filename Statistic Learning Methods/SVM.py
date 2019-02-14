# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 15:11:29 2018

@author: 10904
"""

import random
import numpy as np
import matplotlib.pyplot as plt
# -- coding: utf-8 --
 
R = 8

class SVM:
    def __init__(self):
        self.X = None
        self.G = None
        self.K = None
        self.Alpha = None
        self.Y = None
        self.B = None
        self.E = None
        self.C = None
        self.Toler = None
        self.W = None
        self.Kernel = None
        self.Kernel_Para = 1
    
    #检查数据格式的合法性    
    def data_check(self):
        N = self.X[0].shape[0]
        for data in self.X:
            if not data.shape[0] == N:
                raise Exception('\n数据维度不一致！非法！'+
                                '\n预期的维度: ' + str(N) +
                                '\n非法数据的维度: '+ str(data.shape[0]) + 
                                '\n非法数据: ' + str(data))
        Labels = []
        for label in self.Y:
            if label not in Labels:
                if len(Labels) == 2:
                    raise Exception('存在至少两个以上的标签！不属于二分类问题! ')
                else:
                    Labels.append(label)
        
    def train(self, x, y, kernel=None, kernel_para=1, C=1, toler=1e-3, max_iter=5000):
        self.C = C
        self.Toler = toler
        self.X = np.array(x)
        self.Y = np.array(y)
        self.Kernel = kernel
        self.Kernel_Para = kernel_para
        self.data_check()
        #计算K的内积矩阵
        self.kij(self.Kernel)
        self.smo(max_iter)
        #self.W = np.sum(self.Alpha.reshape(self.Alpha.shape[0], 1)*self.Y.reshape(self.Y.shape[0], 1)*self.X, axis=0)
        #print(self.W)
        #返回SMO算法近似得到的alpha矩阵与截距
        #可以利用公式计算W向量
        return self.Alpha,self.B
        
    def predict(self, x):
        KK = None
        if self.Kernel == None or self.Kernel == 'linear':
            KK = np.array([np.dot(x, xx) for xx in self.X])
        elif self.Kernel == 'poly':
            KK = np.array([(np.dot(x, xx)+1)**self.Kernel_Para for xx in self.X])
        elif self.Kernel == 'gauss':
            KK = np.array([np.exp(-1*np.linalg.norm(x-xx)**2/(2*self.Kernel_Para**2)) for xx in self.X])
        judgement = np.sum(self.Alpha.reshape(self.Alpha.shape[0], 1)*self.Y.reshape(self.Y.shape[0], 1)*KK.reshape(self.X.shape[0],1), axis=0) + self.B
        #print(judgement)
        if judgement >= 0:
            return 1
        else:
            return -1
        
    # 使用核函数
    def kij(self, para=None):
        if self.Kernel == None or self.Kernel == 'linear':
            self.K = np.dot(self.X, self.X.T)
        elif self.Kernel == 'poly':
            self.K = (np.dot(self.X, self.X.T)+1)**self.Kernel_Para
        elif self.Kernel == 'gauss':
            expand_mat = np.repeat(np.expand_dims(self.X,axis=1), self.X.shape[0], axis=1)
            minus_mat = expand_mat - expand_mat.transpose(1,0,2)
            self.K = np.exp(-1*np.apply_along_axis(lambda a: np.linalg.norm(a, 2)**2/(2*self.Kernel_Para**2), 2, minus_mat))
        else:
            raise Exception('核无法识别!\n非法指定的核:'+str(self.Kernel))
 
    #计算对应index坐标的Gx
    def gxi(self, index):
        return np.sum(self.Alpha * self.Y * (self.K[:, index].reshape(self.Y.shape[0], 1))) + self.B
 
    #计算整个Gx预测向量
    def gx(self, length):
        g = []
        for i in range(length):
            g.append(self.gxi(i))
        return g
    
    #计算误差向量，实质上就是预测值减去实际值
    def e(self):
        return self.G - self.Y
 
 
    # 判断是否满足KKT条件，不满足的话，求出违反的绝对误差
    def satisfy_kkt(self, index, variable_absolute_error):
        val = self.Y[index] * self.G[index]
        if self.Alpha[index] == 0:
            if val >= 1 - self.Toler:
                return True
            else:
                variable_absolute_error[index] = abs(1 - self.Toler - val)
                return False
     
        if 0 < self.Alpha[index] < self.C:
            if 1 - self.Toler <= val <= 1 + self.Toler:
                return True
            else:
                variable_absolute_error[index] = max(abs(1 - self.Toler - val), abs(val - 1 - self.Toler))
                return False
     
        if self.Alpha[index] == self.C:
            if val <= 1 + self.Toler:
                return True
            else:
                variable_absolute_error[index] = abs(val - 1 - self.Toler)
                return False
            
    def smo(self, max_iter):
        #print(Kij)
        N = self.X.shape[0]  # 有多少个样本
        
        self.Y = self.Y.reshape(N,1)
        
        # 初始值
        self.Alpha = np.zeros(len(self.X)).reshape(self.X.shape[0], 1)  # 每个alpha
        self.B = 0.0
        self.G = np.array(self.gx(N)).reshape(N,1)
        self.E = self.e()
        #print(G)
        #print(E)
        #input()
        #已经访问过的，可以确认无效的i，j值
        visit_j = {}
        visit_i = {}
        loop = 0
        while loop < max_iter:
            # 选择第一个变量
            # 先找到所有违反KKT条件的样本点
            print(loop)
            viable_indexes = []  # 所有可选择的样本
            viable_indexes_alpha_less_c = []  # 所有可选择样本中alpha > 0 且 < C的
            viable_indexes_and_absolute_error = {}  # 违反KKT的数量以及违反的严重程度，用绝对值表示
            for i in range(N):
                if not self.satisfy_kkt(i, viable_indexes_and_absolute_error) and i not in visit_i:
                    viable_indexes.append(i)
                    if 0 < self.Alpha[i] < C:
                        viable_indexes_alpha_less_c.append(i)
            if len(viable_indexes) == 0:  # 找到最优解了，退出
                break
            # 所有可选择样本中 alpha= 0 或 alpha = C的
            viable_indexes_extra = [index for index in viable_indexes if index not in viable_indexes_alpha_less_c]
            i = -1
     
            # 先选择间隔边界上的支持向量点
            if len(viable_indexes_alpha_less_c) > 0:
                most_obey = -1
                for index in viable_indexes_alpha_less_c:
                    if most_obey < viable_indexes_and_absolute_error[index] and index not in visit_i:
                        most_obey = viable_indexes_and_absolute_error[index]
                        i = index
            #再选择非支持向量点
            else:
                most_obey = -1
                for index in viable_indexes_extra:
                    if most_obey < viable_indexes_and_absolute_error[index] and index not in visit_i:
                        most_obey = viable_indexes_and_absolute_error[index]
                        i = index
            # 到这里以后，i肯定不为-1
            j = -1
     
            # 选择|E1 - Ej|最大的那个j
            max_absolute_error = -1
            for index in viable_indexes:
                if i == index:
                    continue
                try:
                    if max_absolute_error < abs(self.E[i] - self.E[index]) and index not in visit_j:
                        max_absolute_error = abs(self.E[i] - self.E[index])
                        j = index
                except Exception:
                    print(self.E)
                    print('i: ', i)
                    print('index: ', index)
                    raise Exception()
            
            # 找不到j，重新选择i
            #此时清理i对应的不可用的j，同时将当前的i加入到不可用列表中
            if j == -1:
                visit_j.clear()
                visit_i[i] = 1
                continue
     
            # 假设已经选择到了j
            alpha1_old = self.Alpha[i].copy()  # 这里一定要用copy..因为后面alpha[i]的值会改变，它变了，alpha1_old也随之会变,找了好多原因
            alpha2_old = self.Alpha[j].copy()
            alpha2_new_uncut = alpha2_old + self.Y[j] * (self.E[i] - self.E[j]) / (self.K[i][i] + self.K[j][j] - 2 * self.K[i][j])
     
            if self.Y[i] != self.Y[j]:
                L = max(0, alpha2_old - alpha1_old)
                H = min(self.C, self.C + alpha2_old - alpha1_old)
            else:
                L = max(0, alpha2_old + alpha1_old - self.C)
                H = min(self.C, alpha2_old + alpha1_old)
     
            # 剪辑切割
            if alpha2_new_uncut > H:
                alpha2_new = H
            elif L <= alpha2_new_uncut <= H:
                alpha2_new = alpha2_new_uncut
            else:
                alpha2_new = L
     
            # 变化不大，重新选择j
            if abs(alpha2_new - alpha2_old) < 1e-4:
                #此时先将i对应的j置为不可用
                visit_j[j] = 1
                continue
     
            alpha1_new = alpha1_old + self.Y[i] * self.Y[j] * (alpha2_old - alpha2_new)
            
            #如果导致新的alpha<0，违背了KKT条件，于是不可用
            if alpha1_new < 0:
                visit_j[j] = 1
                continue
     
            # 更新值
            self.Alpha[i] = alpha1_new
            self.Alpha[j] = alpha2_new
     
            b1_new = -self.E[i] - self.Y[i] * self.K[i][i] * (alpha1_new - alpha1_old) - self.Y[j] * self.K[i][j] * (alpha2_new - alpha2_old) + self.B
            b2_new = -self.E[j] - self.Y[i] * self.K[i][j] * (alpha1_new - alpha1_old) - self.Y[j] * self.K[j][j] * (alpha2_new - alpha2_old) + self.B
            
                    # 更新值
            #if loop == 0:
                #print(alpha, Y, Kij[0], b,'***********')
                #print('E: ', E)
                #print('b1_new = -E[i] - Y[i] * Kij[i][i] * (alpha1_new - alpha1_old) - Y[j] * Kij[i][j] * (alpha2_new - alpha2_old) + b： ',
                 #     b1_new,E[i],Y[i], Kij[i][i],alpha1_new,alpha1_old,Y[j],Kij[i][j],alpha2_new,alpha2_old,b,'***************')
            
            if 0 < alpha1_new < self.C:
                self.B = b1_new
            elif 0 < alpha2_new < self.C:
                self.B = b2_new
            else:
                self.B = (b1_new + b2_new) / 2
             
            #alpha的更新导致预测值G，误差值E都需要更新
            self.G = np.array(self.gx(N)).reshape(N, 1)
            self.Y = self.Y.reshape(N, 1)
            self.E = self.e()
            #print("iter  ", loop)
            #print("i:%d from %f to %f" % (i, float(alpha1_old), alpha1_new))
            #print("j:%d from %f to %f" % (j, float(alpha2_old), alpha2_new))
            #由于alpha的改变导致所有值都重新计算，因此清理不可用的i，j
            visit_j.clear()
            visit_i.clear()
            loop = loop + 1
            # print(alpha, b)
            
    def func_int(self, i):
        return (np.dot(self.X[i],self.W)+self.B)*self.Y[i] == 1
 
def draw(alpha, bet, data, label, C, mod, a, b):
    plt.xlabel(u"x1")
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    plt.ylabel(u"x2")
    for i in range(len(label)):
        if label[i] > 0:
            plt.plot(data[i][0], data[i][1], 'or')
        else:
            plt.plot(data[i][0], data[i][1], 'og')
    support_vecs_x = [x[0] for x,alp in zip(data,alpha) if alp > 0 and alp < C]
    support_vecs_y = [x[1] for x,alp in zip(data,alpha) if alp > 0 and alp < C]
    #num = [i for i in range(len(data))]
    #support_vecs_xx = [x[0] for i,x in zip(num, data) if mod.func_int(i)]
    #support_vecs_yy = [x[1] for i,x in zip(num, data) if mod.func_int(i)]
    #print(support_vecs_xx)
    plt.plot(support_vecs_x, support_vecs_y, 'xc')
    #plt.plot(support_vecs_xx, support_vecs_yy, 'xk')
    w1 = 0.0
    w2 = 0.0
    for i in range(len(label)):
        w1 += alpha[i] * label[i] * data[i][0]
        w2 += alpha[i] * label[i] * data[i][1]
    w = float(- w1 / w2)
 
    b = float(- bet / w2)
    r = float(1 / w2)
    lp_x1 = list([10, 90])
    lp_x2 = [-20*w+b, 20*w+b]
    lp_x2up = [-20*w+b+r, 20*w+b+r]
    lp_x2down = [-20*w+b-r, 20*w+b-r]
    lp_x2 = list(lp_x2)
    lp_x2up = list(lp_x2up)
    lp_x2down = list(lp_x2down)
    x,y = circle_drawpoints(0,0,R)
    plt.plot(x, y, 'k')
    '''
    plt.plot([-20,20], lp_x2, 'b')
    plt.plot([-20,20], lp_x2up, 'b--')
    plt.plot([-20,20], lp_x2down, 'b--')
    real_line = [-20*a+b, 20*a+b]
    plt.plot([-20,20], real_line, 'r')
    '''
 
 
def get_label(x,y):
    if x**2+y**2/4 <= R**2:
        return 1
    else:
        return -1
    
def circle_drawpoints(x,y,r, num=100):
    X = []
    Y = []
    P = np.pi*2
    for i in range(num):
        X.append(x+r*np.cos(i/num*P))
        Y.append(y+2*r*np.sin(i/num*P))
    return X,Y
 
if __name__ == '__main__':
    
    a = random.uniform(-5,5)
    b = random.uniform(-10,10)
    epoch = 150
    datas = []
    labels = []
    print('生成的直线: ','y=%f*x+%f'%(a,b))
    for i in range(epoch):
        x = random.uniform(-20,20)
        y = random.uniform(-20,20)
        #l = 1 if y-a*x-b >= 0 else -1
        l = get_label(x,y)
        datas.append([x,y])
        labels.append(l)
    #print(datas)
    #print(labels)
    '''
    datas = [[-11.519469224968603, -8.644879142855038],
             [-16.543290313593495, -14.807067053162001],
             [13.915949304193504, -10.344548390705924],
             [-2.979108734535526, -9.731374476756773],
             [17.686967074198982, 17.456578105210575],
             [9.645314131748343, 12.01819852710421],
             [-19.282816933781515, -4.18000437767709],
             [15.207911329941652, -0.7400049803698572],
             [14.549385718481851, -11.414462190309234],
             [2.1441784197034117, -15.107687248630782], 
             [-11.496645216864284, -1.4312836073770825], 
             [-2.50648148493395, -18.558027059721805],
             [0.38824770464222524, -16.470133656939208],
             [15.066252239848794, 17.170465904507275],
             [-16.30973587846496, 1.536579214305], 
             [13.706530320160837, -15.18876637617089],
             [3.5959398708198087, -7.485096575636469], 
             [0.03970066709614173, 16.569787532078564],
             [-19.546762600719244, -14.161739751901248], 
             [-0.3843871694965486, 9.35369727369271]]
    labels = [-1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1]
    '''
    datas = np.array(datas)
    labels = np.array(labels)
    svm = SVM()
    #print(datas)
    C = 1
    eps = 1e-2  # 误差值
    max_iter = 10000  # 最大迭代次数
    alpha, bb = svm.train(datas, labels, 'gauss', 10, C, eps, max_iter)
    #print(alpha)
    #print(bb)
    draw(alpha, bb, datas, labels, C, svm, a, b)
    test_X_pos = []
    test_Y_pos = []
    test_X_neg = []
    test_Y_neg = []
    test_iters = 100
    correct_iters = 0.
    #'''
    for i in range(test_iters):
        test_x = random.uniform(-20,20)
        test_y = random.uniform(-20,20)
        l = svm.predict(np.array([test_x, test_y]))
        #print('点:(',test_x,',',test_y,') ','\n实际标签: ', (1 if test_x**2+test_y**2 <= 100 else -1))
        #print('预测标签: ', l)
        #print('-------------------------------------------')
        if l == 1:
            test_X_pos.append(test_x)
            test_Y_pos.append(test_y)
            if test_x**2+test_y**2/4 <= R**2:
                correct_iters += 1
        else:
            test_X_neg.append(test_x)
            test_Y_neg.append(test_y)
            if test_x**2 + test_y**2/4 > R**2:
                correct_iters += 1
    plt.plot(test_X_pos, test_Y_pos, 'x', color='orange')
    plt.plot(test_X_neg, test_Y_neg, 'x', color='blue')
    plt.show()
    print('测试正确率: ', correct_iters/test_iters)
    #'''