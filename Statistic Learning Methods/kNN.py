# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 02:04:00 2018

@author: 10904
"""

#k近邻方法
from sklearn.neighbors import KDTree
import numpy as np

#这是一句用于github测试的注释
#这是github客户端同步的测试注释

data = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2],[3,4],[1,0],[5,8],[6,2]])
tree = KDTree(data, leaf_size = 2)
dist, index = tree.query(np.array([[2,3]]), k=3)
print(dist, index, sep='\n')
#使用高斯分布函数作为核函数的核密度估计
#其工作原理是：利用给定的距离度量，计算出每个点到x的距离
#以该距离作为样本点，建立起若干个以这些样本点为中心的高斯分布，这些独立分布进行混合称为一个GMM，利用这个GMM计算x点的概率
#将带宽调高有助于更清楚地区分各个点
print(tree.kernel_density(np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2], [3,4],[1,0],[5,8],[6,2]])
        , h=1.0, kernel='gaussian'))