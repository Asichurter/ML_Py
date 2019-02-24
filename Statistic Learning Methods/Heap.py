# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:26:31 2019

@author: 10904
"""

import math
import random
     
class Heap:
    def __init__(self, data, max_min=True, compare=lambda x,y: x > y):
        self.Data = data
        self.Max_Min = max_min
        self.Compare = compare
        sequence = [i for i in range(math.ceil(len(self.Data)/2))][::-1]
        for i in sequence:
            self.sift(i)                   
        
    def swap(self, i, j):
        if i != j:
            temp = None
            try:
                temp = self.Data[i].copy()
            except AttributeError:
                temp = self.Data[i]
            self.Data[i] = self.Data[j]
            self.Data[j] = temp
        
    def sift(self, index):
        #无左节点，由完全二叉树定义，该节点为叶节点
        if self.left(index) == None:
            return None
        #只有左节点，那左节点必定为叶节点，因此停止递归
        elif self.right(index) == None:
            if not self.compare(index, self.left(index)):
                self.swap(index, self.left(index))
        else:            
            #不满足堆的性质
            if not self.compare(index, self.left(index)) or not self.compare(index, self.right(index)):
                l_r = True
                if self.compare(self.right(index), self.left(index)):
                    l_r = False
                self.swap(index, self.left(index) if l_r else self.right(index))
                self.sift(self.left(index) if l_r else self.right(index))
                
    def top(self, remove=True):
        if len(self.Data) == 0:
            raise Exception('堆中没有元素了!')
        if not remove:
            return self.Data[0]
        else:
            if len(self.Data) == 1:
                return self.Data.pop()
            temp = None
            try:
                temp = self.Data[0].copy()
            except AttributeError:
                temp = self.Data[0]
            self.Data[0] = self.Data.pop()
            self.sift(0)
            return temp
    
    #自适应比较方法                
    def compare(self, i, j):
        if self.Max_Min:
            return self.Compare(self.Data[i], self.Data[j])
        else:
            return self.Compare(self.Data[j], self.Data[i])
        
    def length(self):
        return len(self.Data)

                
    def parent(self, index):
        if index == 0:
            return None
        return math.floor((index-1)/2)

    def left(self, index):
        if 2*index+1 >= len(self.Data):
            return None
        return 2*index+1
    
    def right(self, index):
        if 2*index+2 >= len(self.Data):
            return None
        return 2*index+2
    
    def self_check(self, index=0):
        if self.left(index) == None:
            return True
        elif self.right(index) == None:
            return self.compare(index, self.left(index))
        else:
            return self.compare(index, self.left(index)) and self.compare(index,self.right(index)) and\
                    self.self_check(self.left(index)) and self.self_check(self.right(index))
        
    
    def __str__(self):
        return str(self.Data)
    
    def  __len__(self):
        return len(self.Data)
    
if __name__ == '__main__':
    data = random.sample([i for i in range(100)], 10)
    print(data)
    heap = Heap(data)
    print(heap.self_check())
    print(len(heap))
    while heap.length() > 0:
        print(heap.top(True))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    