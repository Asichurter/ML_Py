# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:53:13 2019

@author: 10904
"""

import numpy

class Queue:
    def __init__(self, data=None):
        self.Data = []
        if type(data) == list:
            self.Data = data
        elif type(data) == numpy.ndarray:
            self.Data = data.tolist()
        
    def enqueue(self, item):
        self.Data.append(item)
        
    def dequeue(self):
        item = None
        try:
            item = self.Data[0].copy()
        except AttributeError:
            item = self.Data[0]
        self.Data[0] = None
        self.Data.remove(None)
        return item

    def length(self):
        return len(self.Data)
    
    def __str__(self):
        return str(self.Data)
        
if __name__ == '__main__':
    q = Queue([1,2,3,4])
    q.dequeue()
    q.dequeue()
    q.enqueue(7)
    q.enqueue(8)
    q.dequeue()
    print(q)