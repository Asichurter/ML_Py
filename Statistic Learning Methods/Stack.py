# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:44:31 2019

@author: 10904
"""

import numpy

class Stack:
    def __init__(self, data=None):
        self.Data = []
        if type(data) == list:
            self.Data = data
        elif type(data) == numpy.ndarray:
            self.Data = data.tolist()
        
    def pop(self):
        return self.Data.pop()
    
    def push(self, item):
        self.Data.append(item)
    
    def length(self):
        return len(self.Data)
    
    def __str__(self):
        return str(self.Data)
        