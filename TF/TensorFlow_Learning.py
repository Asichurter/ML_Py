# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 00:38:17 2019

@author: 10904
"""

import tensorflow as tf

print('import successful!')


a = tf.Variable(tf.constant(0.2, shape=[1]), name='a')
b = tf.Variable(tf.constant(0.3, shape=[1]), name='b')
e = tf.train.ExponentialMovingAverage(0.99)
op = e.apply([a])

res = a+b
#print(e.variables_to_restore())
#saver = tf.train.Saver({'a':a, 'b':b})
init = tf.global_variables_initializer()

'''
with tf.Session() as s: 
    saver.restore(s, './model')
    print(s.run(res))
    '''
print(tf.test.is_gpu_available())




