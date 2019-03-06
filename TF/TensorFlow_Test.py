# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 00:49:14 2019

@author: 10904
"""

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import math

#获取到圆形的带标签数据
#par参数用于指定是否需要画图的点
def get_data(x, y, r, num, x_lim=[-20,20], y_lim=[-20,20], par=True):
    X = []
    Y = []
    if par:
        for i in range(num):
            xx = random.uniform(x_lim[0],x_lim[1])
            yy = random.uniform(y_lim[0],y_lim[1])
            X.append([xx,yy])
            if (xx-x)**2+(yy-y)**2 < r**2:
                Y.append([1])
            else:
                Y.append([0])
        return np.array(X),np.array(Y)
    else:
        p = 2*math.pi/num
        for i in range(num):
            X.append(x+r*math.cos(i*p))
            Y.append(x+r*math.sin(i*p))
        return X,Y

def forward(x, w_1, w_2, w_3, b_1, b_2, b_3):
    inter_layer_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)
    inter_layer_2 = tf.nn.relu(tf.matmul(inter_layer_1, w_2) + b_2)
    return tf.nn.sigmoid(tf.matmul(inter_layer_2, w_3) + b_3)

#获得一批的训练样本
def get_batch_samples(x, y, i, batch_size):
    s = (i*batch_size)%x.shape[0]
    e = min(s+batch_size, x.shape[0])
    return x[s:e],y[s:e]

def train(x, y, test_x, test_y, lr=0.1, steps=500, batch_size=8, decay=0.99, threshold=0.5, layer_width=[6,6]):
    width_1 = layer_width[0]
    width_2 = layer_width[1]
    
    X = tf.placeholder(tf.float32, [None, x.shape[1]], name='input')
    Y = tf.placeholder(tf.float32, [None, 1], name='label')
    
    W_1 = tf.Variable(tf.truncated_normal([x.shape[1], width_1], mean=0, stddev=0.1), name='W1')
    W_2 = tf.Variable(tf.truncated_normal([width_1, width_2], mean=0, stddev=0.1), name='W2')
    W_3 = tf.Variable(tf.truncated_normal([width_2, 1], mean=0, stddev=0.1), name='W3')
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.02)(W_1))
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.02)(W_2))
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.02)(W_3))
    
    B_1 = tf.Variable(tf.constant(0.1, shape=[width_1]), name='B1')
    B_2 = tf.Variable(tf.constant(0.1, shape=[width_2]), name='B2')
    B_3 = tf.Variable(tf.constant(0.1, shape=[1]), name='W1')
    
    global_step = tf.Variable(0, trainable=False)
    
    y_ = forward(X, W_1, W_2, W_3, B_1, B_2, B_3)
    
    #variables_average = tf.train.ExponentialMovingAverage(0.99, global_step)
    
    #moving_average_op = variables_average.apply(tf.trainable_variables())
    
    #regularizer = tf.layers
    
    loss = -tf.reduce_mean(Y * tf.log(tf.clip_by_value(y_, 1e-10, 1.)) + \
                           (1-Y) * (tf.log(tf.clip_by_value(1-y_, 1e-10, 1.))))
    
    loss += tf.add_n(tf.get_collection('loss'))
    
    l_r = tf.train.exponential_decay(lr, global_step, x.shape[0]/batch_size, decay, staircase=True)
    
    train_step = tf.train.AdamOptimizer(l_r).minimize(loss, global_step=global_step)
    
    ones = tf.ones_like(Y)
    zeros = tf.zeros_like(Y)
    #大于截断值的输出被判定为正类
    predicts = tf.where(y_ > threshold, ones, zeros)
    acc = tf.reduce_mean(tf.cast(tf.equal(predicts, Y), tf.float32))
    
    with tf.Session() as s:
        tf.global_variables_initializer().run()
        for i in range(steps):
            print(i)
            train_x,train_y = get_batch_samples(x, y, i, batch_size)
            #print(train_x.shape, train_y.shape)
            s.run(train_step, feed_dict={X:train_x, Y:train_y})
            if i % 50 == 0:
                print('第%d轮测试'%i)
                print('测试正确率: ', s.run(acc, feed_dict={X:test_x, Y:test_y}))
                #input()
        print('---------训练完成---------')
        predictions = s.run(predicts, feed_dict={X:test_x, Y:test_y})
        plt.plot([xx[0] for xx,yy in zip(x,y) if yy == 1], [xx[1] for xx,yy in zip(x,y) if yy == 1], 'ro', label='train+')          
        plt.plot([xx[0] for xx,yy in zip(x,y) if yy == 0], [xx[1] for xx,yy in zip(x,y) if yy == 0], 'go', label='train-')  
        plt.plot([xx[0] for xx,yy in zip(test_x,predictions) if yy == 1], [xx[1] for xx,yy in zip(test_x,predictions) if yy == 1], 'rx', label='test+') 
        plt.plot([xx[0] for xx,yy in zip(test_x,predictions) if yy == 0], [xx[1] for xx,yy in zip(test_x,predictions) if yy == 0], 'gx', label='test-') 
        edge_x,edge_y = get_data(0,0,14,100,par=False)
        plt.plot(edge_x, edge_y, 'k')
        plt.legend()
        plt.show()
        print('最终测试正确率: ', s.run(acc, feed_dict={X:test_x, Y:test_y}))
        print(len(tf.trainable_variables()))
                
if __name__ == '__main__':
    datas,labels = get_data(0,0,14,300)
    d,l = get_data(0,0,14,200)
    train(datas, labels, d, l)
    
    