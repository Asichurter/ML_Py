# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:24:59 2019

@author: 10904
"""

#import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def forward(x, reuse=False, ave=None, width=None):
    if ave != None:
        with tf.variable_scope('layer1', reuse=reuse):
            w = tf.get_variable('weights', [x.shape[1], width[0]],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            
            b = tf.get_variable('bias', [width[0]],
                                initializer=tf.constant_initializer(0.1))
            inter_layer_1 = tf.nn.relu(tf.matmul(x, ave.average(w)) + ave.average(b))
            
        with tf.variable_scope('layer2', reuse=reuse):
            w = tf.get_variable('weights', [width[0], width[1]],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable('bias', [width[1]],
                                initializer=tf.constant_initializer(0.1))
            inter_layer_2 = tf.nn.relu(tf.matmul(inter_layer_1, ave.average(w)) + ave.average(b))
        
        with tf.variable_scope('layer3', reuse=reuse):
            w = tf.get_variable('weights', [width[1], 10],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable('bias', [10],
                                initializer=tf.constant_initializer(0.1))
            return tf.matmul(inter_layer_2, ave.average(w)) + ave.average(b)
        
    else:
        with tf.variable_scope('layer1', reuse=reuse):
            w = tf.get_variable('weights', [x.shape[1], width[0]],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.001)(w))
            b = tf.get_variable('bias', [width[0]],
                                initializer=tf.constant_initializer(0.1))
            inter_layer_1 = tf.nn.relu(tf.matmul(x, w) + b)
            
        with tf.variable_scope('layer2', reuse=reuse):
            w = tf.get_variable('weights', [width[0], width[1]],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.001)(w))
            b = tf.get_variable('bias', [width[1]],
                                initializer=tf.constant_initializer(0.1))
            inter_layer_2 = tf.nn.relu(tf.matmul(inter_layer_1, w) + b)
        
        with tf.variable_scope('layer3', reuse=reuse):
            w = tf.get_variable('weights', [width[1], 10],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.001)(w))
            b = tf.get_variable('bias', [10],
                                initializer=tf.constant_initializer(0.1))
            return tf.matmul(inter_layer_2, w) + b

#获得一批的训练样本
def get_batch_samples(x, y, i, batch_size):
    s = (i*batch_size)%x.shape[0]
    e = min(s+batch_size, x.shape[0])
    return x[s:e],y[s:e]

def train(x, y, test_x, test_y, val_x, val_y, lr=0.6, steps=5000, batch_size=128, decay=0.999, threshold=0.5, layer_width=[500,128]):
    
    X = tf.placeholder(tf.float32, [None, x.shape[1]], name='input')
    Y = tf.placeholder(tf.float32, [None, 10], name='label')
    
    '''
    W_1 = tf.Variable(tf.truncated_normal([x.shape[1], width_1], mean=0, stddev=0.1), name='W1')
    W_2 = tf.Variable(tf.truncated_normal([width_1, width_2], mean=0, stddev=0.1), name='W2')
    W_3 = tf.Variable(tf.truncated_normal([width_2, 10], mean=0, stddev=0.1), name='W3')
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.001)(W_1))
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.001)(W_2))
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(0.001)(W_3))
    
    B_1 = tf.Variable(tf.constant(0.1, shape=[width_1]), name='B1')
    B_2 = tf.Variable(tf.constant(0.1, shape=[width_2]), name='B2')
    B_3 = tf.Variable(tf.constant(0.1, shape=[10]), name='W1')
    '''
    
    global_step = tf.Variable(0, trainable=False)
    
    y_ = forward(X, width=layer_width)
    
    variables_average = tf.train.ExponentialMovingAverage(0.99, global_step)
    
    moving_average_op = variables_average.apply(tf.trainable_variables())
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=tf.arg_max(Y,1)))
    
    print('\n\n\n\n\n')
    print(loss)
    print(tf.add_n(tf.get_collection('loss')))
    loss += tf.add_n(tf.get_collection('loss'))
    
    l_r = tf.train.exponential_decay(lr, global_step, x.shape[0]/batch_size, decay, staircase=True)
    
    train_step = tf.train.GradientDescentOptimizer(l_r).minimize(loss, global_step=global_step)
    
    train_op = tf.group([train_step, moving_average_op])
    
    ave_y = forward(X, ave=variables_average, reuse=True, width=layer_width)
    
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(Y,1), tf.arg_max(ave_y, 1)), tf.float32))
    
    with tf.Session() as s:
        tf.global_variables_initializer().run()
        for i in range(steps):
            print(i)
            train_x,train_y = get_batch_samples(x, y, i, batch_size)
            #print(train_x.shape, train_y.shape)
            #print(train_x.shape, train_y.shape)
            s.run(train_op, feed_dict={X:train_x, Y:train_y})
            if i % 1000 == 0:
                print('第%d轮测试'%i)
                print(val_x.shape, val_y.shape)
                print('测试正确率: ', s.run(acc, feed_dict={X:val_x, Y:val_y}))
                print('loss: ', s.run(loss, feed_dict={X:val_x, Y:val_y}))
                #input('继续训练？')
        print('---------训练完成---------')
        print('最终测试正确率: ', s.run(acc, feed_dict={X:test_x, Y:test_y}))
                
if __name__ == '__main__':
    mnist = input_data.read_data_sets('./', one_hot=True)
    train(mnist.train.images, mnist.train.labels, \
          mnist.test.images, mnist.test.labels, \
          mnist.validation.images, mnist.validation.labels)
    
    


