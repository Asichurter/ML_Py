# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:29:29 2019

@author: 10904
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER_1 = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 1e-4
TRAINING_STEP = 5000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_, avg, w_1, b_1, w_2, b_2):
    if avg == None:
        layer_1 = tf.nn.relu(tf.matmul(input_, w_1) + b_1)
        return tf.matmul(layer_1, w_2) + b_2
    else:
        layer_1 = tf.nn.relu(tf.matmul(input_, avg.average(w_1)) + avg.average(b_1))
        return tf.matmul(layer_1, avg.average(w_2)) + avg.average(b_2)
    
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    
    w_1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER_1], stddev=0.1))
    w_2 = tf.Variable(tf.truncated_normal([LAYER_1, OUTPUT_NODE], stddev=0.1))
    b_1 = tf.Variable(tf.constant(0.1, shape=[LAYER_1]))
    b_2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    
    y = inference(x, None, w_1, b_1, w_2, b_2)
    
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    average_y = inference(x, variable_averages, w_1, b_1, w_2, b_2)
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(w_1) + regularizer(w_2)
    loss = cross_entropy_mean + regularization
    
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, \
                                               mnist.train.num_examples / BATCH_SIZE,\
                                               LEARNING_RATE_DECAY)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    train_op = tf.group([train_step, variable_averages_op])
    
    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y_, 1))
    acc = tf.reduce_mean(correct_prediction)
    
    with tf.Session() as s:
        tf.global_variables_initializer().run()
        val = {x:mnist.validation.images, y:mnist.validation.labels}
        
        test = {x:mnist.test.images, y:mnist.test.labels}
        
        for i in range(TRAINING_STEP):
            if i % 1000 == 0:
                val_acc = s.run(acc, feed_dict=val)
                print('第i轮后验证精确度:', val_acc)
            xs, ys = mnist.train.nect_batch(BATCH_SIZE)
            s.run(train_op, feed_dict={x:xs, y:ys})
        
        test_acc = s.run(acc, feed_dict=test)
        print('测试精确度:', test_acc)

if __name__ == '__main__':
    mnist = input_data.read_data_sets('./', one_hot=True)
    train(mnist)
    