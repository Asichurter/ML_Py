#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import argparse
import random

#5 Way 5-Shot 
parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)
#原值为1000000
parser.add_argument("-e","--episode",type = int, default= 10000)
#原值为1000
parser.add_argument("-t","--test_episode", type = int, default = 100)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()

#超参数的设置
# Hyper Parameters
#特征向量的维度
FEATURE_DIM = args.feature_dim
#关系向量的维度
RELATION_DIM = args.relation_dim
#类别数量，即N路
CLASS_NUM = args.class_num
#每一个类的支持集样本数量
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
#每一个类的请求集样本数量
BATCH_NUM_PER_CLASS = args.batch_num_per_class
#训练轮回数量
EPISODE = args.episode
#测试轮回数量
TEST_EPISODE = args.test_episode
#学习速率
LEARNING_RATE = args.learning_rate
#使用GPU加速
GPU = args.gpu
#隐藏层的数量
HIDDEN_UNIT = args.hidden_unit

#基于卷积神经网络的图像嵌入网络
class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        #第一层是一个1输入，64x3x3过滤器，批正则化，relu激活函数，2x2的maxpool的卷积层
        #由于卷积核的宽度是3，因此28x28变为64x25x25,经过了pool后变为64x13x13
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        #第二层是一个64输入，64x3x3过滤器，批正则化，relu激活函数，2x2的maxpool的卷积层
        #卷积核的宽度为3,13变为10，再经过宽度为2的pool变为5
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        #第三层是一个64输入，64x3x3过滤器，周围补0，批正则化，relu激活函数,的卷积层
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        #第四层是一个64输入，64x3x3过滤器，周围补0，批正则化，relu激活函数的卷积层
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        
    #前馈函数，利用图像输入得到图像嵌入后的输出
    def forward(self,x):
        #每一层都是以上一层的输出为输入，得到新的输出
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        #输出的矩阵深度是64
        return out # 64

#关系神经网络，用于在得到图像嵌入向量后计算关系的神经网络
class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
#第一层是128输入（因为两个深度为64的矩阵相加），64个3x3过滤器，周围补0，批正则化，relu为激活函数，2x2maxpool的卷积层
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
#第二层是64输入，64个3x3过滤器，周围补0，批正则化，relu为激活函数，2x2maxpool的卷积层
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        #第三层是一个将矩阵展平的线性全连接层，输入64维度，输出隐藏层维度10维度
        self.fc1 = nn.Linear(input_size,hidden_size)
        #第四层是一个结束层，将10个隐藏层维度转化为1个维度的值，得到关系值
        self.fc2 = nn.Linear(hidden_size,1)

    #关系网络的前馈方法
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        #print(out.size())
        return out

#每个神经网络层的权重的初始化方法，用于传递给module中所有的子模块的函数参数
def weights_init(m):
    classname = m.__class__.__name__
    #如果是卷积层
    if classname.find('Conv') != -1:
        #计算卷积核的长x宽x数量，得到总共需要初始化的个数
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #将权重向量初始化为以0为均值，2/n为标准差 的正态分布
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #如果该层存在偏置项，则将偏置项置为0
        if m.bias is not None:
            m.bias.data.zero_()
    #否则该层为批正则化
    elif classname.find('BatchNorm') != -1:
        #将数据全部置为1
        m.weight.data.fill_(1)
        #偏置项置为0
        m.bias.data.zero_()
    #否则为线性层时
    elif classname.find('Linear') != -1:
        #n为线性层的维度
        n = m.weight.size(1)
        #权重全部初始化为简单正态分布
        m.weight.data.normal_(0, 0.01)
        #偏置项全部置为1
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    # Step 1: init data folders
    print("************init data folders************")
    # init character folders for dataset construction
    #找到训练集的文件夹
    metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders()
    #print(metatrain_character_folders)
    #print(metatrain_character_folders)
    # Step 2: init neural networks
    print("************init neural networks************")

    #实例化嵌入网络和关系网络
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)
    
    #提供初始化函数，初始化权重矩阵
    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    #指定数据运算位置
    #module.gpu()需改为cpu()
    #feature_encoder.cuda(GPU)
    #relation_network.cuda(GPU)
    feature_encoder.cpu()
    relation_network.cpu()

    #指定嵌入网络的优化器为Adam优化
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    #指定嵌入网络的学习速率调整器，参数为：嵌入网络的Adam优化器，每100000次调整一次，调整系数为0.5
    #调用scheduler.step()的时候，代表经过了一个epoch，将会自动对epoch计数加1，在设定的时候对lr进行调整
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    #指定关系网络的优化器为Adam优化
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    #指定关系网络的学习速率调整器，参数为：嵌入网络的Adam优化器，每100000次调整一次，调整系数为0.5
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    #装载已有的模型
    #这是在已有的模型的基础上继续训练，可以考虑重新训练
    
    if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"), map_location='cpu'))
        print("load feature encoder success")
    if os.path.exists(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"), map_location='cpu'))
        print("************load relation network success************")
        
    # Step 3: build graph
    print("************Training*************")
    
    #上一轮的精确率
    last_accuracy = 0.0

    for episode in range(EPISODE):
        
        print(episode, ' th Episode Starting!')
        #学习速率调整器的调用
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        #随机选取一个旋转角度
        degrees = random.choice([0,90,180,270])
        #建立任务，将会根据训练数量随机选取文件夹作为训练集和测试集，同时生成对应文件夹的标签
        task = tg.OmniglotTask(metatrain_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        #通过获取数据加载器方法获得一个dataloader
        sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False, rotation=degrees)
        batch_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)
        
        # sample datas
        #由于dataloader自己实现了__next__方法，因此每调用一次
        #dataloader将会自动从文件夹中读取训练的图像同时向量化，随之贴上一个标签
        #再将所有这样的（28x28向量-标签）的向量储存起来，返回成为样本，标签
        #样本数量：类的数量：5x每一个类包含的数量：5 = 25
        #因此样本向量实际尺寸：25x28x28
        samples,sample_labels = sample_dataloader.__iter__().next()
        batches,batch_labels = batch_dataloader.__iter__().next()
        #print(batch_labels)
    
        # calculate features
        #等同于forward
        #可能的修改：Variable类已经被弃用，因此考虑对tensor调用tensor = requires_grad_(true)方法代替Variable
        #这样的话，张量将会支持自动微分更新
        #将cuda()改为cpu()，该方法将会对samples在cpu中进行数据拷贝
        #sample_features = feature_encoder(Variable(samples).cuda(GPU)) # 25x64*5*5
        sample_features = feature_encoder(Variable(samples).cpu()) # 25x64*5*5
        
        #将样本特征矩阵转换为5x5x64x5x5
        #第一个5代表类的数量
        #第二个5代表每个类含有的样本数量
        #后面的64x5x5是每个样本的嵌入后尺寸
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,5,5)
        #将每个类内部的所有向量相加作为类别的向量，得到一个5x1x64x5x5的向量
        #在squeeze操作以后，将会压缩掉维度为1的向量，即变为5x64x5x5
        sample_features = torch.sum(sample_features, 1).squeeze(1)
        #batch_features = feature_encoder(Variable(batches).cuda(GPU)) # 20x64*5*5
        batch_features = feature_encoder(Variable(batches).cpu()) # 20x64*5*5

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        # 使用repeat方法将会使得第一维中多出15x5=75个的相同的5x64x5x5向量，即75x5x64x5x5
        sample_features_ext = sample_features.repeat(BATCH_NUM_PER_CLASS*CLASS_NUM, 1, 1, 1, 1)
        # 使用repeat方法将会使得第一维中多出5个的相同的75x64x5x5向量，即5x75x64x5x5
        batch_features_ext = batch_features.repeat(CLASS_NUM, 1, 1, 1, 1)
        #transpose方法将会返回矩阵的转置，转置的维度为第0维和第1维
        #转置后的维度：75x5x64x5x5，使得batch与sample的维度相同
        #维度解释：第1维：类别 第二维：每个类中的15个batch，共75个，与一个类中的一个元素的relation，共75
        #第3~5维：特征向量
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

        #利用cat将sample的特征矩阵群和batch的特征矩阵群，沿着第2维度（第三个维度）相连接
        #得到的是一个5x75x128x5x5的向量
        #然后展开成375x128x5x5的向量，这意味着共有375个支持-询问对
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1, FEATURE_DIM*2, 5, 5)
        #将输出的关系矩阵375x1转化为75x5的矩阵，即每个batch对应各个类的概率
        #例如，75中的1个为：[0.1, 0.2 0.5, 0.3, 0.3]，代表这个batch对应第一个类的概率为0.1...后面类推
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
        

        mse = nn.MSELoss().cpu()
        #mse = nn.MSELoss().cuda(GPU)
        #将batch的标签值列表：[0, 1, 2, 3, 4, 3...]映射成one-hot矩阵
        #其中每个位置的值都将作为1的纵坐标，即如果位置代码为3，则纵值向量中，3位置为1，其他位置为0
        #其中scatter_的作用是：将第三个参数src的每一个矩阵元素值，沿着第一个参数dim指定的维度进行移动
        #移动的目的地：由第二个参数index指定：index矩阵中与src中相同的位置的值
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1).type(torch.LongTensor), 1)).cpu()
        #将实际值与结果值传递给损失函数
        #print(type(one_hot_labels))
        loss = mse(relations,one_hot_labels)
        #print(type(loss))

        # training
        #清空梯度为0
        feature_encoder.zero_grad()
        relation_network.zero_grad()
    
        #由于loss是由relations计算得到，relations的输入由CNNNet得到
        #因此在反向传播过程中，将会递归的计算所有网络的梯度
        #在这里，RN与CNN的梯度都会由这个损失函数递归算出
        loss.backward()

        #参数正则化
        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(),0.5)

        #利用优化器对神经网络的参数进行一次优化
        feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode+1)%10 == 0:
            print("episode:",episode+1,"loss: ",loss.item())

        #原值的测试周期是10
        if (episode+1)%100 == 0:
            # test
            print("Testing...")
            total_rewards = 0

            for i in range(TEST_EPISODE):
                print('Test Stage', i, '...')
                degrees = random.choice([0,90,180,270])
                task = tg.OmniglotTask(metatest_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,SAMPLE_NUM_PER_CLASS)
                sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
                test_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)

                sample_images,sample_labels = sample_dataloader.__iter__().next()
                test_images,test_labels = test_dataloader.__iter__().next()

                # calculate features
                #sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                sample_features = feature_encoder(Variable(sample_images).cpu()) # 5x64
                sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,5,5)
                sample_features = torch.sum(sample_features,1).squeeze(1)
                #test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64
                test_features = feature_encoder(Variable(test_images).cpu()) # 20x64

                # calculate relations
                # each batch sample link to every samples to calculate relations
                # to form a 100x128 matrix for relation network
                sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
                test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
                test_features_ext = torch.transpose(test_features_ext,0,1)

                relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,5,5)
                relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                _,predict_labels = torch.max(relations.data,1)

                rewards = [1 if predict_labels[j].type(torch.LongTensor)==test_labels[j].type(torch.LongTensor) else 0 for j in range(CLASS_NUM*SAMPLE_NUM_PER_CLASS)]
                total_rewards += np.sum(rewards)

            test_accuracy = total_rewards/1.0/CLASS_NUM/SAMPLE_NUM_PER_CLASS/TEST_EPISODE

            print("test accuracy:",test_accuracy)

            if test_accuracy > last_accuracy:

                # save networks
                torch.save(feature_encoder.state_dict(),str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                torch.save(relation_network.state_dict(),str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

                print("save networks for episode:",episode)

                last_accuracy = test_accuracy

if __name__ == '__main__':
    main()
