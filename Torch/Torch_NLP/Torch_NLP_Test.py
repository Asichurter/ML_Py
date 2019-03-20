# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:39:08 2019

@author: 10904
"""

import torch as t
import torch.nn.functional as F
import os
import numpy
import re
import random
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#题目最大词语数量
Max_Word = 100

#嵌入层维度
Embed_Size = 128

#训练轮数
epochs = 10

#标签的概率截点
Threhold = 0.5

class Net(t.nn.Module):
    def __init__(self, vocab_size, embed_size, embedding_weights=None):
        super(Net, self).__init__()
        self.Embed = t.nn.Embedding(vocab_size, embed_size, _weight=embedding_weights)
        self.Conv1 = t.nn.Conv2d(1, 3, 3, padding=1, bias=False)
        self.Conv2 = t.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.Conv3 = t.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.Dense1 = t.nn.Linear(2*3*5*16, 128)
        self.Dense2 = t.nn.Linear(128, 2)
        
        #初始化参数
        #self.initialize()
    
    def forward(self, x1, x2):       
        x1 = self.Embed(x1)
        x2 = self.Embed(x2)
        
        #将一个句子变为一通道的矩阵
        x1 = x1.view(1,1,x1.shape[0],-1)
        x2 = x1.view(1,1,x2.shape[0],-1)
        
        x1 = F.max_pool2d(F.relu(self.Conv1(x1)), (2,2))
        x2 = F.max_pool2d(F.relu(self.Conv1(x2)), (2,2))
        
        x1 = F.max_pool2d(F.relu(self.Conv2(x1)), (2,2))
        x2 = F.max_pool2d(F.relu(self.Conv2(x2)), (2,2))
        
        x1 = F.max_pool2d(F.relu(self.Conv3(x1)), (5,2))
        x2 = F.max_pool2d(F.relu(self.Conv3(x2)), (5,2))
        
        #展平为一维向量
        x = t.cat((x1,x2), dim=0).view(1, -1)
        x = F.relu(self.Dense1(x))
        x = self.Dense2(x)
        
        return x
    
    def initialize(self):
        for par in self.parameters():
            t.nn.init.xavier_normal_(par)
            
def make_label(prob):
    return t.Tensor([1-prob, prob])
     

#用于替换题目中LaTex中的数学关键字
def characters_substitude(sentence):
    
    #替换三角形
    sentence = re.sub('triangle', '三角形', sentence)
    #替换根号
    sentence = re.sub(u'sqrt', '根号', sentence)
    #替换加号
    sentence = re.sub(u'\\+', '加', sentence)
    #替换减号
    sentence = re.sub(u'["](.+)?\\-(.+)?["]', '', sentence)
    sentence = re.sub(u'\\-', '减', sentence)
    #替换等号
    sentence = re.sub(u'="', '', sentence)  
    sentence = re.sub(u'=', '等于', sentence) 
    #替换角
    sentence = re.sub(u'angle', '角', sentence)
    sentence = re.sub(u'{}\\^\\\\circ', '度', sentence)
    #替换平行
    sentence = re.sub(u'[^:]//', '平行', sentence)
    #替换平方
    sentence = re.sub(u'\\^\\{2\\}', '的平方', sentence)
    #替换三角函数符号
    sentence = re.sub(u'sin', '正弦', sentence)
    sentence = re.sub(u'cos', '余弦', sentence)
    sentence = re.sub(u'tan', '正切', sentence)
    #替换逻辑符号
    sentence = re.sub(u'∵', '因为', sentence)
    #替换圆符号
    sentence = re.sub(u'⊙', '圆', sentence)
    sentence = re.sub(u'①|②|④|③', '', sentence)
    
    sentence = sentence.split('@@@@@')

    deprecated_words = u'''[a-zA-Z0-9’!"#$%&\'\(\)（）*+,-./:;<=>\?@，。；：?
    ★、…【】『《》？“”‘’！[\\]\^_`\{|\}~]+'''
    deprecated_words = u'[a-zA-Z0-9$\{\}《》\^—_【】★→,：；、\?\|\(\)（）？“”‘’！，\.\．]+'
    sentence[0] = re.sub(deprecated_words, ' ', sentence[0])
    sentence[1] = re.sub(deprecated_words, ' ', sentence[1])
    #替换多个空格为一个空格
    sentence[0] = re.sub(u'\s+', ' ', sentence[0])
    sentence[1] = re.sub(u'\s+', ' ', sentence[1])
    #将行末和行头的空格去掉
    sentence[0] = re.sub(u'^ | $', '', sentence[0])
    sentence[1] = re.sub(u'^ | $', '', sentence[1])

    return [sentence[0].split(' '), sentence[1].split(' '), int(sentence[2])]

#利用两个向量，计算cos相似度
def cos_similarity(l1, l2):
    num = numpy.dot(l1, l2) #计算内积
    denom = numpy.linalg.norm(l1) * numpy.linalg.norm(l2) #计算欧氏长度之积
    
    #汉字被过滤完了时的特殊情况
    if numpy.linalg.norm(l1) == 0 or numpy.linalg.norm(l2) == 0:
        return 0
    
    return (num / denom)

def get_samples(source_left, source_right, num, total=1794, sample_num=100):
    indexes = [i for i in range(total)]
    sample_indexes = random.sample(indexes, sample_num)
    samples_left = numpy.array([source_left[i] for i in sample_indexes])
    samples_right = numpy.array([source_right[i] for i in sample_indexes])
    samples_labels = numpy.array([all_test_lab[i] for i in sample_indexes])
    return samples_left,samples_right,samples_labels

def centralize(data):
    return data-numpy.mean(data, axis=0).reshape(1,data.shape[1]).repeat(data.shape[0], axis=0)

def pad_seq(datas, length, index):
    for data in datas:
        #data = list(map(lambda x : x + 1 , data))
        if len(data) < length:
            for i in range(len(data), length):
                data.append(index)
    return datas

#用于训练的内容
all_train_con = []

#用于训练的标签
all_train_lab = []

#用于训练W2V的所有词语
all_train = []

#用于测试的内容
all_test_con = []

#用于测试的标签
all_test_lab = []

#已弃用
maxNum = 0
maxIndex = 0

#读取并处理训练的文本
fin = open('camp_dataset2/sim_question_train.txt', 'r', encoding='UTF-8')
for i, line in enumerate(fin):
    
    #debug用
    #if i == train_limit:
    #    break
    
    #文本预处理
    first, second, label = characters_substitude(line)
    
    #过滤掉所有长度大于100的题目，为了节省空间
    if len(first) > 100 or len(second) > 100:
        continue
    
    if len(first) > maxNum:
        maxNum = len(first)
    if len(second) > maxNum:
        maxNum = len(second)
        
    #将内容添加到神经网络的所有训练内容中
    #格式为[[左侧内容，右侧内容], [], []...]
    all_train_con.append([first, second])
    all_train_lab.append(label)
    
    #将内容添加到W2V的所有训练内容中
    all_train.append(first)
    all_train.append(second)

#读取并处理测试的文本
fin = open('camp_dataset2/sim_question_test.txt', 'r', encoding='UTF-8')
for i, line in enumerate(fin):
    '''
    if i == test_limit:
        break
    '''
    #文本预处理
    first, second, label = characters_substitude(line)
    
    #为了节省空间，过滤掉所有长度大于100的句子
    if len(first) > 100 or len(second) > 100:
        continue
    
    if len(first) > maxNum:
        maxNum = len(first)
    if len(second) > maxNum:
        maxNum = len(second)
        
    #将所有内容添加到所有神经网络的测试内容中
    all_test_con.append([first, second])
    all_test_lab.append(label)
    
    #将内容添加到所有W2V模型的训练内容中
    all_train.append(first)
    all_train.append(second)

print('*****数据读取完成*****')
#print(maxNum)
fin.close()

#构建W2V模型
model = Word2Vec(all_train, sg=0, size=128, min_count=1, window=5, cbow_mean=1)

#保存W2V模型
#model.save('我的W2V模型.model')
print('*****模型保存成功*****')

#创建用于临时储存单词向量的字典
embedding_index = {}
for List in all_train_con:
    for sentence in List:
        for word in sentence:
            try:
                #将模型中的词汇向量复制到字典中
                embedding_index[word] = model.wv[word]
            except KeyError:
                print('%s 不在词汇表内！' % word)
for List in all_test_con:
    for sentence in List:
        for word in sentence:
            try:
                #将模型中的词汇向量复制到字典中
                embedding_index[word] = model.wv[word]
            except KeyError:
                print('%s 不在词汇表内！' % word)    

#字典长度（单词数量）
vocab_size = len(embedding_index)
print('*****单词权重读取完成*****')

#添加一个空字符的对应下标
embedding_index[''] = vocab_size

#print(len(embedding_index))

#创建由单词到索引的字典，用于将题目的句子初始向量化
word_to_index = {word:i for i,word in enumerate(embedding_index)}

#用于训练的左侧的题干的向量矩阵
all_train_left_matrix = []

#用于训练的右侧的题干的向量矩阵
all_train_right_matrix = []

#用于测试的左侧的题干的向量矩阵
all_test_left_matrix = []

##用于测试的右侧的题干的向量矩阵
all_test_right_matrix = []

#将所有训练集的句子转换成词汇的顺序向量，内容为词汇的字典索引
#如：[2, 78, 1000, 876, 565, 89]
for List in all_train_con:
    seq_left = []
    seq_right = []
    for word in List[0]:
        seq_left.append(word_to_index[word])
    for word in List[1]:
        seq_right.append(word_to_index[word]) 
        
    #用于检测是否有句子长度超过设定值Max_Word   
    if len(seq_left) > 100:
        print(seq_left)
    if len(seq_right) > 100:
        print(seq_right)
    
    #题干向量    
    all_train_left_matrix.append(seq_left)
    all_train_right_matrix.append(seq_right)
    
for List in all_test_con:
    seq_left = []
    seq_right = []
    for word in List[0]:
        seq_left.append(word_to_index[word])
    for word in List[1]:
        seq_right.append(word_to_index[word])
        
    #用于检测是否有句子长度超过设定值Max_Word   
    if len(seq_left) > 100:
        print(seq_left)
    if len(seq_right) > 100:
        print(seq_right)
       
    #题干向量
    all_test_left_matrix.append(seq_left)
    all_test_right_matrix.append(seq_right)
    
all_train_left_matrix = pad_seq(all_train_left_matrix, 100, len(embedding_index))
all_train_right_matrix = pad_seq(all_train_right_matrix, 100, len(embedding_index))

#建立用于输入到神经网络嵌入层的，含有词汇初始化向量权重的矩阵
#若没有该词汇，则会以0代替
embedding_weights = numpy.zeros((vocab_size, Embed_Size), dtype='float32')
#遍历词汇，将所有词汇的向量导入
for word,i in word_to_index.items():
    try:
        embedding_weights[i, :] = embedding_index[word]
    except KeyError:
        print('%s 不在词汇表内！' % word)
print('*****权重矩阵构建完成*****')
embedding_weights = embedding_weights.tolist()
#将空字符对应的词嵌入加入到矩阵中
embedding_weights.append([0 for i in range(Embed_Size)])

net = Net(len(embedding_weights), Embed_Size, t.Tensor(embedding_weights))

net = net.cuda()

opt = t.optim.Adam(net.parameters())

loss_func = t.nn.CrossEntropyLoss()

for name,par in net.named_parameters():
    print(name, par.requires_grad)
    

for i in range(epochs):
    print(i, ' th epoch of training: ')
    running_loss = 0.
    epoch = 0
    
    for left,right,label in zip(all_train_left_matrix, all_train_right_matrix,all_train_lab):
        
        opt.zero_grad()
        
        out = net(t.LongTensor(left).cuda(), t.LongTensor(right).cuda())
        
        label = t.tensor([label], dtype=t.long, requires_grad=False).cuda()
        
        loss = loss_func(out, label).cuda()
        loss.backward()
        
        opt.step()
        
        running_loss += loss.data.item()
        
        epoch += 1
        if epoch % 1000 == 0:
            print('after ', epoch, ', average loss: ', running_loss/(epoch))
        
        
        
        
        
        
        
        
        