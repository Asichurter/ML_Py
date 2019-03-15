# -*- coding: utf-8 -*-

#Week2 Task2 NlP
#Author:唐郅杰
#Task：1.使用W2V浅层表示词向量
#      2.搭建CDSSM神经网络训练语义匹配模型
#      3.将W2V表示的浅层词向量作为神经网络深入，得到题目的深层向量表示
#      4.利用得到的语义向量，计算语义相似度
#      5.利用标注的测试集，对训练得到的模型进行评估
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import os
import keras
import numpy
import re
import random
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras import regularizers
from keras.layers import Input, Dense, Dropout, Flatten, Conv1D, AveragePooling1D, Embedding, multiply,concatenate, LSTM
from keras.optimizers import SGD, RMSprop, Adam
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.sequence import pad_sequences

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#题目最大词语数量
Max_Word = 100

#嵌入层维度
Embed_Size = 128

#最大训练量，Debug用
train_limit = 5000

#最大测试量，debug用
test_limit = 5000

#训练轮数
epochs = 300

#标签的概率截点
Threhold = 0.5

#模型的保存/加载地址
save_address = r'my_model_final.h5'
load_address = r'my_model_final_20181224.h5'

#加载已有模型还是重新训练
if_train = True

#人工选择的进行语义相似度进行计算的题目
my_pre = [0, 56, 888, 300, 400]

#文本输入
out = open(r'sim.txt', 'w', encoding='UTF-8')

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
#print(all_train_left_matrix)

#利用keras自带的工具，将所有长度（含有词汇量）的矩阵转化为同一长度（Max_Word）的向量
#[78, 666, 272, 71, 12, 99, 0...]              [78, 666, 272, 71, 12, 99, 0...]
#[111, 89, 907, 771, 0...]           ——>       [111, 89, 907, 771, 0,  0, 0...] 
#[2, 999, 78, 0...]                            [2, 999, 78,   0,  0,  0,  0...]
all_train_left_matrix = pad_sequences(all_train_left_matrix, maxlen=Max_Word, padding='post')
all_train_right_matrix = pad_sequences(all_train_right_matrix, maxlen=Max_Word, padding='post')
all_test_left_matrix = pad_sequences(all_test_left_matrix, maxlen=Max_Word, padding='post')
all_test_right_matrix = pad_sequences(all_test_right_matrix, maxlen=Max_Word, padding='post')
print('*****输入数据处理完成*****')
print(all_train_left_matrix.shape)

#建立用于输入到神经网络嵌入层的，含有词汇初始化向量权重的矩阵
#若没有该词汇，则会以0代替
embedding_weights = numpy.zeros((vocab_size, Embed_Size), dtype='float32')
#遍历词汇，将所有词汇的向量导入
for word,i in word_to_index.items():
    try:
        embedding_weights[i, :] = embedding_index[word]
    except KeyError:
        print('%s 不在词汇表内！' % word)
numpy.savetxt(r'embedding_weights.txt', embedding_weights)
print('*****权重矩阵构建完成*****')

#如果重新训练的话
#搭建网络结构
if if_train:
    
    input1 = Input(shape=(Max_Word,))
    input2 = Input(shape=(Max_Word,))
    
    embeding = Embedding(vocab_size, 
                       Embed_Size, 
                       input_length=Max_Word, 
                        weights=[embedding_weights], 
                       trainable=False)
    
    embed1 = embeding(input1)
    embed2 = embeding(input2)
    
    '''
    lstm = LSTM(128,
                recurrent_dropout=0.1,
                dropout=0.4)
    
    bn = keras.layers.BatchNormalization()
    
    lstm1 = bn(lstm(embed1))
    lstm2 = bn(lstm(embed2))
    
    '''
    
    dropout_1 = Dropout(0.5)
    dropout_2 = Dropout(0.3)
    dropout_3 = Dropout(0.3)
    dropout_4 = Dropout(0.3)
    
    flat = Flatten()
    
    conv1 = Conv1D(128, 5, 
                   activation='tanh', 
                   padding='same',
                   kernel_regularizer=regularizers.l2(0.001),
                   kernel_initializer='he_normal')   
    
    conv2 = Conv1D(128, 5, 
                   activation='tanh', 
                   padding='same',
                   kernel_regularizer=regularizers.l2(0.001),
                   kernel_initializer='he_normal')
    conv3 = Conv1D(128, 5, 
                   activation='tanh', 
                   padding='same',
                   kernel_regularizer=regularizers.l2(0.001),
                   kernel_initializer='he_normal')
    conv4 = Conv1D(128, 5, 
                   activation='tanh', 
                   padding='same',
                   kernel_regularizer=regularizers.l2(0.001),
                   kernel_initializer='he_normal')


    pool1 = AveragePooling1D(3, 
                             padding='same', 
                             strides=2)
    pool2 = AveragePooling1D(3, 
                             padding='same', 
                             strides=2)
    pool3 = AveragePooling1D(3, 
                             padding='same', 
                             strides=2)
    pool4 = AveragePooling1D(3, 
                             padding='same', 
                             strides=2)
    

    out1 = flat(dropout_4(pool4(conv4(dropout_3(pool3(conv3(dropout_2(pool2(conv2(dropout_1(pool1(conv1(embed1)))))))))))))
    out2 = flat(dropout_4(pool4(conv4(dropout_3(pool3(conv3(dropout_2(pool2(conv2(dropout_1(pool1(conv1(embed2)))))))))))))
    
    #merge = concatenate([lstm1, lstm2], axis=-1)
    merge = multiply([out1, out2])

    '''
    #左侧输入层
    input1 = Input(shape=(Max_Word,))
    
    #左侧嵌入层
    embed1 = Embedding(vocab_size, 
                       Embed_Size, 
                       input_length=Max_Word, 
                       weights=[embedding_weights], 
                       trainable=True)(input1)
    
    #左侧卷积层
    conv1 = Conv1D(64, 3, 
                   activation='relu', 
                   padding='same',
                   kernel_regularizer=regularizers.l2(0.002))(embed1)
    
    #左侧丢失层
    drop1 = Dropout(0.3)(conv1)
    
    #左侧池化层
    pool1 = AveragePooling1D(3, 
                             padding='same', 
                             strides=2)(drop1)
    
    dense_1 = Dense(256,
                   activation='relu',
                   kernel_regularizer=regularizers.l2(0.002))(pool1)
    
    #左侧展平层
    out1 = Flatten()(dense_1) 

    #右侧输入层
    input2 = Input(shape=(Max_Word,))
    
    #右侧嵌入层
    embed2 = Embedding(vocab_size, 
                       Embed_Size, 
                       input_length=Max_Word, 
                       weights=[embedding_weights], 
                       trainable=True)(input2)
    
    #右侧卷积层
    conv2 = Conv1D(64, 3, 
                   activation='relu', 
                   padding='same',
                   kernel_regularizer=regularizers.l2(0.002))(embed2)
    
    #右侧丢失层
    drop2 = Dropout(0.3)(conv2)
    
    #右侧池化层
    pool2 = AveragePooling1D(3, 
                             padding='same', 
                             strides=2)(drop2)
    
    dense_2 = Dense(256,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.002))(pool2)
    
    #右侧展平层
    #文本的向量将会从这里进行输出
    out2 = Flatten()(dense_2)
    
    #将两个题目的向量做元素相乘运算，从而将其合并为一层
    multiplied = multiply([out1, out2])
    '''
    


    #全连接层1
    dense1 = Dense(256, 
                   activation='relu',
                   kernel_regularizer=regularizers.l2(0.001),
                   kernel_initializer='he_normal')(merge) 
    
    #合并丢失层1
    drop = Dropout(0.5)(dense1)
    
    #全连接层2
    dense2 = Dense(128, 
                   activation='relu',
                   kernel_regularizer=regularizers.l2(0.001),
                   kernel_initializer='he_normal')(drop) 
    
    #合并丢失层2
    drop_2 = Dropout(0.5)(dense2)
    
    dense3 = Dense(128,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.001),
                    kernel_initializer='he_normal')(drop_2)
    
    drop_3 = Dropout(0.5)(dense3)
    
    #全连接层3，输出层，以sigmoid为激活函数，解决二分类问题
    dense4 = Dense(1, 
                   activation='sigmoid')(drop_3)
       
    #建立函数式模型
    my_model = Model(inputs=[input1, input2], outputs=dense4)
    
    #已弃用的随机梯度优化器
    sgd = SGD(lr=3.0e-1, decay=1.0e-2, momentum=1e-2, nesterov=True)
    rmsprop = RMSprop(lr=0.01)
    adam = Adam(lr=5e-3)
    
    #编译。采用rmsprop作为优化器，二分类熵作为损失函数，准确率作为评价值
    my_model.compile(optimizer='adam', 
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
    
    #网络结构展示
    my_model.summary()
    print('*****开始训练*****')
    
    #训练，有两个输入一个输出，输入值是词汇组成的文本矩阵，输出值是相似度
    #轮回数量由epochs指定，批处理大小为64（最优），验证比例为0.2
    my_model.fit([all_train_left_matrix, all_train_right_matrix], all_train_lab, 
                 verbose = 1,
                 epochs=epochs,
                 batch_size=128, 
                 validation_split=0.2, 
                 shuffle=True)
    print('*****训练完成*****')
    
    #保存训练好的神经网络模型
    my_model.save(save_address)
    print('*****保存完成*****')

    #模型的评价值，由测试集得到
    score = my_model.evaluate([all_test_left_matrix, all_test_right_matrix], 
                              all_test_lab, 
                              verbose=1) 
    print('Test score:', score[0])#测试集中的loss
    print('Test accuracy:', score[1]) #测试集中的准确率

#如果读取已经训练好的模型
else:
    
    #读取模型
    my_model = load_model(load_address)
    print('*****读取模型成功*****')
    
    my_model.summary()
    
    #模型评估
    score = my_model.evaluate([all_test_left_matrix, all_test_right_matrix], 
                              all_test_lab, 
                              verbose=1) 
    print('Test score:', score[0])#测试集中的loss
    print('Test accuracy:', score[1]) #测试集中的准确率

'''
#构建用于左侧输入的文本向量的神经网络向量化的模型，将会输出深度学习以后的语义向量
#该向量由展平层的输出提供
vec_model_1 = Model(inputs=my_model.input[0], 
                    outputs=my_model.get_layer('flatten_3').output)

#右侧输入向量对应的模型，由右侧的展平层输出提供
vec_model_2 = Model(inputs=my_model.input[1],
                    outputs=my_model.get_layer('flatten_4').output)

#又左侧向量的向量化模型，预测选出的题目的向量
pre_vec = vec_model_1.predict([[all_train_left_matrix[i] for i in my_pre]], 
                              verbose=1)

#设置numpy的最大控制台显示输出
#numpy.set_printoptions(threshold=3200)
#print(len(pre_vec[0]))

#右侧的所有向量的向量化预测值。由于要找到与选出的题目最相近的题目，因此右侧所有题目都要向量化
all_vec = vec_model_2.predict([all_train_right_matrix],
                              verbose=1)

#[每一个选择的题目[每一个测量的题目[相似度，编号], [], []...],[],[]...[]]
pre_vec_sim = []
for vec_1 in pre_vec:
    seq = []
    for i,vec_2 in enumerate(all_vec):
        #[相似度，编号] 格式的列表
        seq.append([cos_similarity(vec_1, vec_2), i])
    pre_vec_sim.append(seq)

#将选定好的题目的相似度列表按照相似度降序进行排序
for i,List in enumerate(pre_vec_sim):
    List.sort(key=lambda com: com[0], reverse=True)
    
   # if i == 3:
    #    print(List)
    
    #文本打印
    out.write(u'与 %d 号题目最相似的题目为（题号，相似度）:\n' % my_pre[i])
    for i in range(10):
        out.write(u'%d %.8f\n' % (List[i][1], List[i][0]))
    out.write(u'\n')

#测试集由已经训练好的神经网络模型进行预测得到的结果，为概率列表'''
label_pro_predict = my_model.predict([all_test_left_matrix, all_test_right_matrix], verbose=1)
numpy.savetxt(r'pro_list.txt', label_pro_predict)


#将概率列表转化为固定的预测结果，以Trehold为截点
label_predict = []
for pro in label_pro_predict:
    #大于截点取标签1
    if pro > Threhold:
        label_predict.append(1)
    #小于截点取标签0
    else:
        label_predict.append(0)

#由固定的测试集标签预测结果，结合skleran工具得到precision，accuracy，recall，f1，auc等指标
#macro代表两个标签取同等比例的分数
print('精确度precision:')
print(precision_score(all_test_lab, label_predict, average='macro'))
print('准确度accuracy:')
print(accuracy_score(all_test_lab, label_predict, normalize=True))
print('召回率recall:')
print(recall_score(all_test_lab, label_predict, average='macro'))
print('f1得分f1-score:')
print(f1_score(all_test_lab, label_predict, average='macro'))

#利用测试集样本分类得分来构建ROC曲线
fpr, tpr, thresholds = roc_curve(all_test_lab, label_pro_predict)
#利用ROC曲线计算AUC值
roc_auc=auc(fpr,tpr)
print(u'\nAUC指数：%.2f' % (roc_auc))

#画ROC曲线图
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('TRP')
plt.xlabel('FPR')
plt.show()

#out.close()
    




























