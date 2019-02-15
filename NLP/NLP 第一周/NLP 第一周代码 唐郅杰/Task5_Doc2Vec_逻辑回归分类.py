# -*- coding: utf-8 -*-

#Week1 Task5 NLP
#Author:唐郅杰
#Task:1.中文分词与数据预处理
#     2.利用人工分类出的部分知识点为“导数与函数”和“立体几何”的题目，对D2V模型进行训练
#     3.利用训练好的模型，得到所有的题目文档对应的向量表示
#     4.利用题目的向量表示，对Logistics Regression分类器进行训练
#     5.利用测试集，对训练好的分类器进行评估

from gensim.models.doc2vec import Doc2Vec
from sklearn.linear_model import LogisticRegression
import jieba
import gensim
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import numpy
import re
import csv

#W2V训练前使用的文本标记
TaggededDocument = gensim.models.doc2vec.TaggedDocument

#文本输出对象
out = open('8.18/evaluation.txt', 'w', encoding='UTF-8')

#csv文件搜索最大长度
search_length = 950

#由于数据集中知识点列内容无法读取，因此采用人工分类标签的方法，分类出了140个确定标签的样本，
#将其分为了80个训练样本和60个测试样本

#函数与导数的训练集题号
train_one_index = [1, 5, 39, 40, 53, 54, 60, 64, 67,\
                   82, 87, 90, 104, 105, 108, 109, 111,\
                   117, 121, 130, 158, 164,  209, 211,\
                   213, 219, 222, 224, 227, 233, 239,\
                   263, 264, 268, 285, 304, 313, 316,\
                   323, 329]

#立体几何的训练集题号
train_two_index = [11, 14,  103, 126,\
                  136, 150, 155, 177, 187, 234, 258,\
                  261, 273, 292, 300, 315, 322, 339,\
                  340, 345, 346, 363, 377, 385, 412,\
                  419, 426, 480,\
                  496, 517, 523, 533, 539, 549, 579,\
                  598, 616, 623, 686, 692]

#函数与导数的测试集题号
test_one_index = [7, 18, 19, 28, 174, 179, 191, 194,\
                  271, 341, 344, 350, 352, 371, 372,\
                   388, 394, 397, 404, 409, 416, 420,\
                   427, 428, 436, 438, 440, 444, 458,\
                   808]

#立体几何的测试集题号
test_two_index = [43, 59, 74, 75, 425, 443, 448, 451, 671, 709,\
                  716, 745, 747, 752, 766, 767, 780, 802, 803,\
                  810, 811, 817, 829, 847, 862, 863, 898, 908,\
                  936, 948]

#用于替换题目中LaTex中的数学关键字
def characters_substitude(word_list):
    for i in range(len(word_list)):
        #替换三角形
        word_list[i] = re.sub('triangle', '三角形', word_list[i])
        #替换根号
        word_list[i] = re.sub(u'sqrt', '根号', word_list[i])
        #替换加号
        word_list[i] = re.sub(u'\\+', '加', word_list[i])
        #替换减号
        word_list[i] = re.sub(u'["](.+)?\\-(.+)?["]', '', word_list[i])
        word_list[i] = re.sub(u'\\-', '减', word_list[i])
        #替换等号
        word_list[i] = re.sub(u'="', '', word_list[i])  
        word_list[i] = re.sub(u'=', '等于', word_list[i]) 
        #替换角
        word_list[i] = re.sub(u'angle', '角', word_list[i])
        word_list[i] = re.sub(u'{}\\^\\\\circ', '度', word_list[i])
        #替换平行
        word_list[i] = re.sub(u'[^:]//', '平行', word_list[i])
        #替换平方
        word_list[i] = re.sub(u'\\^\\{2\\}', '的平方', word_list[i])
        #替换三角函数符号
        word_list[i] = re.sub(u'sin', '正弦', word_list[i])
        word_list[i] = re.sub(u'cos', '余弦', word_list[i])
        word_list[i] = re.sub(u'tan', '正切', word_list[i])
    print('*****字符替换完成******')
        
#用于过滤题干中的字符，只剩下汉字
def word_filter(word_list):
    deprecated_words = u'''[a-zA-Z0-9’!"#$%&\'()（）*+,-./:;<=>?@，。；：?
    ★、…【】《》？“”‘’！[\\]^_`{|}~]+'''
    word2 = r'\\+'
    for i in range(len(word_list)):
        word_list[i] = re.sub(deprecated_words, '', word_list[i])
        word_list[i] = re.sub(word2, '', word_list[i])
    print('*****字符过滤完成******')
        
def cut_words(all_content):
    
    #特殊字词的人工处理
    jieba.suggest_freq('平行六面体', tune=True)
    jieba.suggest_freq('增函数', tune=True)
    jieba.suggest_freq('减函数', tune=True)
    jieba.suggest_freq('奇函数', tune=True)
    jieba.suggest_freq('单调递增', tune=True)
    jieba.suggest_freq('单调递减', tune=True)
    jieba.suggest_freq('偶函数', tune=True)
    jieba.suggest_freq('三棱锥', tune=True)
    jieba.suggest_freq('棱台', tune=True)
    jieba.suggest_freq('解析式', tune=True)
    jieba.suggest_freq('二面角', tune=True)
    jieba.suggest_freq('幂函数', tune=True)
    jieba.suggest_freq('表面积', tune=True)
    jieba.suggest_freq('导函数', tune=True)
    jieba.suggest_freq('的平方', tune=True)
    #jieba分词后添加为带空格的字符串
    all_content = [jieba.lcut(content) for content in all_content]
    return all_content
    print('*****切词完成******')
    
#再次处理预处理完成后的文本，使其贴上一个标签，变为TaggedDoc类型
def labelize(content, type_words):
        labelized = []
        for i,word_list in enumerate(content):
            l = len(word_list)
            #保证词语列表最后一个没有空格
            word_list[l-1] = word_list[l-1].strip()
            #将[内容，标签]作为原内容的新内容，即为原内容添加了标签
            labelized.append(TaggededDocument(word_list, tags=['%s_%d' % (type_words, i)]))
        print('*****标签化完成******')
        return labelized
    
#预处理文本，包括去除字母和标点、特殊符号，替换词语，内容标签化等
def preprocess_words(all_content, type_words):
    #替换词语
    characters_substitude(all_content)
    
    #字符过滤
    word_filter(all_content)
    
    #切词
    all_content = cut_words(all_content)
    
    #数据标签化
    all_content = labelize(all_content, type_words)
    
    print('*****数据预处理完成******')
    return all_content

#训练D2V模型，使之拥有所有词语/句子的向量，从而可以从模型中调取对应的句子或者词语向量
def train(train_con, test_con):
    print('*****开始训练文档向量化模型******')
    
    #单词最少出现1次，采用3词上下文，词语/句子向量维度为100
    model = Doc2Vec(min_count=1, window=3, vector_size=100)
    
    #将训练样本和测试样本连接起来，一起构建模型的词汇表
    all_con = train_con[:]
    all_con.extend(test_con[:])
    
    #构建模型词汇表
    model.build_vocab(all_con)
    
    #利用训练集和测试集分别训练模型，训练次数为70；后一项为固定项，指定数据长度
    model.train(train_con, epochs=70, total_examples=model.corpus_count)
    model.train(test_con, epochs=70, total_examples=model.corpus_count)
    
    #保存训练模型
    model.save('8.18/my_vectorizer_model.model')
    
    print('*****文档向量化模型训练完成******')
    return model

#通过已经训练好的D2V模型，得到所有的题目对应的句子/文档向量表示
def getVectors(model, con):
    #通过句子标签获得句子向量
    #vecs = [numpy.array(model.docvecs[z.tags[0]]).reshape((1, 100)) for z in con]
    #return numpy.concatenate(vecs)
    
    #通过计算句子中词汇向量的平均值获得句子向量，效果更好
    vecs = []
    for text in con:
        #text是已经标签化的文本，其中每个信息中含有
        #[0]：内容，也称words
        #[1]: 标签，也称tags
        #这里利用列表推断将每个词的向量分别释放到一个大的临时向量中储存
        tmp = [model[w] for w in text.words]
        tmp = numpy.array(tmp)
        #将大临时向量中的所有向量相加后求平均，将其作为句子向量添加到文档向量中
        #axis=0:指定向量内部相加
        vecs.append(tmp.sum(axis=0)/len(tmp))
    return numpy.array(vecs)

#构建逻辑回归分类器进行分类训练
def regression_sort(train_vectors, test_vectors, train_labels, test_labels):
    print('*****开始训练向量分类器******')
    #构建分类器，并使用训练集来对分类器进行训练
    cla=LogisticRegression(penalty='l2')
    cla.fit(train_vectors, train_labels)
    print('*****向量分类器训练完成******')
    
    #指标评价
    score = cla.score(test_vectors, test_labels)
    cv_score = cross_val_score(cla, test_vectors, test_labels, cv=5)
    print('测试样本准确率:', score)
    print('交叉验证准确率：', cv_score)
    print('交叉验证平均准确率：', numpy.mean(cv_score))

    #标签二进制化
    lb = LabelBinarizer()

    #储存二进制化标签
    #lb是一个将标签二进制化的对象，利用的是类似的fit_transform方法，返回的第一个元素[0]就是二进制文本
    test_labels_bin = numpy.array([number[0] for number in lb.fit_transform(test_labels)])

    #利用二进制标签，测试集样本的精确率
    precisions = cross_val_score(cla, test_vectors, test_labels_bin, cv=5, scoring='precision')
    print (u'\n精确率：', numpy.mean(precisions), precisions, sep='\n')

    #利用二进制标签，测试集样本的召回率
    recalls = cross_val_score(cla, test_vectors, test_labels_bin, cv=5, scoring='recall')
    print (u'\n召回率：', numpy.mean(recalls), recalls, sep='\n')

    #利用二进制标签，测试f-measure综合指标
    f_score = cross_val_score(cla, test_vectors, test_labels_bin, cv=5, scoring='f1')
    print('\nf1综合指标:', numpy.mean(f_score), f_score, sep='\n')

    #获得测试样本的分类得分
    predictions=cla.predict_proba(test_vectors)
    
    #利用测试集样本分类得分来构建ROC曲线
    #predictions的第一项[0]是逆向得分，[1]才是正向得分
    fpr, tpr, thresholds = roc_curve(test_labels_bin, predictions[:, 1])
    
    #利用ROC曲线计算AUC值
    #AUC是ROC曲线围成的面积
    roc_auc=auc(fpr,tpr)
    print(u'\nAUC指数：%.2f' % (roc_auc))
    
    #文本输出
    out.write(u'测试集准确率：\n'+str(score))
    out.write(u'\n\n交叉验证准确率：\n'+str(numpy.mean(cv_score)))
    out.write(u'\n\n交叉验证精确率：\n'+str(numpy.mean(precisions)))
    out.write(u'\n\n交叉验证召回率：\n'+str(numpy.mean(recalls)))
    out.write(u'\n\n交叉验证f-meausre：\n'+str(numpy.mean(f_score)))
    out.write(u'\n\nROC指数：%.2f' % (roc_auc))
    print('*****文本输出完成******')
    out.close
    
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

def main():
    #从csv数据文件中读取数据
    csv_reader = csv.DictReader(open('tiku_question_sx.csv', 'r', encoding='UTF-8'))
    #只获取“题干”这一列的信息
    content_row = [row['content'] for row in csv_reader]
    #训练集内容
    all_train_con = []
    #训练集标签
    all_train_lab = []
    #测试集内容
    all_test_con = []
    #测试集标签
    all_test_lab = []
    #将数据分割
    for i in range(search_length):
        if i in train_one_index:
            all_train_con.append(content_row[i])
            all_train_lab.append('导数与函数')
        elif i in train_two_index:
            all_train_con.append(content_row[i])
            all_train_lab.append('立体几何')
        elif i in test_one_index:
            all_test_con.append(content_row[i])
            all_test_lab.append('导数与函数')
        elif i in test_two_index:
            all_test_con.append(content_row[i])
            all_test_lab.append('立体几何')
    print('训练集长度:%d\n测试集长度：%d' % (len(all_train_lab), len(all_test_lab)))
    print('*****读取数据完成******')
    
    all_train_con = preprocess_words(all_train_con, 'TRAIN')
    all_test_con = preprocess_words(all_test_con, 'TEST')
    model = train(all_train_con, all_test_con)
    train_vecs = getVectors(model, all_train_con)
    test_vecs = getVectors(model, all_test_con)
    '''
    print(train_vecs)
    print('----------------------------------------------------------')
    print(test_vecs)
    '''
    regression_sort(train_vecs, test_vecs, all_train_lab, all_test_lab)

if __name__ == '__main__':
    main()