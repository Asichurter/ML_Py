# -*- coding: utf-8 -*-

#Week1 NLP Task2
#Author:唐郅杰
#Task:1.中文分词和题目预处理
#     2.抽取部分（人工分类的）立体几何和函数与导数的题目，使用tf-idf表示每个题目
#     3.利用训练集训练Logistics Regression分类器，并使用测试集测试该分类器性能

from sklearn.linear_model import LogisticRegression
import jieba
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import numpy
import re
import csv

#用于文本输出指标
out = open('8.16/evaluation.txt', 'w', encoding='UTF-8')
#csv文件搜索最大长度
search_length = 950
#数据集长度
all_num = 140
#训练集长度
all_train_num = 80

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
print_con = False

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
        
#用于过滤题干中的字符，只剩下汉字
def word_filter(word_list):
    deprecated_words = u'''[a-zA-Z0-9’!"#$%&\'()（）*+,-./:;<=>?@，。；：?
    ★、…【】《》？“”‘’！[\\]^_`{|}~]+'''
    word2 = r'\\+'
    for i in range(len(word_list)):
        word_list[i] = re.sub(deprecated_words, '', word_list[i])
        word_list[i] = re.sub(word2, '', word_list[i])
        
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
    print('训练集长度:%d\n测试集长度：%d' % (len(all_train_con), len(all_test_con)))
    print('*****读取数据完成******')
    
    #预处理文本
    characters_substitude(all_train_con)
    characters_substitude(all_test_con)
    word_filter(all_train_con)
    word_filter(all_test_con)
    print(all_train_con)
    print('*****数据过滤完成******')
    
    if print_con:
        print(all_train_con)
        print(all_test_con)
    
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
    all_train_con = [' '.join(jieba.lcut(content)) for content in all_train_con]
    all_test_con = [' '.join(jieba.lcut(content)) for content in all_test_con]
    vec = CountVectorizer()
    
    #tf计算
    train_count = vec.fit_transform(all_train_con).toarray()
    #测试集利用训练集划分出来的字典进行匹配
    test_count = vec.transform(all_test_con).toarray()
    numpy.savetxt('8.16/train_count.txt', train_count)
    numpy.savetxt('8.16/test_count.txt', test_count)
    
    #tf-idf
    trans = TfidfTransformer()
    train_tfidf = trans.fit_transform(train_count).toarray()
    #测试集利用训练集划分出来的字典进行匹配
    test_tfidf = trans.transform(test_count).toarray()
    
    #文本输出矩阵
    numpy.savetxt('8.16/train_tdidf.txt', train_tfidf)
    numpy.savetxt('8.16/test_tdidf.txt', test_tfidf)
    
    #逻辑回归分类器的训练
    cla=LogisticRegression(penalty='l2')
    cla.fit(train_tfidf,all_train_lab)
    print('*****训练完成******')
    
    #分类器对测试集的预测
    predictions = cla.predict(test_tfidf)
    for i ,prediction in enumerate(predictions[:]):
        print ('第%d个：\n预测类型：%s\n信息：%s\n\n' %(i+1, prediction,all_test_lab[i]))
    
    #指标评价
    score = cla.score(test_tfidf, all_test_lab)
    cv_score = cross_val_score(cla, test_tfidf, all_test_lab, cv=5)
    print('测试样本准确率:', score)
    print('交叉验证准确率：', cv_score)
    print('交叉验证平均准确率：', numpy.mean(cv_score))
    
    #标签二进制化
    lb = LabelBinarizer()
    
    #储存二进制化标签
    all_test_lab_bin = numpy.array([number[0] for number in lb.fit_transform(all_test_lab)])
    
    #测试集样本的精确率
    precisions = cross_val_score(cla, test_tfidf, all_test_lab_bin, cv=5, scoring=
    'precision')
    print (u'\n精确率：', numpy.mean(precisions), precisions, sep='\n')
    
    #测试集样本的召回率
    recalls = cross_val_score(cla, test_tfidf, all_test_lab_bin, cv=5, scoring='recall')
    print (u'\n召回率：', numpy.mean(recalls), recalls, sep='\n')
    
    #f-measure综合指标
    f_score = cross_val_score(cla, test_tfidf, all_test_lab_bin, cv=5, scoring='f1')
    print('\nf1综合指标:', numpy.mean(f_score), f_score, sep='\n')
    
    #获得测试样本的分类得分
    predictions=cla.predict_proba(test_tfidf)
    #利用测试集样本分类得分来构建ROC曲线
    fpr, tpr, thresholds = roc_curve(all_test_lab_bin, predictions[:
    , 1])
    #利用ROC曲线计算AUC值
    roc_auc=auc(fpr,tpr)
    print(u'\nAUC指数：%.2f' % (roc_auc))
    
    #文本输出
    out.write(u'测试集准确率：\n'+str(score))
    out.write(u'\n\n交叉验证准确率：\n'+str(numpy.mean(cv_score)))
    out.write(u'\n\n交叉验证精确率：\n'+str(numpy.mean(precisions)))
    out.write(u'\n\n交叉验证召回率：\n'+str(numpy.mean(recalls)))
    out.write(u'\n\n交叉验证f-meausre：\n'+str(numpy.mean(f_score)))
    out.write(u'\n\nAUC指数：%.2f' % (roc_auc))
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
    
if __name__ == '__main__':
    main()



