# -*- coding: utf-8 -*-

#Week1 NLP Task1
#Author:唐郅杰
#Task:1.中文分词和题目预处理
#     2.构建词袋模型，利用词袋向量表示题目（tf,tf-idf）
#     3.选择5个题目，计算与其他题目的cos相似度

import jieba
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import numpy
import re
import csv

#用于记录i相似度的文本
out = open(r'8.15/result815.txt', 'w', encoding='UTF-8')

#录入的数据长度
data_length = 500

#选择计算cos相似度的题目
my_preferrence = [4, 120, 180, 323, 478]

#储存相似度的列表
my_pre_similar = []

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
        
#利用两个向量，计算cos相似度
def cos_similarity(l1, l2, i, num):
    num = numpy.dot(l1, l2) #计算内积
    denom = numpy.linalg.norm(l1) * numpy.linalg.norm(l2) #计算欧氏长度之积
    
    #汉字被过滤完了时的特殊情况
    if numpy.linalg.norm(l1) == 0 or numpy.linalg.norm(l2) == 0:
        return 0
    
    return (num / denom)

def main():
    #从csv数据文件中读取数据
    csv_reader = csv.DictReader(open('tiku_question_sx.csv', 'r', encoding='UTF-8'))
    
    #只获取“题干”这一列的信息
    content_row = [row['content'] for row in csv_reader]
    all_content = []
    
    #将题干储存在列表中
    for i in range(data_length):
        all_content.append(content_row[i])
    
    #预处理文本
    characters_substitude(all_content)
    word_filter(all_content)
    
    #tf
    #两个特殊字词的人工处理
    jieba.suggest_freq('离心率', tune=True)
    jieba.suggest_freq('度则', tune=False)
    
    #jieba分词后添加为带空格的字符串
    all_content = [' '.join(jieba.lcut(content)) for content in all_content]
    vec = CountVectorizer()
    
    #tf计算
    count = vec.fit_transform(all_content).toarray()
    numpy.savetxt(r'8.15/counter815.txt', count)
    
    #tf-idf
    trans = TfidfTransformer()
    tfidf = trans.fit_transform(count).toarray()
    numpy.savetxt(r'8.15/tdidf815.txt', tfidf)
    
    #print(vec.vocabulary_)
    
    #计算选定题目与其他题目的cos相似度
    for my in my_preferrence:
        my_similar = []
        for i in range(data_length):
            if i != my:
                my_similar.append([i, cos_similarity(count[my], count[i], i, my)])
        my_pre_similar.append(my_similar)
        
    #先对所有题目进行相似度的排序，再分别打印
    for my in my_pre_similar:
        my.sort(key=lambda com: com[1], reverse=True)
    
    #将最相似的3个题目的题序号和cos相似度打印到文本中
    item = 0
    for my in my_pre_similar:
        out.write('第%d道题最相似（题号，cos相似度）:\n' % (my_preferrence[item]))
        for i in range(3):
            out.write(str(my[i]))
            out.write('\n')
        out.write('\n\n')
        item+=1
        
    out.close()
    
if __name__ == '__main__':
    main()

    




























