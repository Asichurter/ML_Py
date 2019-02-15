# -*- coding: utf-8 -*-

#Week1 Task3,4 NLP
#Author:唐郅杰
#Task:1.中文分词和数据预处理
#     2.利用所有的题目对W2V模型进行训练
#     3.利用训练好的W2V模型，表示出所有知识点的词语，并计算cos相似度，筛选出最相似的10个词语
#     4.利用网络上已经训练好的W2V模型，对所有知识点进行表示并计算cos相似度，筛选出最相似的10个词语

import jieba
import re
import csv
from gensim.models.word2vec import Word2Vec

#是否利用数据集进行模型训练
if_train = True

#是否利用其他训练好的模型，否则将会加载已有的模型
if_otherModel = False

#知识点的人工写入
#由于许多知识点只是概括性的语句，如解析几何，导致语料库中根本没有此类词汇
#因此选择人工录入知识点
knowledge_points = ['函数，导数，极值，零点，根',\
                    '二面角，圆柱，三棱锥，圆台，正方体长方体，三视图',\
                    '三角形，余弦，正弦，正切，面积， 周长，边长，余弦定理，正弦定理',\
                    '数列，通项公式，递推公式，前项和',\
                    '圆，相切，直线，斜率，弦，外接，内接，切线',\
                    '直方图，统计，概率，频率，方差，期望',\
                    '椭圆，双曲线，抛物线，焦点，交点',\
                    '复数，共轭，虚部，实部',\
                    '集合，交集，并集，子集']   

#选择的知识点词汇
pre_knowledge_points = ['函数', '余弦', '正方体', '前项和', '方差']     
             
#stop_words = ['①', '②', '③', '⑶', '☆', '⑨', '与', '和', '或', '的', '还', '即']

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
jieba.suggest_freq('解三角形', tune=True)
jieba.suggest_freq('通项公式', tune=True)
jieba.suggest_freq('地推', tune=True)
jieba.suggest_freq('前项和', tune=True)
jieba.suggest_freq('直方图', tune=True)
jieba.suggest_freq('余弦定理', tune=True)
jieba.suggest_freq('正弦定理', tune=True)
jieba.suggest_freq('三棱锥', tune=True)
jieba.suggest_freq('圆台', tune=True)
jieba.suggest_freq('组合概率', tune=False)

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
        #替换逻辑符号
        word_list[i] = re.sub(u'∵', '因为', word_list[i])
        #替换圆符号
        word_list[i] = re.sub(u'⊙', '圆', word_list[i])
        
#用于过滤题干中的字符，只剩下汉字
def word_filter(word_list):
    deprecated_words = u'''[a-zA-Z0-9’!"#$%&\'()（）*+,-./:;<=>?@，。；：?
    ★、…【】『《》？“”‘’！．[\\]^_`{|}~]+'''
    word2 = r'\\+'
    for i in range(len(word_list)):
        word_list[i] = re.sub(deprecated_words, '', word_list[i])
        word_list[i] = re.sub(word2, '', word_list[i])
        
#利用所有的题目训练向量化模型
def train():
    #从csv数据文件中读取数据
    csv_reader = csv.DictReader(open('tiku_question_sx.csv', 'r', encoding='UTF-8'))
    #只获取“题干”这一列的信息
    content_row = [row['content'] for row in csv_reader]

    all_content = []
    #i = 1
    for content in content_row:
        #print('第%d条数据读取！' % i)
        #i+=1
        all_content.append(content)
    print('*****数据读取完成******')
    
    #预处理文本
    characters_substitude(all_content)
    word_filter(all_content)
    print('*****数据过滤完成******')

    #jieba分词后添加为带空格的字符串
    all_content = [jieba.lcut(content) for content in all_content]
    print(all_content)
    print('*****切词完成******')

    #模型初始化
    #window参数允许设置上下文词的数量
    model = Word2Vec(all_content, sg=0, min_count=1, window=5)
    #模型导出
    model.save('8.17/我的模型.model')
    model.wv.save_word2vec_format('8.17/训练模型数据.txt')
    print('*****训练完成******')
    return model

#将集合内的字符串展开
def flatten(points):
    for sentence in points:
        for word in sentence:
            yield word

#利用模型分析知识点
def knowledge_point_vec(knowledge_points, model):
    print('*****开始分析知识点******')
    out_all = open('8.17/知识点分词相似度(所有).txt', 'w', encoding='UTF-8')
    out_pre= open('8.17/知识点分词相似度(选择).txt', 'w', encoding='UTF-8')
    
    #数据预处理
    characters_substitude(knowledge_points)
    word_filter(knowledge_points)
    
    #分词
    points = [jieba.lcut(con) for con in knowledge_points]
    
    #数据展平
    points = list(flatten(points))
    
    #打印信息
    for word in points:
        #直接在模型的字典中对词语进行寻找
        if word in model.wv.index2word:
            if word in pre_knowledge_points:
                out_pre.write('与 %s 最相近的10个词语：\n' % word)
                for w, s in model.wv.similar_by_word(word):
                    out_pre.write('%s %.16f\n'%(w, s))
                out_pre.write('\n')
            else:
                out_all.write('与 %s 最相近的10个词语：\n' % word)
                for w, s in model.wv.similar_by_word(word):
                    out_all.write('%s %.16f\n'%(w, s))
                out_all.write('\n')
        #如果没有找到待向量化的词语，说明该词语不在模型的词汇表中，即没有训练过对应的词汇
        else: 
            out_all.write('%s不在词汇表内！\n\n' % word)
    out_pre.close()
    out_pre.close()
    print('*****知识点数据打印完成******')
  
#主程序
def main():
    if if_train:
        print('*****开始训练******')
        model=train()
        knowledge_point_vec(knowledge_points, model)
    else: 
        if if_otherModel:
            model = Word2Vec.load('8.17/model/Word60.model')
            #print(model['正弦'])
        else:
            model = Word2Vec.load('8.17/我的模型.model')
        print('*****读取成功！******')
        knowledge_point_vec(knowledge_points, model)

if __name__ == '__main__':
    main()



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    