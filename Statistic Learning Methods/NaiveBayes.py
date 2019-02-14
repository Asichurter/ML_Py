# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 17:29:27 2018

@author: Asichurter
"""
from DecisionTree import DecisionTree as DTree
import math

#朴素贝叶斯分类器
class NaiveBayes:
    #使用所有属性的所有取值和所有标签的取值，还有Laplace平滑系数来初始化分类器
    def __init__(self, values, labels, Lambda):
        #Laplace平滑系数
        self.Lambda = Lambda
        #所有属性的可能取值
        self.Datas = values
        #所有的标签
        self.Labels = {}
        for i in range(labels.__len__()):
            self.Labels[labels[i]] = 0
        #储存条件概率
        #最外层使用标签名做键，内部一层是属性分出的列表，列表的每一个元素是以属性取值为键的条件概率
        self.Probs = {}
        for label_index in range(labels.__len__()):
            self.Probs[labels[label_index]] = []
            for attr_index in range(values.__len__()):
                self.Probs[labels[label_index]].append({})
                for val_index in range(values[attr_index].__len__()):
                    self.Probs[labels[label_index]][attr_index][values[attr_index][val_index]] = 0
        
    def train(self, datas):
        for i in range(datas.__len__()):
            self.Labels[datas[i][1]] += 1
            #对一个样本的所有属性都进行更新
            if not self.Datas.__len__() == datas[i][0].__len__():
                raise Exception('\n样本的属性个数与应有的属性个数不一致！' + 
                                '\n样本编号: ' + str(i) + 
                                '\n期望属性个数: ' + str(self.Datas.__len__()) + 
                                '\n实际属性个数: ' + str(datas[i][0].__len__()))
            #一个样本会对该样本所有的所有属性的对应取值产生影响
            #因此要对属性个数进行遍历
            for attr in range(self.Datas.__len__()):
                if datas[i][0][attr] in self.Datas[attr]:
                    #对应标签的
                    self.Probs[datas[i][1]][attr][datas[i][0][attr]] += 1         
                else:
                    raise Exception('\n训练样本内的属性值不在属性列表中！'+ 
                                    '\n属性编号：' + str(i) + 
                                    '\n样本编号: ' + str(attr) + 
                                    '\n属性值：' + str(datas[i][0][attr]) +
                                    '\n属性列表值：' + str(self.Datas[attr]))
        
        #归一化与平滑处理
        for l,attr_list in self.Probs.items():
            for attr in range(attr_list.__len__()):
                for val in range(self.Datas[attr].__len__()):
                    attr_list[attr][self.Datas[attr][val]] = (attr_list[attr][self.Datas[attr][val]] + self.Lambda)/(self.Labels[l]+self.Datas[attr].__len__()*self.Lambda)
            self.Labels[l] = (self.Labels[l]+self.Lambda)/(datas.__len__() + self.Labels.__len__()*self.Lambda)
                    
    def print_probs(self):
        for label,fre in self.Labels.items():
            print(label,': ',fre,end='  ')
        print('')
        for label,attr_dict in self.Probs.items():
            for attr in range(self.Datas.__len__()):
                for val in range(self.Datas[attr].__len__()):
                    print(attr_dict[attr][self.Datas[attr][val]], end=' ')
                print('')
            print('')
    
    def predict(self, data, return_dict=False):
        if not data.__len__() == self.Datas.__len__():
            raise Exception('\n预测样本的期望的输入属性个数与实际输入的属性个数不一致！'+
                            '\n期望输入属性维度: ' + str(self.Datas.__len__()) + 
                            '\n实际输入属性维度: ' + str(data.__len__()))
        prob_dict = {}
        for l,f in self.Labels.items():
            prob_dict[l] = f
        for attr in range(self.Datas.__len__()):
            for label,fre in self.Labels.items():
                if not data[attr] in self.Datas[attr]:
                    raise Exception('\n预测样本输入的属性不在合法的属性取值内！'+
                                    '\n对应属性的编号: '+ str(attr) + 
                                    '\n合法取值序列: ' + str(self.Datas[attr]) +
                                    '\n输入样本的属性对应取值: ' + str(data[attr]))
                prob_dict[label] *= self.Probs[label][attr][data[attr]]
        return (prob_dict if return_dict else max(prob_dict, key=prob_dict.get))
        
if __name__ == '__main__':
    #统计学习方法上的数据集
    data = [[[1,'S'],-1],[[1,'M'],-1],[[1,'M'],1],[[1,'S'],1],
            [[1,'S'],-1],[[2,'S'],-1],[[2,'M'],-1],[[2,'M'],1],
            [[2,'L'],1],[[2,'L'],1],[[3,'L'],1],[[3,'M'],1],
            [[3,'M'],1],[[3,'L'],1],[[3,'L'],-1]]    
    classifier = NaiveBayes([[1,2,3],['S','M','L']], [1,-1], 1)
    first_tree = DTree([[1,2,3],['S','M','L']], [1,-1], data, 1)
    
    classifier.train(data)
    #classifier.print_probs()
    dic = classifier.predict([2,'M'])
    print(dic)
    print(first_tree.predict([2,'M']))
    
    melo_classifier = NaiveBayes([['gre','bla','whi'],['str','cur','mcu'],
                                  ['dul','blu','cri'],['cle','vag','mva'],
                                  ['ao','mao','pla'],['sol','sof']],
                                ['好瓜','坏瓜'], 1)
    #西瓜数据集
    melo_data = [[['gre','cur','blu','cle','ao','sol'],'好瓜'],
                 [['bla','cur','dul','cle','ao','sol'],'好瓜'],
                 [['bla','cur','blu','cle','ao','sol'],'好瓜'],
                 [['gre','cur','dul','cle','ao','sol'],'好瓜'],
                 [['whi','cur','blu','cle','ao','sol'],'好瓜'],
                 [['gre','mcu','blu','cle','mao','sof'],'好瓜'],
                 [['bla','mcu','blu','mva','mao','sof'],'好瓜'],
                 [['bla','mcu','blu','cle','mao','sol'],'好瓜'],
                 [['bla','mcu','dul','mva','mao','sol'],'坏瓜'],
                 [['gre','str','cri','cle','pla','sof'],'坏瓜'],
                 [['whi','str','cri','vag','pla','sol'],'坏瓜'],
                 [['whi','cur','blu','vag','pla','sof'],'坏瓜'],
                 [['gre','mcu','blu','mva','ao','sol'],'坏瓜'],
                 [['whi','mcu','dul','mva','ao','sol'],'坏瓜'],
                 [['bla','mcu','blu','cle','mao','sof'],'坏瓜'],
                 [['whi','cur','blu','vag','pla','sol'],'坏瓜'],
                 [['gre','cur','dul','mva','mao','sol'],'坏瓜']]
    melo_classifier.train(melo_data)
    #melo_classifier.print_probs()
    melo_dic = melo_classifier.predict(['gre','cur','blu','cle','ao','sol'])
    #print(melo_dic, max(melo_dic, key=melo_dic.get), sep='\n')
    tree = DTree([['gre','bla','whi'],
                  ['str','cur','mcu'],
                  ['dul','blu','cri'],
                  ['cle','vag','mva'],
                  ['ao','mao','pla'],
                  ['sol','sof']],
    ['好瓜','坏瓜'], melo_data, 1)
    print('朴素贝叶斯分类结果: ',melo_classifier.predict(['whi','cur','dul','vag','mao','sof']))
    print('决策树分类结果: ',tree.predict(['whi','cur','dul','vag','mao','sof']))
    tree.print_tree()
    

    
    
    
    
           
               
            
                    
                    
        
        