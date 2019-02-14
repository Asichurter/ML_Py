# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 12:46:20 2018

@author: Asichurter
"""
import math
import numpy as np
    
#决策树节点类    
class Node:
    def __init__(self, parent, datas=None, tag=None):
        self.Parent = parent
        self.Children = []
        self.Datas = datas
        #叶节点的标签
        self.Tag = tag
        #本节点作为决策划分，依据的属性
        self.Attribute = None
        #本节点作为决策划分，依据的属性的取值
        self.Value = None
        
    def is_leaf(self):
        return not self.Tag == None
    
    #收集本节点一下的所有位于子节点的数据     
    def collect_data(self):
        if not self.Tag == None:
            return self.Datas
        else:
            all_data = []
            for child in self.Children:
                all_data += child.collect_data()
            return all_data
    
    #收集本节点一下的所有子节点
    def collect_leaf_node(self):
        if not self.Tag == None:
            return [self]
        else:
            all_leaf = []
            for child in self.Children:
                all_leaf += child.collect_leaf_node()
            return all_leaf

#决策树
class DecisionTree:
    #老样子，data的数据格式：
    #[[属性...],标签]组成的一个列表
    #即[[[属性1...],标签1],[[属性2...],标签2]...]
    #All_Attr:所有属性及其所有可能的取值
    #labels: 所有可能的标签
    def __init__(self, All_Attr, labels, data, alpha=0.5):
        #用于平衡损失函数中，熵与正则项的系数
        self.Alpha = alpha
        self.All_Child = []
        self.Labels = labels
        self.Data = data
        self.Attr = All_Attr
        self.Root = None
        self.grow_tree(data, self.Root, [], True)
        
     
    #建立决策树的递归方法
    #datas:当前节点的数据
    #node:当前的节点
    #depre_attr:当前节点之前已经被用掉的属性的下标
    #root:是否是在建立根节点
    #注意：
    #1.建树的原则是：将数据data全部存储在叶节点，因此内部节点实际上是没有数据的
    #2.内部节点和叶节点公用一个节点类，内部节点使用Attribute属性来标识该节点的判断依据的属性下标
    #叶节点使用Tag属性来表示叶节点对应的标签，所有节点的Value属性来表示父节点到本节点的属性取值
    def grow_tree(self, datas, node, depre_attr, root=False):
        if root:
            check_res = self.check_datas_type(datas)
            if check_res == None:
                self.Root = Node(None)
                self.grow_tree(datas, self.Root, [])
            #如果建树的根节点的类本来就是同一类的话，则不会递归调用
            else:
                self.Root = Node(None, datas, check_res)
        else:
            #如果本节点无数据，则使用整棵树的数据来进行多数表决，同时停止递归
            if datas.__len__() == 0:
                node.Tag = self.find_majority(datas, self.Data)
            else:
                check_res = self.check_datas_type(datas)
                #如果发现当前节点的类还有不同，同时属性还没有用完
                #则找到信息增益比最大的属性，划分后递归调用
                if check_res == None and not depre_attr.__len__() == self.Attr.__len__():
                    attr_ratio = []
                    attr_entro = []
                    attr_above_mean = {}
                    for i in range(self.Attr.__len__()):
                        if i not in depre_attr:
                            g,r = self.cal_entro_ratio(i, datas)
                            attr_ratio.append(r)
                            attr_entro.append(g)
                        #就算属性已经被废弃使用但还是要填入占位，因为属性是基于下标的
                        else:
                            attr_ratio.append(0)
                            attr_entro.append(0)
                    ratio_mean = np.mean(attr_ratio)
                    #由于信息增益比倾向于采用取值少的属性，而直接使用信息增益又偏向于取值多的属性
                    #因此将信息增益比作为启发值，选取高于平均信息增益比中，信息增益最大的属性
                    #print(attr_entro, attr_ratio, sep='\n')
                    for i,r in enumerate(attr_ratio):
                        if r >= ratio_mean:
                            attr_above_mean[i] = attr_entro[i]
                    max_attr = max(attr_above_mean, key=attr_above_mean.get)
                    #将本节点的分支属性设置为信息增益比例最大的一个属性，以下标的形式
                    node.Attribute = max_attr
                    for v,dat in self.partition_by_attr_val(max_attr, datas).items():
                        child_node = Node(node)
                        #将对应的子节点的分支取值设置为对应的值
                        child_node.Value = v
                        node.Children.append(child_node)
                        #递归调用，不同的是属性的弃用表需要加上当前用于分类的属性
                        self.grow_tree(dat, child_node, depre_attr+[max_attr])
                        
                #否则，当前的这个node为叶节点，设置叶节点的属性，将数据储存在其中
                else:
                    self.All_Child.append(node)
                    node.Datas = datas
                    node.Tag = self.find_majority(datas, self.Data)
             
    #利用公式，gR(D,A)=g(D,A)/HA(D)来计算信息增益比
    #本建树算法采用的是改进后的ID3，即C4.5
    #attr:计算的属性的下标
    def cal_entro_ratio(self, attr, datas):
        attr_entro = self.cal_entro('attrs', datas, attr)
        #如果发现属性值的熵为0，代表就算当前所有数据都是属于这个属性的，那么使用这个属性来分割不会得到任何收益
        if attr_entro == 0:
            return 0,0
        data_entro = self.cal_entro('types', datas)
        datas_split = {at:[] for at in self.Attr[attr]}
        for data in datas:
            datas_split[data[0][attr]].append(data)
        attr_data_entro = 0
        for at,dat in datas_split.items():
            attr_data_entro += dat.__len__()/datas.__len__()*self.cal_entro('types', dat)
        return data_entro - attr_data_entro,(data_entro - attr_data_entro)/attr_entro  
        #return data_entro - attr_data_entro               
    
    #计算当前数据中，依类别分布的信息熵或者是以属性值分布的信息熵
    #attr_type: 用于指示到底是对类别分布还是属性值分布求信息熵
    #attr_type==types：计算按类型分布的信息熵
    #attr_type==attrs: 计算按属性值分布的信息熵
    #attr:如果是按属性值分布求熵的话，想要计算的属性的下标
    def cal_entro(self, attr_type, datas, attr=None):
        length = datas.__len__()
        total = 0
        if attr_type == 'types':
            types = {}
            for label in self.Labels:
                types[label] = 0
            #先进行计数
            for data in datas:
                types[data[1]] += 1
            for label,num in types.items():
                if not num == 0:
                    total += (num/length)*math.log2(num/length)
            return total*-1
        
        elif attr_type == 'attrs':
            if attr == None:
                raise Exception('\n计算熵时，指定为计算数据集关于特征attr的熵，但是没有指明attr！')
            if attr >= self.Attr.__len__() or attr < 0 or not type(attr) == int:
                raise Exception('\n计算熵时，指定为计算数据集关于特征attr的熵，但是指定了非法的attr下标！'+
                                '\n指定下标：' + str(attr))
            attrs = {}
            for at in self.Attr[attr]:
                attrs[at] = 0
            for data in datas:
                if not data.__len__() == 2 or not data[0].__len__() == self.Attr.__len__():
                    raise Exception('\n输入向量维度与预期不一致!' + 
                                    '\n非法向量: ' + str(data))
                #先进行计数
                if data[0][attr] in self.Attr[attr]:
                    attrs[data[0][attr]] += 1
                else:
                    raise Exception('\n给定的数据中，存在不在合理属性值列表中的非法属性取值!'+
                                     '\n数据: ' + str(data) +
                                     '\n属性下标: '+ str(attr) + 
                                     '\n合理取值: '+ str(self.Attr[attr]))
            for at,num in attrs.items():
                #0log0定义为0
                if not num == 0:
                    total += (num/length)*math.log2(num/length)
            return -1*total
        
        else:
            raise Exception('\n在计算熵时，指定了非法的计算依据！' +
                            '\n合理值：[types, attrs]'+
                            '\n输入值: ' + attr_type)
        
    #检查数据中，是否所有数据都属于同一类的收敛情况
    #如果是的话，将会返回这个收敛值，否则返回None        
    def check_datas_type(self, datas):
        if datas.__len__() == 1:
            return True
        else:
            pivot = datas[0][1]
            for data in datas:
                if not data[1] == pivot:
                    return None
            return pivot
    
    #依据给定的属性，将数据按照属性值进行划分
    #返回的类型是一个字典，字典的键是该属性的对应取值，值是对应取值的所有数据      
    #attr:用于划分数据集的属性下标      
    def partition_by_attr_val(self, attr, datas):
        res = {v:[] for i,v in enumerate(self.Attr[attr])}
        for i,data in enumerate(datas):
            if not data.__len__() == 2 or not data[0].__len__() == self.Attr.__len__():
                raise Exception('\n输入的数据中，维度与预期的不一致!'+
                                '\n数据下标: ' + str(i))
            res[data[0][attr]].append(data)
        return res
    
    #寻找数据集中的占大多数的标签，这是多数表决时调用的方法
    #hyper_datas:当data为空的时候，用于多数表决建立叶节点标签的超数据集，这个取值通常为整棵树的数据集
    def find_majority(self, datas, hyper_datas=None):
        label_dic = {l:0 for l in self.Labels}
        if datas.__len__() == 0:
            #hyper_datas是在节点无数据分划的时候，使用整棵树的数据进行多数表决
            if not hyper_datas == None:
                for dat in hyper_datas:
                    label_dic[dat[1]] += 1
            else:
                raise Exception('在无数据节点上，没有传递hyper_data来进行多数表决！')
        else:
            for dat in datas:
                label_dic[dat[1]] += 1
        return max(label_dic, key=label_dic.get)
    
    #递归调用的打印树的方法
    #hierachy：层次列表，每一个取值代表经过了一个层，值的含义是通过的是该层的哪一个子节点
    def print_tree(self, node=None, hierachy=[]):
        if node == None:
            node = self.Root
        print('')
        print('层次: ' + str(hierachy))
        print('节点属性划分: ' + str(node.Attribute))
        print('节点属性值: ' + str(node.Value))
        if not node.Tag == None:
            print('叶节点的数据: ' + str(node.Datas))
            print('叶节点标签: ' + str(node.Tag))
        else:
            for i,child in enumerate(node.Children):
                self.print_tree(child, hierachy+[i])
       
    #利用数据进行预测         
    def predict(self, data):
        if not data.__len__() == self.Attr.__len__():
            raise Exception('\n预测时，输入维度与预期维度不一致！' + 
                            '\n输入的维度: ' + str(data.__len__()) + 
                            '\n期望的维度: ' + str(self.Attr.__len__()))
        else:
            for i,at in enumerate(data):
                if at not in self.Attr[i]:
                    raise Exception('\n预测时，输入的属性不在合法的属性表内！' + 
                                    '\n属性下标: ' + str(i) + 
                                    '\n合法的输入: ' + str(self.Attr[i]) + 
                                    '\n实际的输入: ' + str(at))
            cur = self.Root
            while not cur.is_leaf():
                for child in cur.Children:
                    if data[cur.Attribute] == child.Value:
                        cur = child
                        break
            return cur.Tag
    
    #根据损失函数计算剪枝前后的损失
    #after_pruning:剪枝前还是后
    #这将决定是用父节点下的所有数据的分布计算熵还是对每一个取值单独计算熵
    def cal_loss(self, node, after_pruning=False):
        if not node.Tag == None:
            return self.cal_entro('types', node.Datas) + self.Alpha
        else:
            if not after_pruning:
                loss = 0
                all_leaf = node.collect_leaf_node()
                for leaf in all_leaf:
                    loss += leaf.Datas.__len__()*self.cal_entro('types', leaf.Datas)
                return loss + self.Alpha*all_leaf.__len__()
            else:
                all_data = node.collect_data()
                return all_data.__len__() * self.cal_entro('types', all_data) + self.Alpha
    
    #使用损失函数为指标进行后剪枝，该损失函数是添加了正则项的经验风险函数
    def pruning_with_lossFunc(self):
        #本轮循环中，弃用的节点表
        #一般来说，是已经剪枝过后的节点的父节点，或者是不可能剪枝节点的父节点
        close_list = []
        #本轮待搜索的叶节点
        open_list = []
        #填充搜索的叶节点
        open_list.extend(self.Root.collect_leaf_node())
        loop_control = True
        #当没有一个节点进行了剪枝时，循环结束
        while loop_control:
            loop_control = False
            #清除列表重新填充
            open_list.clear()
            open_list.extend(self.Root.collect_leaf_node())
            close_list.clear()
            for leaf in open_list:
                #如果当前的叶节点不是根节点，而且不在弃用表中才会进行搜索
                if not leaf.Parent == None and leaf.Parent not in close_list:
                    loss_before = self.cal_loss(leaf.Parent, False)
                    loss_after = self.cal_loss(leaf.Parent, True)
                    #如果剪枝以后的损失函数更小，则进行剪枝
                    if loss_after <= loss_before:        
                    #if leaf.Parent.Attribute == 2:
                    #剪枝的具体操作：
                    #1.将父节点的Data置为其下所有叶节点的Data，并利用这些Data多数表决来决定Tag
                    #2.将Attribute置为None，代表这个是叶节点，不可再分，同时清除子节点
                    #3.改变循环控制变量，以便递归的对新的节点的父节点进行检索
                        parent = leaf.Parent
                        pre_data = parent.collect_data()
                        parent.Tag = self.find_majority(pre_data)
                        parent.Datas = pre_data
                        parent.Attribute = None
                        parent.Children.clear() 
                        loop_control = True
                    #不论是否对这个节点进行了剪枝，都应该加入到弃用表中
                    #理由：
                    #1.如果剪枝了，则子节点的应该消失，或者说根本不应该检索
                    #2.如果没有剪枝，则同一父节点的叶节点就没有必要再对同一个父节点进行检索了，因为两者完全相同
                    close_list.append(leaf.Parent) 
            
if __name__ == '__main__':
    data = [[['Y',False,False,'S'],False], 
            [['Y',False,False,'G'],False],
            [['Y',True,False,'G'],True],
            [['Y',True,True,'S'],True],
            [['Y',False,False,'S'],False],
            [['M',False,False,'S'],False],
            [['M',False,False,'G'],False],
            [['M',True,True,'G'],True],
            [['M',False,True,'VG'],True],
            [['M',False,True,'VG'],True],
            [['O',False,True,'VG'],True],
            [['O',False,True,'G'],True],
            [['O',True,False,'G'],True],
            [['O',True,False,'VG'],True],
            [['O',False,False,'S'],False]]
    
    tree = DecisionTree([['Y','M','O'],[False, True],[False, True],['S','G','VG']], [True, False], data)
    tree.print_tree()
    print(tree.predict(['O',False,False,'S']))
    #tree.pruning_with_lossFunc()
    #print('\n******************剪枝前后的分界线*********************')  
    #tree.print_tree()  
    
    '''
    a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    b = np.array([[2,2,2,1],[2,2,2,1],[2,2,2,1]])
    #print(a*b)
    c = np.array([[2],[2]])
    d = c[0]
    print(a[0].shape[0])
    if c[0] == c[1]:
        print('1')
    '''

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        