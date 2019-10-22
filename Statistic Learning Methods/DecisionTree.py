# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 12:46:20 2018

@author: Asichurter
"""
import math
import numpy as np
import random as rd
import matplotlib.pyplot as plt
    
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
        self.Threshold = None
        
    def is_leaf(self):
        return not self.Tag is None
    
    #收集本节点一下的所有位于子节点的数据     
    def collect_data(self):
        if not self.Tag is None:
            return self.Datas
        else:
            all_data = []
            for child in self.Children:
                all_data += child.collect_data()
            return all_data
    
    #收集本节点一下的所有子节点
    def collect_leaf_node(self):
        if not self.Tag is None:
            return [self]
        else:
            all_leaf = []
            for child in self.Children:
                all_leaf += child.collect_leaf_node()
            return all_leaf

#决策树
class DecisionTree:
    #data的数据格式：
    #[[属性...],标签]组成的一个列表
    #即[[[属性1...],标签1],[[属性2...],标签2]...]
    #All_Attr:所有属性及其所有可能的取值
    #labels: 所有可能的标签
    def __init__(self, data, alpha=0.5, criteria='C4.5', least_gain=0.1):
        #用于平衡损失函数中，熵与正则项的系数
        self.Alpha = alpha
        self.LeastGain = least_gain
        self.All_Child = []
        #self.Labels = labels
        self.Data = data
        #self.Attr = All_Attr
        self.Root = None
        self.extract_values_labels(self.Data)
        print(self.Attr)
        if criteria in ["C4.5", "ID3"]:
            self.Criteria = criteria
        else:
            assert False, "criteria must be ID3 or C4.5!"
        self.grow_tree(data, self.Root, [], True)

    def extract_values_labels(self, datas):
        '''
        遍历数据，从数据中获取所有属性的可能取值和标签
        '''
        self.Attr = [set() for i in range(len(datas[0][0]))]
        self.Labels = set()
        for item in datas:
            data,label = item
            assert len(data)==len(self.Attr), "数据维度不一致！"
            for i,val in enumerate(data):
                if type(val) is not str and type(val) is not bool:    # 发现连续值属性
                    if self.Attr[i] is not None:
                        self.Attr[i] = None    # 发现第一个连续值，将对应维度的值域设为None
                else:
                    assert self.Attr[i] is not None, "连续值和字符串同时出现在同一个维度中！dim=%d"%i
                    self.Attr[i].add(val)             # 将值添加到对应维度的值域集合中
            self.Labels.add(label)   # 将标签添加到对应维度的值域集合中

    def process_cont_val(self, datas):
        assert len(datas) != 0, "节点处无数据，不需要处理连续值！"
        cont_attr_thresh = {}
        for i,attr in enumerate(self.Attr):
            if attr is None:
                cont_attr_thresh[i] = 0   # 找到所有是连续属性的属性维度

        for attr_index in cont_attr_thresh.keys():         # 处理每个连续属性
            datas.sort(key=lambda x:x[0][attr_index])
            can_thresh = set()
            for i in range(len(datas)-1):
                can_thresh.add( (datas[i][0][attr_index] + datas[i+1][0][attr_index])/2 )   # 取排序后的每个值的中点值作为待选截断点

            can_thresh_entro = {}
            for can_item in can_thresh:
                can_thresh_entro[can_item],_ = self.cal_entro_ratio(attr_index,datas,cont_thresh=can_item) # 计算每个截断值的信息增益

            cont_attr_thresh[attr_index] = max(can_thresh_entro, key=can_thresh_entro.get)  # 将对应连续值属性的截断值设置为信息增益最大的截断值

        cont_attr_thresh.setdefault(None)
        return cont_attr_thresh

    def grow_tree(self, datas, node, depre_attr, root=False):
        '''建立决策树的递归方法
        datas:当前节点的数据
        node:当前的节点
        depre_attr:当前节点之前已经被用掉的属性的下标
        root:是否是在建立根节点
        注意：
        1.建树的原则是：将数据data全部存储在叶节点，因此内部节点实际上是没有数据的
        2.内部节点和叶节点公用一个节点类，内部节点使用Attribute属性来标识该节点的判断依据的属性下标
        叶节点使用Tag属性来表示叶节点对应的标签，所有节点的Value属性来表示父节点到本节点的属性取值'''
        if root:
            check_res = self.check_datas_type(datas)
            if check_res is None:
                self.Root = Node(None)
                self.grow_tree(datas, self.Root, [])
            #如果建树的根节点的类本来就是同一类的话，则不会递归调用
            else:
                self.Root = Node(None, datas, check_res)
        else:
            def make_leaf(myself):
                myself.All_Child.append(node)
                node.Datas = datas
                node.Tag = myself.find_majority(datas, myself.Data)

            attr_thresh_dict = self.process_cont_val(datas)           # 获取连续值属性及其对应的截断值
            assert len(set(attr_thresh_dict.keys()).intersection(set(depre_attr)))==0, \
                "连续值属性加入到了弃用列表中！弃用列表:%s，连续值属性:%s"%(depre_attr,attr_thresh_dict.keys())
            #如果本节点无数据，则使用整棵树的数据来进行多数表决，同时停止递归
            if len(datas) == 0:
                node.Tag = self.find_majority(datas, self.Data)
            else:
                check_res = self.check_datas_type(datas)
                #如果发现当前节点的类还有不同，同时属性还没有用完
                #则找到信息增益比最大的属性，划分后递归调用
                if check_res is None and len(depre_attr) != len(self.Attr):
                    attr_ratio = []
                    attr_entro = []
                    attr_above_mean = {}
                    for i in range(len(self.Attr)):
                        if i not in depre_attr:
                            g,r = self.cal_entro_ratio(i, datas, cont_thresh=attr_thresh_dict.get(i))  # 如果该属性属于连续值属性，则将对应的截断值输入
                            attr_ratio.append(r)
                            attr_entro.append(g)
                        #就算属性已经被废弃使用但还是要填入占位，因为属性是基于下标的
                        else:
                            attr_ratio.append(0)
                            attr_entro.append(0)
                    ratio_mean = np.mean(attr_ratio)
                    #由于信息增益比倾向于采用取值少的属性，而直接使用信息增益又偏向于取值多的属性
                    #因此将信息增益比作为启发值，选取高于平均信息增益比中，信息增益最大的属性(来自西瓜书)
                    #print(attr_entro, attr_ratio, sep='\n')
                    for i,r in enumerate(attr_ratio):
                        if r >= ratio_mean:
                            attr_above_mean[i] = attr_entro[i]
                    #根据划分依据来选择最大属性
                    #如果是ID3，选择信息增益最大的属性
                    #如果是C4.5，选择信息增益比最大的属性
                    if self.Criteria == "C4.5":
                        max_attr = max(attr_above_mean, key=attr_above_mean.get)
                        entro_gain = max(attr_above_mean)
                    else:
                        max_attr = attr_entro.index(max(attr_entro))
                        entro_gain = max(attr_entro)
                        # print(max(attr_entro))

                    if entro_gain < self.LeastGain:      # 若信息增益（率）小于阈值，则将当前制作为叶节点并且直接停止递归
                        make_leaf(self)
                        return

                    #将本节点的分支属性设置为信息增益比例最大的一个属性，以下标的形式
                    node.Attribute = max_attr
                    # 如果是连续值属性，在划分的时候也要输入截断值
                    for v,dat in self.partition_by_attr_val(max_attr, datas, cont_thresh=attr_thresh_dict.get(max_attr)).items():
                        child_node = Node(node)
                        #将对应的子节点的分支取值设置为对应的值
                        child_node.Value = v
                        child_node.Threshold = attr_thresh_dict.get(max_attr)
                        node.Children.append(child_node)
                        # 如果使用的是连续值属性，则不加入到弃用列表中
                        new_depre_attr = depre_attr+[max_attr] if max_attr not in attr_thresh_dict.keys() else depre_attr
                        #递归调用，不同的是属性的弃用表需要加上当前用于分类的属性
                        self.grow_tree(dat, child_node, new_depre_attr)
                        
                #否则，当前的这个node为叶节点，设置叶节点的属性，将数据储存在其中
                else:
                    make_leaf(self)
                    # self.All_Child.append(node)
                    # node.Datas = datas
                    # node.Tag = self.find_majority(datas, self.Data)
             
    #利用公式，gR(D,A)=g(D,A)/HA(D)来计算信息增益比
    #本建树算法采用的是改进后的ID3，即C4.5
    #attr:计算的属性的下标
    def cal_entro_ratio(self, attr, datas, cont_thresh=None):
        assert not (self.Attr[attr] is None and cont_thresh is None), "下标为%d的属性为连续值，但是没有提供截断值！"%attr
        attr_entro = self.cal_entro('attrs', datas, attr, cont_thresh=cont_thresh)
        #如果发现属性值的熵为0，代表就算当前所有数据都是属于这个属性的，那么使用这个属性来分割不会得到任何收益
        if attr_entro == 0:
            return 0,0
        data_entro = self.cal_entro('types', datas)
        datas_split = {at:[] for at in self.Attr[attr]} if cont_thresh is None else {False:[], True:[]}
        for data in datas:
            if cont_thresh is None:
                datas_split[data[0][attr]].append(data)    # 将数据按照某属性值进行分入列表
            else:
                datas_split[data[0][attr] >= cont_thresh].append(data)   # 对连续值，将将数据按照属性值是否大于等于截断值划分入列表中
        attr_data_entro = 0
        for at,dat in datas_split.items():
            attr_data_entro += len(dat)/len(datas)*self.cal_entro('types', dat)    # 将按属性分开的数据计算类别熵
        return data_entro - attr_data_entro,(data_entro - attr_data_entro)/attr_entro  
        #return data_entro - attr_data_entro               
    
    #计算当前数据中，依类别分布的信息熵或者是以属性值分布的信息熵
    #attr_type: 用于指示到底是对类别分布还是属性值分布求信息熵
    #attr_type==types：计算按类型分布的信息熵
    #attr_type==attrs: 计算按属性值分布的信息熵
    #attr:如果是按属性值分布求熵的话，想要计算的属性的下标
    def cal_entro(self, attr_type, datas, attr=None, cont_thresh=None):
        length = len(datas)
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
            if attr is None:
                raise Exception('\n计算熵时，指定为计算数据集关于特征attr的熵，但是没有指明attr！')
            if attr >= self.Attr.__len__() or attr < 0 or not type(attr) == int:
                raise Exception('\n计算熵时，指定为计算数据集关于特征attr的熵，但是指定了非法的attr下标！'+
                                '\n指定下标：' + str(attr))
            attrs = {}
            if cont_thresh is None:
                for at in self.Attr[attr]:
                    attrs[at] = 0
            else:
                attrs[False] = 0
                attrs[True] = 0     # 如果为连续值，则只指定大于等于或者小于的情况两个值

            for data in datas:
                if not data.__len__() == 2 or not data[0].__len__() == self.Attr.__len__():
                    raise Exception('\n输入向量维度与预期不一致!' + 
                                    '\n非法向量: ' + str(data))
                #先进行计数
                if cont_thresh is None:
                    if data[0][attr] in self.Attr[attr]:
                        attrs[data[0][attr]] += 1
                    else:
                        raise Exception('\n给定的数据中，存在不在合理属性值列表中的非法属性取值!' +
                                        '\n数据: ' + str(data) +
                                        '\n属性下标: ' + str(attr) +
                                        '\n合理取值: ' + str(self.Attr[attr]))
                else:
                    attrs[data[0][attr] >= cont_thresh] += 1    # 连续值时，根据是否大于等于截断值将数据分为两个类

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
    def partition_by_attr_val(self, attr, datas, cont_thresh=None):
        res = {v:[] for i,v in enumerate(self.Attr[attr])} if cont_thresh is None else \
                {True:[], False:[]}
        for i,data in enumerate(datas):
            if not data.__len__() == 2 or not data[0].__len__() == self.Attr.__len__():
                raise Exception('\n输入的数据中，维度与预期的不一致!'+
                                '\n数据下标: ' + str(i))
            if cont_thresh is None:
                res[data[0][attr]].append(data)
            else:
                res[data[0][attr] >= cont_thresh].append(data)
        return res
    
    #寻找数据集中的占大多数的标签，这是多数表决时调用的方法
    #hyper_datas:当data为空的时候，用于多数表决建立叶节点标签的超数据集，这个取值通常为整棵树的数据集
    def find_majority(self, datas, hyper_datas=None):
        label_dic = {l:0 for l in self.Labels}
        if datas.__len__() == 0:
            #hyper_datas是在节点无数据分划的时候，使用整棵树的数据进行多数表决
            if not hyper_datas is None:
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
        if node is None:
            node = self.Root
        print('')
        print('层次: ' + str(hierachy))
        print('节点属性值: ' + str(node.Value))
        print('节点截断值: ' + str(node.Threshold))
        print("----------------------------------")
        print('节点属性划分: ' + str(node.Attribute))
        if not node.Tag is None:
            print('叶节点的数据: ' + str(node.Datas))
            print('叶节点标签: ' + str(node.Tag))
        else:
            for i,child in enumerate(node.Children):
                self.print_tree(child, hierachy+[i])
       
    #利用数据进行预测
    #允许多个数据同时进行预测，因此输入应该是一个二维数组
    def predict(self, datas):
        cont_attrs = [index for index in range(len(self.Attr)) if self.Attr[index] is None]  # 获取所有连续值属性下标
        def cont_comp(thresh, drt, val):
            return (drt and val >= thresh) | (not drt and val < thresh)

        results = []
        for d in datas:
            assert len(d)==len(self.Attr), "预期数据维度：%d 不合法的属性维度：%d" %(len(self.Attr),len(d))
            for i,a in enumerate(d):
                assert self.Attr[i] is None or a in self.Attr[i], \
                    "数据的属性值不在预期内，不合法的属性值: " + a + " 合法属性值：" + self.Attr[i]
            cur = self.Root
            while not cur.is_leaf():
                for child in cur.Children:
                    if cur.Attribute in cont_attrs:
                        if cont_comp(child.Threshold, child.Value, d[cur.Attribute]):
                            cur = child
                            break
                    elif d[cur.Attribute] == child.Value:
                        cur = child
                        break
            results.append(cur.Tag)
        return results
    
    #根据损失函数计算剪枝前后的损失
    #after_pruning:剪枝前还是后
    #这将决定是用父节点下的所有数据的分布计算熵还是对每一个取值单独计算熵
    def cal_loss(self, node, after_pruning=False):
        if not node.Tag is None:
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
                if not leaf.Parent is None and leaf.Parent not in close_list:
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

def get_square_region_data(x, y, x_w, y_w, num, label):
    datas = []
    for i in range(num):
        data_x = rd.uniform(x,x+x_w)
        data_y = rd.uniform(y,y+y_w)
        datas.append([[data_x,data_y],label])
    return datas
            
if __name__ == '__main__':
    disease_data = [
        [['Y','U',True,True],False],
        [['Y','U',True,False],False],
        [['M','U',True,True],True],
        [['O','M',True,True],True],
        [['O','O',False,True],True],
        [['O','O',False,False],False],
        [['M','O',False,False],True],
        [['Y','M',True,True],False],
        [['Y','O',False,True],True],
        [['O','M',False,True],True],
        [['Y','M',False,False],True],
        [['M','M',True,False],True],
        [['M','U',False,True],True],
        [['O','M',True,False],False]
    ]
    loan_data = np.array([
        [['y',False,False,'f'],False],
        [['y',False,False,'g'],False],
        [['y',True,False,'g'],True],
        [['y',True,True,'f'],True],
        [['y',False,False,'f'],False],
        [['m',False,False,'f'],False],
        [['m',False,False,'g'],False],
        [['m',True,True,'g'],True],
        [['m',False,True,'e'],True],
        [['m',False,True,'e'],True],
        [['o',False,True,'e'],True],
        [['o',False,True,'g'],True],
        [['o',True,False,'g'],True],
        [['o',True,False,'e'],True],
        [['o',False,False,'f'],False]
    ])

    cont_data = []
    cont_data += get_square_region_data(0,0,5,5,20,True)
    cont_data += get_square_region_data(-5,2,5,3,10,True)
    cont_data += get_square_region_data(-5,0,5,2,10,False)
    cont_data += get_square_region_data(-5,-5,5,5,20,False)
    cont_data += get_square_region_data(0,-5,5,5,20,False)

    cont_test_data = []
    cont_test_data += get_square_region_data(0,0,5,5,20,True)
    cont_test_data += get_square_region_data(-5,2,5,3,10,True)
    cont_test_data += get_square_region_data(-5,0,5,2,10,False)
    cont_test_data += get_square_region_data(-5,-5,5,5,20,False)
    cont_test_data += get_square_region_data(0,-5,5,5,20,False)

    cont_tree = DecisionTree(cont_data, least_gain=1e-2, criteria='ID3')
    cont_tree.print_tree()

    plt.scatter([x[0][0] for x in cont_data if x[1]], [x[0][1] for x in cont_data if x[1]], color='red')
    plt.scatter([x[0][0] for x in cont_data if not x[1]], [x[0][1] for x in cont_data if not x[1]], color='blue')
    # plt.show()

    correct_counter = 0
    for item in cont_test_data:
        test_data,test_label = item
        pred = cont_tree.predict([test_data])[0]
        if pred == test_label:
            correct_counter += 1
    print("acc: %f"%(correct_counter/len(cont_test_data)))

    # disease_tree = DecisionTree(disease_data)
    # disease_tree.print_tree()
    # print('*****************************************************')
    # disease_tree.pruning_with_lossFunc()
    # disease_tree.print_tree()
    # loan_tree = DecisionTree(loan_data, criteria='ID3')
    # loan_tree.print_tree()

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        