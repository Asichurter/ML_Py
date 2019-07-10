import math
import numpy as np
import random as rd

class K_Median:
    def __init__(self, data, k=3, metric='Euc'):
        self.K = k
        assert metric in ['Euc', 'Man'], "指定的距离度量不在合法取值内！"
        if metric == "Euc":
            self.Metric = lambda x,y: math.sqrt(np.sum((np.array(x)-np.array(y))**2))
        else:
            self.Metric = lambda x,y: np.sum(np.absolute(np.array(x)-np.array(y)))
        self.Data = data
        self.Clusters = [[] for i in range(k)]
        self.Centers = []
        #随机从数据点中抽取k个为初始聚类中心
        for i in range(k):
            self.Centers.append(self.Data[rd.randint(0,len(self.Data)-1)])

    def fit(self, max_iter=1000, epsilon=1e-10):
        for i in range(max_iter):
            for j in range(self.K):
                self.Clusters[j].clear()
            for d in self.Data:
                dists = []
                #填充距离列表
                for k in range(self.K):
                    dists.append(self.Metric(d, self.Centers[k]))
                #利用距离列表找到最近距离的中心点
                self.Clusters[dists.index(min(dists))].append(d)
            shift = 0.
            #利用数据的中位数修正聚类中心
            for k in range(self.K):
                assert len(self.Clusters[k])>0, \
                    "聚类中心:(%s) 没有对应的数据点！" % str(self.Centers[k])
                new_center = np.median(self.Clusters[k], axis=0)
                shift += np.sum(new_center-self.Centers[k])
                self.Centers[k] = new_center
            if shift <= epsilon:
                break
    
    


