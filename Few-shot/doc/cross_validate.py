import torch as t
from torch.nn import BCELoss
from ResNetForMalwareImage import ResNet
from utils import validate
from dataUtils import DirDataset
from torch.utils.data import DataLoader,Dataset

model_list = ['D:/peimages/one class/best_loss_model.h5',
              'D:/peimages/one class 2/best_loss_model.h5',
              'D:/peimages/one class 3/best_loss_model.h5',
              'D:/peimages/one class 4/best_loss_model.h5']

valset_list = ['D:/peimages/one class/intern validate/',
               'D:/peimages/one class 2/intern validate/',
               'D:/peimages/one class 3/intern validate/',
               'D:/peimages/one class 4/validate/']

criteria = BCELoss()

results = {}

for i in range(len(model_list)-1):
    for j in range(i+1, len(model_list), 1):
        results['%d,%d'%(i+1,j+1)] = {}

        print('data %d th --> model %d th '%(j+1,i+1))
        model = t.load(model_list[i])
        set = DirDataset(valset_list[j])
        loader = DataLoader(set, batch_size=16, shuffle=False)
        results['%d,%d'%(i+1,j+1)]['%d<-%d'%(i+1,j+1)] = validate(model, loader, criteria)

        print('data %d th --> model %d th ' % (i+1, j+1))
        model = t.load(model_list[j])
        set = DirDataset(valset_list[i])
        loader = DataLoader(set, batch_size=16, shuffle=False)
        results['%d,%d'%(i+1,j+1)]['%d->%d'%(i+1,j+1)] = validate(model, loader, criteria)
