import torch as t
from ResNetForMalwareImage import ResNet
from utils import validate
from dataUtils import DirDataset
from torch.utils.data import DataLoader

save_path = 'D:/ML_Py/Few-shot/doc/基于resnet18的单类实验/'
name = 'class_intern'

model = t.load('doc/基于resnet18的单类实验(LdPinch)/best_loss_model.h5')
dataset = DirDataset(r'D:/peimages/validate/')
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

acc,loss = validate(model, dataloader, t.nn.BCELoss())
print(acc)
print(loss)