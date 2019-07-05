import torch as t
from ResNetForMalwareImage import ResNet
from utils import validate
from dataUtils import DirDataset
from torch.utils.data import DataLoader

model_path_1 = 'D:/peimages/few-shot test/class4_basedon_class1/few_shot_model.h5'
model_path_2 = 'D:/peimages/one class 1/best_loss_model.h5'
val_path_1 = 'D:/peimages/few-shot test/class4_basedon_class1/validate/'
val_path_2 = 'D:/peimages/one class/intern validate/'

model = t.load(model_path_1)
dataset = DirDataset(val_path_1)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

acc,loss = validate(model, dataloader, t.nn.BCELoss())
print(acc)
print(loss)