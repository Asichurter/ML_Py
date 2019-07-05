import torch as t
import torch.nn.functional as F
from torch.nn import Linear,BCELoss
from utils import validate
from dataUtils import DirDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from ResNetForMalwareImage import ResNet

MAX_ITER = 30
model_load_path = 'D:/peimages/one class/best_loss_model.h5'
model_save_path = 'D:/peimages/fine-tuning/1适应4/fine-tuning_model_4to1.h5'
train_dataset_path = 'D:/peimages/fine-tuning/1适应4/train/'
validate_dataset_path = 'D:/peimages/fine-tuning/1适应4/validate/'
test_dataset_path = 'D:/peimages/fine-tuning/1适应4/test/'

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

train_set = DirDataset(train_dataset_path)
validate_set = DirDataset(validate_dataset_path)
test_set = DirDataset(test_dataset_path)

train_loader = DataLoader(train_set, batch_size=48, shuffle=True)
train_val_loader = DataLoader(train_set, batch_size=16, shuffle=False)
validate_loader = DataLoader(validate_set, batch_size=16, shuffle=False)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

# 使用二元交叉熵为损失函数（可以替换为交叉熵损失函数）
criteria = BCELoss()

class FineTuningModel(t.nn.Module):
    def __init__(self, pre_model, out, ft=False):
        super(FineTuningModel, self).__init__()
        self.ResNet = pre_model
        self.Dense = Linear(512, out)
        if ft:
            for par in self.ResNet.parameters():
                par.requires_grad = False

    def forward(self, x):
        x = self.ResNet.Ahead(x)
        x = self.ResNet.Layer1(x)
        x = self.ResNet.Layer2(x)
        x = self.ResNet.Layer3(x)
        x = self.ResNet.Layer4(x)
        #256/4/2/2/2=8,变成1的话需要长度为8的平均池化
        x = F.avg_pool2d(x, 8)
        #将样本整理为(批大小，1)的形状
        x = x.view(x.shape[0], -1)
        x = self.Dense(x)
        return F.softmax(x, dim=1)

M = t.load(model_load_path)#, map_location=lambda storage, loc: storage
# M = ResNet(1).cuda()
# M.load_state_dict(m)

print('before training, model condition is :')
ACC,LOS = validate(M, train_val_loader, criteria)
print(ACC)
print(LOS)

#model = FineTuningModel(M, out=2)
#
# input('---------------')
#model = model.cuda()
#
# # 根据resnet的论文，使用1e-4的权重衰竭
opt = t.optim.Adam(M.parameters(), lr=1e-4, weight_decay=1e-4)
# for name,par in M.named_parameters():
#     print(name, par.shape)

# opt = t.optim.SGD(pars, lr=1e-2, momentum=0.9, weight_decay=0.2, nesterov=True)

# 学习率调整器，使用的是按照指标的变化进行调整的调整器
#scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-5)
# criteria = t.nn.CrossEntropyLoss()
num = 0
best_val_loss = 0.

#**************模型冻结************
for name,par in M.named_parameters():
    if not name.startswith('Dense'):
        par.requires_grad = False
    else:
        par.requires_grad = True
# *******************************

print('training...')
for i in range(MAX_ITER):
    print(i, ' th')
    a = 0
    c = 0
    Loss = 0.

    # 将模型调整为学习状态
    M.train()
    for datas, l in train_loader:
        opt.zero_grad()
        datas = datas.cuda()

        # 创建可以输入到损失函数的float类型标签batch
        labels = [[1, 0] if L == 0 else [0, 1] for L in l]
        labels = t.FloatTensor(labels).cuda()

        out = M(datas).squeeze()
        loss = criteria(out, labels).cuda()
        loss.backward()
        opt.step()

        # 计算损失和准确率
        Loss += loss.data.item()
        # 进行与实际标签的比较时，由于标签是LongTensor类型，因此转化
        # 选用值高的一个作为预测结果
        predict = t.LongTensor([0 if x[0] >= x[1] else 1 for x in out])
        a += predict.shape[0]
        c += (predict == l).sum().item()
    print('train loss: ', Loss)
    train_loss_history.append(Loss)
    print('train acc: ', c / a)
    train_acc_history.append(c / a)

    # val_acc,val_loss = validate(resnet, test_loader, criteria)
    # print('val loss: ', val_loss)
    # val_loss_history.append(val_loss)
    # print('val accL: ', val_acc)
    # val_acc_history.append(val_acc)

    val_acc, val_loss = validate(M, validate_loader, criteria)
    print('intern val loss: ', val_loss)
    val_loss_history.append(val_loss)
    print('intern val acc: ', val_acc)
    val_acc_history.append(val_acc)

    if len(val_loss_history) == 1 or val_loss < best_val_loss:
        best_val_loss = val_loss
        t.save(M, model_save_path)
        print('save model at epoch %d' % i)

    num += 1

print('testing...')
test_acc,test_loss = validate(M, test_loader, criteria)
print(test_acc)
print(test_loss)