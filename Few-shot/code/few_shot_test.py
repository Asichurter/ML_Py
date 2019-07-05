import torch as t
import torch.nn.functional as F
from torch.nn import Linear,BCELoss
from utils import validate
from dataUtils import DirDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from ResNetForMalwareImage import ResNet
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

MAX_ITER = 2000
info_save_path = 'D:/ML_Py/Few-shot/doc/基于resnet18 的few-shot实验（1基础上训练4）/'
model_load_path = 'D:/peimages/one class/best_loss_model.h5'
model_save_path = 'D:/peimages/few-shot test/class4_basedon_class1/few_shot_model.h5'
train_dataset_path = 'D:/peimages/few-shot test/class4_basedon_class1/train/'
validate_dataset_path = 'D:/peimages/few-shot test/class4_basedon_class1/validate/'

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

train_set = DirDataset(train_dataset_path)
validate_set = DirDataset(validate_dataset_path)

train_loader = DataLoader(train_set, batch_size=48, shuffle=True)
train_val_loader = DataLoader(train_set, batch_size=16, shuffle=False)
validate_loader = DataLoader(validate_set, batch_size=16, shuffle=False)

Using_Frozing = True

# 使用二元交叉熵为损失函数（可以替换为交叉熵损失函数）
criteria = BCELoss()

M = t.load(model_load_path)
M = M.cuda()

print('before training, model condition is :')
ACC,LOS = validate(M, train_val_loader, criteria)
print(ACC)
print(LOS)

# # 根据resnet的论文，使用1e-4的权重衰竭
opt = t.optim.Adam(M.parameters(), lr=1e-4, weight_decay=1e-4)

# 学习率调整器，使用的是按照指标的变化进行调整的调整器
scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-5)

num = 0
best_val_loss = 0.

#**************模型冻结************
if Using_Frozing:
    for name,par in M.named_parameters():
        #冻结除以Dense开头的模以外的模型
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

x = [i+1 for i in range(MAX_ITER)]

plt.title('Accuracy')
plt.plot(x, val_acc_history, linestyle='-', color='green', label='validate')
plt.plot(x, train_acc_history, linestyle='-', color='red', label='train')
plt.legend()
plt.savefig(info_save_path+'acc.png')
plt.show()

plt.title('Loss')
plt.plot(x, val_loss_history, linestyle='--', color='green', label='validate')
plt.plot(x, train_loss_history, linestyle='--', color='red', label='train')
plt.legend()
plt.savefig(info_save_path+'loss.png')
plt.show()

acc_np = np.array(val_acc_history)
los_np = np.array(val_loss_history)

np.save(info_save_path+'acc.npy', acc_np)
np.save(info_save_path+'loss.npy', los_np)