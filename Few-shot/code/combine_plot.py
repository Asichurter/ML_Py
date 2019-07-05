import matplotlib.pyplot as plt
import numpy as np

full_acc = np.load('doc/基于resnet18的非缺省windows实验/acc.npy')
full_loss = np.load('doc/基于resnet18的非缺省windows实验/loss.npy')

single_acc = np.load('doc/基于resnet18的缺类实验/acc.npy')
single_loss = np.load('doc/基于resnet18的缺类实验/loss.npy')

multi_acc = np.load('doc/基于resnet18的多类缺省/acc.npy')
multi_loss = np.load('doc/基于resnet18的多类缺省/loss.npy')

assert len(full_acc)==len(full_loss)==len(single_acc)==len(single_loss)==\
    len(multi_acc)==len(multi_loss)==30, '数据长度不一致'

x = [i+1 for i in range(30)]

plt.title('Validate Acc Comparison')
plt.plot(x, full_acc, linestyle='-', color='red', label='full')
plt.plot(x, single_acc, linestyle='-', color='orange', label='single')
plt.plot(x, multi_acc, linestyle='-', color='green', label='multiple(n=5)')
plt.legend()
plt.savefig('doc/三种缺省的验证准确率比较.png')
plt.show()

plt.title('Validate Loss Comparison')
plt.plot(x, full_loss, linestyle='--', color='red', label='full')
plt.plot(x, single_loss, linestyle='--', color='orange', label='single')
plt.plot(x, multi_loss, linestyle='--', color='green', label='multiple(n=5)')
plt.legend()
plt.savefig('doc/三种缺省的验证损失值比较.png')
plt.show()