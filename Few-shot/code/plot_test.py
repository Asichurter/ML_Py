import matplotlib.pyplot as plt
import numpy as np
import random as rd

'''
a = [2,3]
var = [0.5, 1]
width = 0.4
x = [1,2]

plt.xlim(0,3)
plt.bar(x, a, width, yerr=var, color='orange')
plt.show()'''

acc_matrix = np.array([[1,0.985,0.988,0.833],
                   [0.946,1,0.94,0.812],
                   [0.967,0.977,1,0.842],
                   [0.982,0.975,0.958,1]])
loss_matrix = np.array([[0,3.95,2.59,29.92],
                        [6.99,0,7.99,21.98],
                        [3.28,3.22,0,13.01],
                        [1.95,3.24,3.34,0]])

labels = ['trojan.LdPinch','trojan.OnlineGames','backdoor.PcClient','backdoor.Agent']

fig, ax = plt.subplots()
im = ax.imshow(acc_matrix)

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('realative acc', rotation=-90, va="bottom")

# We want to show all ticks...
ax.set_xticks(np.arange(len(acc_matrix)))
ax.set_yticks(np.arange(len(acc_matrix)))
# ... and label them with the respective list entries
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)



# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, acc_matrix[i][j],
                       ha="center", va="center", color="w")

ax.set_title('Cross Validation between classes')
fig.tight_layout()
plt.show()