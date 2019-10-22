import matplotlib.pyplot as plt
import numpy as np
import random as rd

# a = 'a'
# b = True
# c = [[1,'a'],[3,'b'],[4,'c'],[2,'d']]
# c.sort(key=lambda x: x[0], reverse=True)
# c.sort(cmp=lambda x,y: x[0]-y[0])

def entro(xx):
    return -1*(xx*np.log2(xx)+(-1*xx+1)*np.log2(-1*xx+1))

x = np.linspace(0,1,50)
y = entro(x)
plt.plot(x,y)
# plt.show()

for i in range(10):
    print(rd.uniform(0,10))

# a = {1:1,2:2}
# a.setdefault(None)
# aa = a.get(3)

