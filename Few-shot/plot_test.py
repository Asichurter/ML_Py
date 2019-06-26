import matplotlib.pyplot as plt

a = [2,3]
var = [0.5, 1]
width = 0.4
x = [1,2]

plt.xlim(0,3)
plt.bar(x, a, width, yerr=var, color='orange')
plt.show()