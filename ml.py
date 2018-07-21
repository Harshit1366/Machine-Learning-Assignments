import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import pandas as pd

x = []
y = []
z = []
a = []

data= np.genfromtxt('iris.csv',delimiter=',')
for row in data:
        x.append(float(row[0]))
        y.append(float(row[1]))
        z.append(float(row[2]))
        a.append(float(row[3]))

#print("x: ",x)

#df=pd.DataFrame(x,y)
df=pd.read_csv('iris.csv')
#print(df)

data = pd.read_csv('iris.csv', sep=',',header=None, index_col =0)

data.plot(kind='bar')

#data = np.vstack([x, y, z, a]).T
#bins = np.linspace(0, 8, 10)

plt.hist(x, bins=20, histtype='stepfilled', normed=True, alpha=0.3)
plt.hist(y, bins=20, histtype='stepfilled', normed=True, color='b', alpha=0.3)
plt.hist(z, bins=20, histtype='stepfilled', normed=True, color='r', alpha=0.3)
plt.hist(a, bins=20, histtype='stepfilled', normed=True, color='k', alpha=0.3)
#plt.legend(loc='upper right')
#df.plot(kind='bar')
plt.show()

plt.scatter(z, y, color='DarkBlue', label='Group 1');
plt.scatter(x, a, color='DarkGreen', label='Group 2');
plt.show()
