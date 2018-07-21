import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('winequality-red.csv')

#df1 = df[['alcohol','quality']]

#df1.plot(kind='bar')

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = df)

plt.show()


