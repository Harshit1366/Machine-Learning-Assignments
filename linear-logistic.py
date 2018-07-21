import pandas as pd
import numpy as np
from sklearn import preprocessing
#import matplotlib.pyplot as plt 
#plt.rc("font", size=14)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
#import seaborn as sns
#sns.set(style="white")
#sns.set(style="whitegrid", color_codes=True)

data= pd.read_csv('winequality-red.csv')

X = data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

c = LinearRegression()
c.fit(X_train, y_train)

coefficientdf = pd.DataFrame(c.coef_, X.columns, columns=['Coeff'])
print("c.intercept: \n",c.intercept_)
print("c.coef: \n",c.coef_)
print("c.coefficientdf: \n",coefficientdf)
predictions = c.predict(X_test)
print("predictions: \n",predictions)

print('Accuracy of linear regression classifier on test set: {:.2f}'.format(c.score(X_test, y_test)))
