import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('winequality-red.csv')

x = df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
y = df['quality']
z = df[['fixed acidity','alcohol']]
 
#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x, y)

#Predict Output 
predicted= model.predict(z)
print(predicted)
print("YEs")






