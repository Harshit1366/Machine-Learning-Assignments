import pandas as pd
# Load libraries

import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import preprocessing

lab_enc = preprocessing.LabelEncoder()

data = pd.read_csv('2017.csv')
#df=data[['fixed acidity','pH','density','alcohol']]
# Split-out validation dataset
array = data.values
#X=df.values
X = array[:,3:7]
#X = np.c_[data[:, 0], data[:, 7], data[:, 8],  data[:, 10]]
#X = np.nan_to_num(X)
Y = array[:,2]
#X=X.astype('float')
Y=Y.astype('int')
#print(X)
#print(Y)
#lab_enc.fit_transform(X)
#lab_enc.fit_transform(Y)
#print(X)
#print(Y)
validation_size = 0.20
seed = 7
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Spot Check Algorithms
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
print("Name\tMean\tStd. Diviation")
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print("ACCURACY SCORE OF KNN: ",accuracy_score(Y_validation, predictions))
print("CONFUSION MATRIX:\n",confusion_matrix(Y_validation, predictions))
print("CLASSIFICATION REPORT:\n",classification_report(Y_validation, predictions))


gbn = GaussianNB()
gbn.fit(X_train, Y_train)
predictions = gbn.predict(X_validation)
print("ACCURACY SCORE OF GNB: ",accuracy_score(Y_validation, predictions))
print("CONFUSION MATRIX:\n",confusion_matrix(Y_validation, predictions))
print("CLASSIFICATION REPORT:\n",classification_report(Y_validation, predictions))


svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print("ACCURACY SCORE OF SVM: ",accuracy_score(Y_validation, predictions))
print("CONFUSION MATRIX:\n",confusion_matrix(Y_validation, predictions))
print("CLASSIFICATION REPORT:\n",classification_report(Y_validation, predictions))





