# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:12:25 2021

@author: hrushikesh.bhosale
"""



import pandas as pd
import numpy as np 

data = pd.read_csv('iris.csv')

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split,cross_val_score

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

import matplotlib.pyplot as plt
tree.plot_tree(clf)
plt.show()
plt.savefig("tree.png",dpi=80)

acc_cv = cross_val_score(tree.DecisionTreeClassifier(),X,y,cv=5)
print(np.mean(acc_cv))

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(50,50))
_ = tree.plot_tree(clf, 
                   filled=True)
plt.savefig("tree.png")



import matplotlib.pyplot as plt
fig = plt.figure(figsize=(50,50))
tree.plot_tree(clf,
               feature_names= X.columns,
               class_names = np.unique(y))
plt.savefig("tree.png")

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   filled=True)
