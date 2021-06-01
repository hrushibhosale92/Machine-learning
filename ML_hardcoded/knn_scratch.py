# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:32:51 2021

@author: hrushikesh.bhosale
"""


"Train Test"
#importing labraries
import numpy as np
import pandas as pd

#calculate distance funcrion 
def distance(x,y):
	return np.sqrt(sum((x-y)**2))

#importing datasets
data = pd.read_csv('iris.csv')

#selecting 100 examples to convert data to binary classification
X = data.iloc[50:,:-1] 
y = data.iloc[50:,-1]

K = 3 #nearest nighbour


#deviding data to train and test
shuf = np.random.choice(2,len(X),p=[0.8,0.2])

X_test = X.iloc[np.where(shuf == 1)[0],:]
X_train = X.iloc[np.where(shuf == 0)[0],:]

y_test = y.iloc[np.where(shuf == 1)[0]]
y_train = y.iloc[np.where(shuf != 1)[0]]


y_pred = []
for j in np.arange(len(X_test)): #for total examples in test data
    
    dist = [] #dist of a test example to all examples in training set
    for i in range(len(X_train)): 
        d = distance(X_test.iloc[j],X_train.iloc[i])
        dist.append(d)
     
    #finding k nearest examples 
    min_index = np.argsort(dist)[0:K]
    classes = y_train.iloc[min_index]
    #print(min_index)
    
    #assigning the majority class label to the test example
    if list(classes).count('Iris-versicolor') >= np.ceil(K/2):
        y_pred.append('Iris-versicolor')
    else:
        y_pred.append('Iris-virginica')
        
    print(X_test.iloc[j].name,y[X_test.iloc[j].name],y_pred[-1])
    
acc = sum(y_pred == y_test) / len(y_test)
print(acc)
    












"5 fold cross validation"

#importing labraries
import numpy as np
import pandas as pd
from statistics import mode

#calculate distance funcrion 
def distance(x,y):
	return np.sqrt(sum((x-y)**2))

#importing datasets
data = pd.read_csv('iris.csv')

#selecting 100 examples to convert data to binary classification
X = data.iloc[:100,:-1] 
y = data.iloc[:100,-1]

K = 3 #nearest nighbour

#dividing data to 5 fold cv
fold = 5
#np.random.seed(0)
shuf = np.random.choice(fold,len(X))

fold_accuracy = []
for i in range(fold):
    print(i)
    X_test = X.iloc[np.where(shuf == i)[0],:]
    X_train = X.iloc[np.where(shuf != i)[0],:]
    
    y_test = y.iloc[np.where(shuf == i)[0]]
    y_train = y.iloc[np.where(shuf != i)[0]]
    
    y_pred = []
    for j in range(len(X_test)):#for total examples in test data
    
        dist = []  #distince of a test example to all examples in training set   
        for l in range(len(X_train)):
            d = distance(X_test.iloc[j],X_train.iloc[l])
            dist.append(d)
            
        #finding k nearest neighbours 
        k_min_index = np.argsort(dist)[0:K]
        classes = y_train.iloc[k_min_index]
        #print(classes)
        y_pred.append(mode(classes))
            
    acc = sum(y_pred == y_test) / len(y_test)
    
    print(acc)
    fold_accuracy.append(acc)
    
CV_accuracy = np.mean(fold_accuracy)
print("average accuracy of 5 folds : ",CV_accuracy)