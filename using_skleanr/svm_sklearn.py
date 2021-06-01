# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:36:53 2021

@author: hrushikesh.bhosale
"""


import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('spam_classification.csv')

X = data.iloc[:,:-2]
y = data.iloc[:,-2]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

clf = SVC(kernel='rbf',probability=True)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test,y_pred))

y_prob = clf.predict_proba(X_test)
    

from sklearn.metrics import accuracy_score,roc_auc_score,average_precision_score,precision_recall_curve,auc

from sklearn.metrics import precision_score,recall_score
print(average_precision_score(y_test,y_prob))

print(roc_auc_score(y_test,y_prob))


precision, recall, thresholds = precision_recall_curve(y_test,y_prob[:,1],pos_label=1)
auPR = auc(recall,precision)#,reorder=True)





from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score,make_scorer,precision_recall_curve,auc,roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit,cross_val_score
from IPython.display import display, clear_output
from sklearn.preprocessing import StandardScaler

def custom_func(yTest, y_prob):
    precision, recall, thresholds = precision_recall_curve(yTest,y_prob,pos_label=1)
    error = auc(recall,precision)
    return error
custom_scorer = make_scorer(custom_func, greater_is_better=True,needs_proba=True)


estimator = SVC(probability=True)
param_grid = {   
    'kernel': ['rbf'],
    'C':[0.01,0.1,1,10,100],
    'class_weight': [None,'balanced'],
    'gamma': [0.001,0.01,0.1,1,10,100,'scale'],
}

import warnings
warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars')

grid = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    verbose=3,cv = 5,
    scoring=custom_scorer,
    n_jobs=11)

grid.fit(X_train,y_train)




