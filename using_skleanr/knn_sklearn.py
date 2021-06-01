# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 18:38:39 2021

@author: hrushikesh.bhosale
"""


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,matthews_corrcoef
from sklearn.metrics import classification_report

data = pd.read_csv('spam_classification.csv')

X = data.iloc[:,:-2]
y = data.iloc[:,-2]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).reshape(-1,)





#########################################################3





import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,matthews_corrcoef,make_scorer
from sklearn.metrics import classification_report

data = pd.read_csv('spam_classification.csv')

X = data.iloc[:,:-2]
y = data.iloc[:,-2]

clf = KNeighborsClassifier(n_neighbors=3)
acc = make_scorer(accuracy_score)
acc_cv = cross_val_score(clf, X, y, cv=5,scoring=acc)
mcc = make_scorer(matthews_corrcoef)
mcc_cv = cross_val_score(clf, X, y, cv=5,scoring=mcc)


print(np.mean(acc_cv))
print(np.mean(mcc_cv))










import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,matthews_corrcoef
from sklearn.metrics import classification_report

#importing datasets
data = pd.read_csv('C:/Users/hrushikesh.bhosale/Desktop/flame/neural_network/diabetes.csv')

#selecting 100 examples to convert data to binary classification
X = data.iloc[:,:-1] 
y = data.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)



# for fining different 
acc_list = []
for k in np.arange(3,28,2):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train,y_train)
    
    y_pred = clf.predict(X_test)
    acc= accuracy_score(y_test,y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).reshape(-1,)
    positive_accuracy = tp / (tp+fn)
    negative_accuracy = tn / (tn+fp)
    MCC = matthews_corrcoef(y_test, y_pred)
    
    acc_list.append([k,acc,positive_accuracy,negative_accuracy,MCC])
    

df = pd.DataFrame(acc_list, columns=['k', 'Acc', 'P_acc','N_acc','MCC'])



 



