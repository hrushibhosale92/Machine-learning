# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:11:25 2021

@author: hrushikesh.bhosale
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


data = pd.read_csv("Iris.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

km = KMeans(n_clusters=2)
km.fit(X)

#print(km.labels_)

label = km.labels_


dff = pd.DataFrame()
dff['actual_cluster'] = y
dff['cluster_label'] = label

sorted_label = dff.sort_values(by=['cluster_label'])








######################################################################3




import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


data = pd.read_csv("glass_csv.csv")

X = data.iloc[:,:-1]
y = data.iloc[:,-1]


km = KMeans(n_clusters=6)
km.fit(X)

print(km.labels_)

label = km.labels_

dff = pd.DataFrame()
dff['actual_cluster'] = y
dff['cluster_label'] = label

sorted_label = dff.sort_values(by=['cluster_label'])










############################################################################33
## elbow method 

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris['feature_names'])
#print(X)
#data = X[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']]

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k).fit(X)
    #data["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ 
    # Inertia: Sum of distances of samples to their closest cluster center
    
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
