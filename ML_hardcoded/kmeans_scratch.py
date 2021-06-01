# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 12:41:54 2021

@author: hrushikesh.bhosale
"""

import numpy as np 
import pandas as pd

data = pd.read_csv("Iris.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

def calc_distance(X1, X2):
    return np.sqrt(sum((X1 - X2)**2))

data_mean = np.mean(X)
print(data_mean)

cg1 = np.random.uniform(data_mean*1.1,data_mean*0.9)
cg2 = np.random.uniform(data_mean*1.1,data_mean*0.9)
cg3 = np.random.uniform(data_mean*1.1,data_mean*0.9)
print(cg1,cg2,cg3)

obj_old = 0
for iteration in range(50):

    # distance of all examples from each cluster center
    dist_cg1 = []
    for i in range(len(X)):
        dist_cg1.append(calc_distance(cg1,X.iloc[i,:]))

    dist_cg2 = [] 
    for i in range(len(X)):
        dist_cg2.append(calc_distance(cg2,X.iloc[i,:]))

    dist_cg3 = []
    for i in range(len(X)):
        dist_cg3.append(calc_distance(cg3,X.iloc[i,:]))

    dist = pd.DataFrame()
    dist['cg1'] = dist_cg1
    dist['cg2'] = dist_cg2
    dist['cg3'] = dist_cg3

    # finding the min distance of cg from each example
    labels = []
    for k in range(len(X)):
        labels.append(np.argmin(dist.iloc[k,:]))
    label = np.array(labels)
    
    #Each example is assigned to the cluster with whose cg is closest to example
    cluster_1 = X.iloc[np.where(label == 0)[0],:]
    cluster_2 = X.iloc[np.where(label == 1)[0],:]
    cluster_3 = X.iloc[np.where(label == 2)[0],:]

    #finding the objective function 
    d1 = []
    for i in range(len(cluster_1)):
        d1.append(calc_distance(cg1,cluster_1.iloc[i,:]))
    d2 = [] 
    for i in range(len(cluster_2)):
        d2.append(calc_distance(cg2,cluster_2.iloc[i,:]))
    d3 = []
    for i in range(len(cluster_3)):
        d3.append(calc_distance(cg3,cluster_3.iloc[i,:]))

    obj_new = sum(d1)+sum(d2)+sum(d3)

    # calculating new cg for next iteration 
    cg1 = np.array(np.mean(cluster_1))
    cg2 = np.array(np.mean(cluster_2))
    cg3 = np.array(np.mean(cluster_3))
    print(label)
    print(cg1,cg2,cg3)

    #stopping criteria 
    if (abs(obj_old - obj_new ) < 0.00001):
        break

    print(iteration)
    print(obj_old,obj_new)

    obj_old = obj_new 
