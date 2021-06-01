# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:35:21 2021

@author: hrushikesh.bhosale
"""



from sklearn.linear_model import LinearRegression,SGDRegressor
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
X,y = load_boston(return_X_y=True)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

reg = RandomForestRegressor()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

mean_squared_error(y_test,y_pred)

