# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 23:22:26 2019

@author: Shyam Parmar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error

# Load data 
data = pd.read_csv('student.csv')

# Get scores to array
math = data['Math'].values
read = data['Reading'].values
write = data['Writing'].values

# Ploting the scores as scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(math, read, write, color='#ef1234')
plt.show()

X = np.array([math, read]).T
Y = np.array(write)

# Splitting training and testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

# Model Intialization
reg = LinearRegression()

# Data Fitting
reg = reg.fit(X, Y)

# Y Prediction
Y_pred = reg.predict(X)

reg.fit(X_train, y_train)
confidence = reg.score(X_test, y_test)
print("Confidence : ", confidence)  # Confidence is accuracy of the model

# Model Evaluation
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)

print("RMSE : ", rmse)
print("R2 Score : ", r2)
