# -*- coding: utf-8 -*-
"""
@author: Taroon
"""

import pandas as pd
import pickle
from numpy import array
import numpy as np


# Assign column names
names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'name']

gesturedata = pd.read_csv("sensorCleaned.csv", names=names)


# print first 5 column
print(gesturedata.head())

# Assign data from first 6 columns to X variable
x = gesturedata.iloc[:, 0:6]
#print(X)
gesturedata_X = array(x)
print (gesturedata_X)

print("reshaping...")
X =  gesturedata_X.reshape(3, 180)
print(X)
print("reshaping done")

# Assign data from first twelve columns to y variable

Y = gesturedata.select_dtypes(include=['object'])

#Y = np.array(Y)
#Y = np.unique(Y)
print (Y)

#'''
# converting categorical values to numerical values

from sklearn import preprocessing

encode = preprocessing.LabelEncoder()

Y = Y.apply(encode.fit_transform)

# check numeric value assign to each class
# 1 = clockwise/ 0 = anticlockwise/4=zoomin/2=moveleft/3=moveright
#print(Y.Class.unique())

Y = np.array(Y)
Y = np.unique(Y)
print (Y)

'''
# splits dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, Y, train_size=0.80, test_size = 0.20)

'''

# scaling feature to perform evaluation uniformly

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#scaler.fit(X_train)
scaler.fit(X)


#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

X = scaler.transform(X)



#import MLPClassifierto train the neural network to make predictions
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(6,5), max_iter=1000, random_state = 42,

                    verbose = True,
                    learning_rate_init = 0.1,
                    solver = 'adam',
                    alpha = 0.05,
                    activation = "relu")


#mlp.fit(X_train, y_train.values.ravel())
mlp.fit(X,Y)

filename = 'mlp_v.sav'
pickle.dump(mlp, open(filename, 'wb'))