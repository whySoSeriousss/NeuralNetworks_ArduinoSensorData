# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:05:36 2021

@author: Taroon
"""

#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_train = pd.read_csv('sensorCleaned1.csv')

training_set = dataset_train.iloc[:, 0:6].values

# Feature Scaling  
from sklearn.preprocessing import MinMaxScaler  
sc = MinMaxScaler(feature_range = (0, 1))  
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output  
X_train = []  
y_train = []  
for i in range(80, 107):  
    X_train.append(training_set_scaled[i-60:i, 0])  
    y_train.append(training_set_scaled[i, 0])  
X_train, y_train = np.array(X_train), np.array(y_train) 


# Reshaping  
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 


# Importing the Keras libraries and packages  
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout 

# Initialising the RNN  
regressor = Sequential() 

# Adding the first LSTM layer and some Dropout regularization  
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) 

regressor.add(Dropout(0.2)) 

# Adding a second LSTM layer and some Dropout regularization  
regressor.add(LSTM(units = 50, return_sequences = True))  
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularization  
regressor.add(LSTM(units = 50, return_sequences = True))  
regressor.add(Dropout(0.2))  


# Adding a fourth LSTM layer and some Dropout regularization  
regressor.add(LSTM(units = 50))  
regressor.add(Dropout(0.2)) 


# Adding the output layer  
regressor.add(Dense(units = 1))  

# Compiling the RNN  
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')  


# Fitting the RNN to the Training set  
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) 

dataset_test = pd.read_csv('sensorTest1.csv')  
real_stock_price = dataset_test.iloc[:, 0:6].values  

# Getting the predicted stock price of 2017  
dataset_total = pd.concat((dataset_train['name'], dataset_test['name']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 80:].values 

inputs = inputs.reshape(-1,1)  

inputs = sc.transform(inputs)

X_test = []  
for i in range(22, 80):  
    X_test.append(inputs[i-22 :i, 0])  
X_test = np.array(X_test)  
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 



predicted_data = regressor.predict(X_test)

predicted_data = sc.inverse_transform(predicted_data)









