# -*- coding: utf-8 -*-
"""
@author: Taroon
"""

#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#loading the dataset
data = pd.read_csv('sensorCleaned.csv')

print(data.head(100))


#LabelEncoder
from sklearn import preprocessing

#initiating labelEncoder
labelEncoder = preprocessing.LabelEncoder()

#every field should be floats for hidden layer processing
data['name'] = labelEncoder.fit_transform(data['name'])


#Splitting the dataset into input set(X) and output set(y)
X = data[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']]
y = data['name']


#Splitting the training and testing set - ratio(train_test_split function)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


#Building the classifier - MLPClassifier
from sklearn.neural_network import MLPClassifier

#Instantiating an object to represent the Model - defining parameters for the Classifier
mlp = MLPClassifier(hidden_layer_sizes = (6,5),
                    random_state = 42,
                    max_iter = 1000,
                    verbose = True,
                    learning_rate_init = 0.01,
                    activation = "relu",
                    solver = 'adam')

#fitting the data onto the model - training sets
mlp.fit(X_train, y_train)


#predictions on test dataset
prediction = mlp.predict(X_test)
print(prediction)
print("_____________________________________")

#confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, prediction)
print(confusion_matrix(y_test, prediction))
print("_____________________________________")

#Getting the accuracy score
from sklearn.metrics import accuracy_score

accuracy_score(y_test, prediction)
print("Model Accuracy: " + str(accuracy_score(y_test, prediction)))
print("_____________________________________")


#Classification report
from sklearn.metrics import classification_report

classification_report(y_test, prediction)
print("_____________________________________")
print(classification_report(y_test, prediction))


# #data vizualisation
# plt.plot(X_train, marker = 'o')
# plt.title("X_train")
# plt.show()

# plt.plot(y_train, marker = 'o')
# plt.title("y_train")
# plt.show()

# plt.plot(X_test, marker = 'o')
# plt.title("X_test")
# plt.show()

# plt.plot(y_test, marker = 'o')
# plt.title("y_test")
# plt.show()

# plt.scatter(y_test, prediction)
# plt.title("Sensor Data Prediction")
# plt.show()

# print("_____________________________________") 
# print("0: beat_right, 1: hit_right, 2: pause") 

# filename = 'mlp.sav'
# pickle.dump(mlp, open(filename, 'wb'))


# import serial

# serialPort = serial.Serial(
#     port="COM3", baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
# )

# sensor = []

# serialString = ""
# while 1:
#     #Only when sensor readings are being detected
#     if serialPort.in_waiting > 0:

#         #Keep reading until promted to stop
#         serialString = serialPort.readline()

#         # Print the contents of the serial data in sensor() list
#         try:
#             print(serialString.decode("Ascii"))
#             sensor.append(serialString.decode("Ascii"))
#         except:
#             pass




# prediction_test =  mlp.predict()
# print(prediction_test)




data_test = pd.read_csv('sensorTest.csv')

X1 = data_test[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']]
y1 = data_test['name']



mlp.fit(X1, y1)

test_predict = mlp.predict(X1) 
print(test_predict) 

accuracy_score(y1, test_predict)
print("New testing Model Accuracy: " + str(accuracy_score(y1, test_predict))) 
























