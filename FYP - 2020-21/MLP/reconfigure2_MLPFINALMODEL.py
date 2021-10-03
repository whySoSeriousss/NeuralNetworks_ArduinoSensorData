

import pandas as pd
import pickle
from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold, cross_val_score


# Assign column names
names = ['Thumbval', 'Indexval', 'Middleval', 'Ringval', 'Littleval', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Class']

gesturedata = pd.read_csv("finaldataset.csv", names=names)

# print first 5 column
print(gesturedata.head())

# converting columns 0-5 to int
gesturedata[["Thumbval", "Indexval","Middleval","Ringval","Littleval"]] = gesturedata[["Thumbval", "Indexval","Middleval","Ringval","Littleval"]].apply(pd.to_numeric)
print(gesturedata.dtypes)


# Assign data from first 11 columns to X variable
x = gesturedata.iloc[:, 0:11]
#print(X)
gesturedata_X = array(x)
print (gesturedata_X)

print("reshaping...")
X =  gesturedata_X.reshape(5, 220)
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
mlp = MLPClassifier(hidden_layer_sizes=(100,60), max_iter=200, random_state = 5,

                    verbose = True,
                    learning_rate_init = 0.1,
                    solver = 'adam',
                    alpha = 0.05,
                    activation = "tanh")


#mlp.fit(X_train, y_train.values.ravel())
mlp.fit(X,Y)



'''
# sklearn optimization tool
# search the best parameter for the model to train with ...
from sklearn.neural_network import  MLPClassifier

mlp = MLPClassifier(max_iter = 100)

parameter_space = {
    
    'hidden_layer_sizes': [(5,20,5), (10,20,10),  (5,5,5), (10,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    
    
    
    }


from sklearn.model_selection import GridSearchCV 

clf = GridSearchCV(mlp, parameter_space, n_jobs = -1, cv = KFold(n_splits=3)) 

clf.fit(X, Y)

print('best parameters found:\n', clf.best_params_)
'''




'''
# Make prediction on our test data

prediction = mlp.predict(X_test)
print(prediction)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))


#Getting the accuracy score
from sklearn.metrics import accuracy_score

accuracy_score(y_test, prediction)
print("Model Accuracy: " + str(accuracy_score(y_test, prediction)))

'''
filename = 'reconfigure2_finalMLPmodel.sav'
pickle.dump(mlp, open(filename, 'wb'))

#pickle.dump(model, open(filename, 'wb'))

