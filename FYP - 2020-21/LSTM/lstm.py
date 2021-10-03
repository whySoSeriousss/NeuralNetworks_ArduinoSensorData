# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 22:11:50 2021
@author: Taroon
"""

from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
plt.style.use('ggplot')


dataset = pd.read_csv('arduino.csv')


print(dataset)  
print(dataset.head())
print(dataset.shape)

X = dataset.loc[:, 'gest_name']
y = dataset.loc['gest_name', :]






