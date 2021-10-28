# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 13:52:01 2021

@author: DELL
"""


import serial
import time
import pickle
import pandas as pd
import numpy as np
from numpy import array

start = time.time()

PERIOD_OF_TIME = 12 # set timer to 12 sec

serialPort = serial.Serial(
    port="COM7", baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
)

array2 = []

fixed_interval = 10


serialString = ""  # Used to hold data coming over UART
while 1:

    
    # Wait until there is data waiting in the serial buffer
    if serialPort.in_waiting > 0:

        # Read data out of the buffer until a carraige return / new line is found
        serialString = serialPort.readline()
        
        #for I in range(10):

          # Print the contents of the serial data
        try:
          print(serialString.decode("Ascii"))
          #array2.append(serialString.decode("Ascii").strip('\r\n'))
          serialString = serialString.decode("Ascii").strip('\r\n')
          
           
          
          array2.append(serialString)
          
          
        except:
             pass
    if time.time() > start + PERIOD_OF_TIME :
         break
     
        

loaded_model = pickle.load(open('reconfigure2_finalMLPmodel.sav', 'rb'))



gesturedata = array(array2)



def check_integer(potential_float):
    try:
        int(potential_float)
        
#Try to convert argument into a float

        return True
    except ValueError:
        return False


arr = []

for x in gesturedata:
    
    #arr = np
    
    
    if(check_integer(x)):
        
      #x = x.astype(np.int())
      print("Integer")
      print(x)
      #x = x.astype(np.int())
      arr.append(int(x))
      
    else:
        
      print("float")
      print(x)
      #x = x.astype(np.float())
        
      arr.append(float(x))
      
      

print("printing gesturedata......")    
#gesturedata = gesturedata.astype(np.float())
print(gesturedata)

print("printing arr......")
print(arr)

#feature = len(array2)

gesturedata = gesturedata.reshape(1,220)

arr = array(arr)

arr = arr.reshape(1,220)

prediction = loaded_model.predict(arr)


print("model predicted ")
print(prediction)
        