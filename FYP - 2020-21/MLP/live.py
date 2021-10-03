# -*- coding: utf-8 -*-
"""
@author: Taroon
"""

#importing required libraries
import serial
import time
import pickle


#allocate timer
start = time.time() #init
PERIOD_OF_TIME = 90 #set timer to 15 seconds


#Allocation of serial port parameters
serialPort = serial.Serial(port="COM3", baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)

#numpy array
from numpy import array

arr = []

#fixed interval
fixed_interval = 10

serialString = ""  # Used to hold data coming over UART
while 1:
    
     #Check for data waiting in the serial buffer
    if serialPort.in_waiting > 0:

        #Read data out of the buffer until a carraige return / new line is found
        serialString = serialPort.readline()


          #Print contents from serial port data
        try:
          print(serialString.decode("Ascii"))
         
          serialString = serialString.decode("Ascii").strip('\r\n')
          
           
          
          arr.append(serialString)
          
          
        except:
             pass
    if time.time() > start + PERIOD_OF_TIME :
         break
     
#load the trained mlp model
loadModel = pickle.load(open('mlp_v.sav', 'rb'))

gesturedata = array(arr)

def check_integer(potential_float):
    try:
        int(potential_float)
        
#Try to convert argument into a float

        return True
    except ValueError:
        return False
    


arr1 = []

for x in gesturedata:
    
    
    if(check_integer(x)):
        
      print("Integer")
      print(x)
      arr1.append(int(x))
      
    else: 
      print(x)  
      arr1.append((x))
      

print("printing gesturedata......")    
print(gesturedata)

print("printing arr1")
print(arr1)


gesturedata = gesturedata.reshape(1,180) 

arr1 = array(arr1)

arr1 = arr1.reshape(1,180)

prediction = loadModel.predict(arr1)

print("model predicted ")
print(prediction)
























