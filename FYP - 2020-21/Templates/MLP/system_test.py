#import serial library to be able to work with serial ports
import serial

serialPort = serial.Serial(
    port="COM3", baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
)

sensor = []

serialString = ""
while 1:
    #Only when sensor readings are being detected
    if serialPort.in_waiting > 0:

        #Keep reading until promted to stop
        serialString = serialPort.readline()

        # Print the contents of the serial data in sensor() list
        try:
            print(serialString.decode("Ascii"))
            sensor.append(serialString.decode("Ascii"))
        except:
            pass







