# -*- coding: utf-8 -*
import serial
import time
import struct

#ser = serial.Serial("/dev/ttyAMA1",1000000,parity='E',timeout=0.000005)
ser = serial.Serial("/dev/ttyAMA1",115200*16,parity='E',timeout=0.00001)

if not ser.isOpen():
    print("open failed")
else:
    print("open success: ")
    print(ser)
    
array_f=[1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.1,11.1,]*6
data_array=struct.pack('<66f',*array_f)	

#ser.write(b'Hello, world!\n')
ser.write(data_array)
try:
    while True:
        count = ser.inWaiting()
        if count >= 1:
            #data1=ser.readline()
            #data = ser.readline()
            print("count",count,time.time())
            data1 = ser.read(24)
            if count>=24:
                data_unpack=struct.unpack('<6f',data1)
                print("unpack all",data_unpack,"time:",time.time())
            #for line in data1:
            #line_data=data1.decode('utf-8')
            #data = ser.readline().decode('utf-8').rstrip()
            #print('Received: ' + data)
            #print(data1.decode('utf-8'))
            #ser.flushInput()
            #recv = ser.read(count)
            #data = ser.readline()
            #recv1=recv.decode('utf-8')
            #print("recv: " + recv1)
            print(time.time())
            #line_data='12345646487895265556663745656655455666222'
            #ser.write(b'Hello, world!') # 向串口发送数据
            #ser.write(line_data.encode('utf-8'))
            ser.write(data_array)
        #time.sleep(0.000) 
except KeyboardInterrupt:
    if ser != None:
        ser.close()
