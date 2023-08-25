# -*- coding: utf-8 -*
import serial
import time
import struct

#ser = serial.Serial("/dev/ttyAMA1",1000000,parity='E',timeout=0.000005)
ser = serial.Serial("/dev/ttyAMA4",115200*16,parity='E',timeout=0.001)

if not ser.isOpen():
    print("open failed")
else:
    print("open success: ")
    print(ser)
    
array_f=[1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.1,11.1,]*6
array_f=[1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,]*3
data_array=struct.pack('<27f',*array_f)	

def crc16_cal(datalist):
    test_crc=0xFFFF                 #预置1个16位的寄存器为十六进制FFFF（即全为1），称此寄存器为CRC寄存器；
    poly=0xa001
    # poly=0x8005
    numl=len(datalist)
    for num in range(numl):
        data=datalist[num]
        test_crc=(data&0xFF)^test_crc   #把第一个8位二进制数据（既通讯信息帧的第一个字节）与16位的CRC寄存器的低8位相异或，把结果放于CRC寄存器，高八位数据不变；
        
        #右移动
        for bit in range(8):
            if(test_crc&0x1)!=0:
                test_crc>>=1
                test_crc^=poly
            else:
                test_crc>>=1
    #print(hex(test_crc))
    return test_crc



#ser.write(b'Hello, world!\n')
ser.write(data_array)
try:
    while True:
        count = ser.inWaiting()
        
        #print("count",count,time.time())
        if count >= 75:
            #data1=ser.readline()
            #data = ser.readline()
            count1 = ser.inWaiting()
            if count1>count:
                count=count1
            data1 = ser.read(count)
            print("count",len(data1),time.time())
            start_time1=time.time()
            
            
            if len(data1)==75:
                data_array_read=data1[:72]
                crc_read=data1[72:74]
                crc_int=int.from_bytes(crc_read, byteorder='big', signed=True)
                crc_cal=crc16_cal(data_array_read)
            
                if crc_int==crc_cal:
                    data_right=1
                    #ser.write(b'OK')
                    print("ok")
                else:
                    data_right=0
                #ser.write(b'RE')
                    print("re")
                    #data_unpack=struct.unpack('<18f',data_array_read)
                    #print("unpack all",data_unpack,"time:",time.time())
            else:
                data_right=0
                print("data length not enough")
                
                
               
            if data_right==1:
                data_unpack=struct.unpack('<18f',data_array_read)
                print("unpack all",data_unpack,"time:",time.time())
            
            print("last time",(time.time()-start_time1)*1000)
            ser.write(data_array)
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
            #print(time.time())
            #line_data='12345646487895265556663745656655455666222'
            #ser.write(b'Hello, world!') # 向串口发送数据
            #ser.write(line_data.encode('utf-8'))
            
        #time.sleep(0.000) 
except KeyboardInterrupt:
    if ser != None:
        ser.close()
