#coding=GBK"
import json
from sys import path
path.append("../../")
import math
import numpy as np
import random
#import Servos
import sys
import os

import socket
import setproctitle
import numpy as np
from pathlib import Path
from threading import Thread, current_thread

#from stable_baselines.common.env_checker import check_env

import math

import json
from multiprocessing import Process
from multiprocessing import Process, Queue
#import queue

from threading import Timer

import time
import datetime
import platform
import struct
import lib.device_model as deviceModel
from lib.data_processor.roles.jy901s_dataProcessor import JY901SDataProcessor
from lib.protocol_resolver.roles.wit_protocol_resolver import WitProtocolResolver

import socket
import time
import sys
import datetime
import numpy as np
import random 
import struct
import numpy as np

from Servos import *

from set_imu import *
from utils import *
from reflex_related import *

import pybullet as p
import pybullet_data as pd
import math
import numpy as np
import random
path.append("./hexapod_real-main/Hexapod_Real/envs/")
from reflex_related import *
from pybullet_functions import *
from distill_model.action_net import *
import torch
import serial
import time
import struct











def read_servos_only(servos,cpg_index,step,imu,coef_real,phase_now,cpg_now):
    
    # 输出应该是 仿真环境下的机器人状态
    # imu 输入的应该是角度 单位为度数
    agent_num=6
    
    start_t0=time.time()
    position_Read=servos.read_all_positions() # np array 18
    
    end_t=time.time()
    print("read time: ",(end_t-start_t0)*1000)
    start_t1=time.time()
    theta_sim=real_angles_to_sim(position_Read)
   
    (roll ,pitch,yaw)=imu[0:3]/180*math.pi # 绕着 xy z轴转动
    length_12=0.126
    length_13=0.266
    dz1=-length_12*math.tan(roll)
    dz2=-length_13*math.tan(roll)
    foot_z=np.zeros(6)
    #phase=phase_r[cpg_index,:]
    phase=phase_now
    observation_temp=[]
    print("cal time: ",(time.time()-start_t1)*1000)
    
    start_t111=time.time()
    for agent_index in range(agent_num):
        start_t0=time.time()
        
        robot_joint_positions_agenti=theta_sim[agent_index]
        if agent_index>=3:
            end_pos,end_ori=get_forward_pos(-robot_joint_positions_agenti)
        else:
            end_pos,end_ori=get_forward_pos(robot_joint_positions_agenti)
            
        foot_z[agent_index]=end_pos[2]
        if agent_index%3==1:
            foot_z[agent_index]=end_pos[2]+dz1            
        if agent_index%3==2:
            foot_z[agent_index]=end_pos[2]+dz2
            
        relative_foot_z=np.array([foot_z[agent_index]-foot_z[0]])
        print("forward pos time: ",(time.time()-start_t0)*1000)
            # 相位信息
            
        #phase_continus=cpg_r[agent_index,2:4,cpg_index]
        phase_continus=cpg_now[agent_index,2:4]
        start_t0=time.time()
        observation_agenti=np.concatenate((robot_joint_positions_agenti,imu,relative_foot_z,phase_continus,np.array([phase[agent_index]]),np.array([coef_real[agent_index]]),),)
        print("concatenate time: ",(time.time()-start_t0)*1000)
            
        start_t0=time.time()
            #observation_agenti = np.append(robot_joint_positions_agenti, imu)
            #observation_agenti = np.append(observation_agenti,relative_foot_z)
            #observation_agenti = np.append(observation_agenti,phase_continus)
            #observation_agenti = np.append(observation_agenti,phase[agent_index])
            #observation_agenti = np.append(observation_agenti,coef_real[agent_index])
        observation_temp.append(observation_agenti)
        print("append time: ",(time.time()-start_t0)*1000)
    
    observation = np.array(np.vstack(observation_temp)).flatten()
    print("loop time: ",(time.time()-start_t111)*1000)
    return observation,position_Read



def onUpdate(deviceModel):
    """
    数据更新事件  Data update event
    :param deviceModel: 设备模型    Device model
    :return:
    """
    global IMU_data
    '''
    print("芯片时间:" + str(deviceModel.getDeviceData("Chiptime"))
         , " 温度:" + str(deviceModel.getDeviceData("temperature"))
         , " 加速度：" + str(deviceModel.getDeviceData("accX")) +","+  str(deviceModel.getDeviceData("accY")) +","+ str(deviceModel.getDeviceData("accZ"))
         ,  " 角速度:" + str(deviceModel.getDeviceData("gyroX")) +","+ str(deviceModel.getDeviceData("gyroY")) +","+ str(deviceModel.getDeviceData("gyroZ"))
         , " 角度:" + str(deviceModel.getDeviceData("angleX")) +","+ str(deviceModel.getDeviceData("angleY")) +","+ str(deviceModel.getDeviceData("angleZ"))
        , " 磁场:" + str(deviceModel.getDeviceData("magX")) +","+ str(deviceModel.getDeviceData("magY"))+","+ str(deviceModel.getDeviceData("magZ"))
        , " 经度:" + str(deviceModel.getDeviceData("lon")) + " 纬度:" + str(deviceModel.getDeviceData("lat"))
        , " 航向角:" + str(deviceModel.getDeviceData("Yaw")) + " 地速:" + str(deviceModel.getDeviceData("Speed"))
         , " 四元素:" + str(deviceModel.getDeviceData("q1")) + "," + str(deviceModel.getDeviceData("q2")) + "," + str(deviceModel.getDeviceData("q3"))+ "," + str(deviceModel.getDeviceData("q4"))
          )
    '''
    
    IMU_data=np.array([deviceModel.getDeviceData("angleX"),deviceModel.getDeviceData("angleY"),deviceModel.getDeviceData("angleZ"),deviceModel.getDeviceData("gyroX"),
                       deviceModel.getDeviceData("gyroY"),deviceModel.getDeviceData("gyroZ"),deviceModel.getDeviceData("accX"),deviceModel.getDeviceData("accY"),deviceModel.getDeviceData("accZ")])
    #q_imu_1.append(q_imu)
    q_imu_1.put(IMU_data)
   


    
    global _IsWriteF
    _IsWriteF = False             # 标记不可写入标识    Tag cannot write the identity
    #_writeF.close()               #关闭文件 Close file
     
def set_pybullet():
    ## 加载servo_client
    servo_client=p.connect(p.DIRECT)
        #servo_client=p.connect(p.GUI)
    p.setGravity(0, 0, -9.8,physicsClientId=servo_client)  # 设置重力值
    p.setAdditionalSearchPath(pd.getDataPath(),physicsClientId=servo_client)  # 设置pybullet_data的文件路径

        # 加载地面
    floor_servo = p.loadURDF("plane.urdf",physicsClientId=servo_client)
    box_servo=[]
    box_servo.append(floor_servo)

    startPos_servo=[0,0,1]
        # 加载urdf文件
    robot_servo = p.loadURDF("hexapod_34/urdf/hexapod_34.urdf", startPos_servo,physicsClientId=servo_client,useFixedBase=True,)
    dt=1/1000
    p.setTimeStep(dt,physicsClientId=servo_client)        
        

       

def read_imu(q_imu):
     #init servos
    global  q_imu_1
    q_imu_1=q_imu
    device = deviceModel.DeviceModel(
        "我的JY901",
        WitProtocolResolver(),
        JY901SDataProcessor(),
        "51_0"
    )

    if (platform.system().lower() == 'linux'):
        device.serialConfig.portName = "/dev/ttyUSB0"   #设置串口   Set serial port
    else:
        device.serialConfig.portName = "COM39"          #设置串口   Set serial port
    device.serialConfig.baud = 230400                     #设置波特率  Set baud rate
    device.ADDR=0x50
    device.openDevice()                                 #打开串口   Open serial port
    #setConfig()
    
    readConfig(device)                                  #读取配置信息 Read configuration information
    
    device.dataProcessor.onVarChanged.append(onUpdate)  #数据更新事件 Data update event
    #q_imu.put(IMU_data)
    




              

def reflex_(q_imu,ser):
    

    #set_pybullet()
    
    
    
    with open('force_real17.json', 'r') as f:
    
        data_read_json = json.load(f)
        force_r = np.asarray(data_read_json['contact_force'])
        phase_r = np.asarray(data_read_json['phase'])
        cpg_r = np.asarray(data_read_json['cpg'])
        theta_r = np.asarray(data_read_json['theta'])
        theta_ini_r=np.asarray(data_read_json['theta_ini'])
        ini_index_r=np.asarray(data_read_json['ini_index'])
        torque_r=np.asarray(data_read_json['torque_n'])
        end_pos_r=np.asarray(data_read_json['end_pos'])
    
    
    
    #with open('pos_0_5_10_1.json', 'r') as f:
    with open('pos_20_17.json', 'r') as f:
        
        data_read = json.load(f)
        positions_tick = np.asarray(data_read['positions_tick'])
        current_pos_tick = np.asarray(data_read['current_pos_tick'])
        goal_pos_sim = np.asarray(data_read['goal_pos_sim'])
        phase = np.asarray(data_read['phase'])
        
    step= 0
    cpg_index=ini_index_r[0]
    
    

    # init servos
    servos=Servos()
    voltage=servos.read_voltage(1)
    servos.set_position_control()
    #position_Read=servos.read_position_loop()
    position_all=range(18)
    print("Press any key to enable legs! (or press ESC to escape!)")
    #if getch() != chr(0x1b):
        #servos.enable_torque(position_all)
    
    
    
    
    swing_coef=2
    stance_coef=20
    step=0
    T=240
    T_count=0
    coef=2
    coef_stance=1
    step=0
    reflex=np.zeros(6)
    # 计数在每个swing or reflex中的步数
    swing_step_count=0
    IMU_data_init=q_imu.get(True,10)
    while( not q_imu.empty()):
        IMU_data_init=q_imu.get(True,10)
    



    
    for count in range(int(T*1)):
        start_time_t=time.time()
        
        # 接收imu的数据
        print("empty",q_imu.empty())
        IMU_data=q_imu.get(True,3)
        
        while( not q_imu.empty()):
            IMU_data=q_imu.get(True,10)
        #if count==0:
        #    imu_init=IMU_data
        #    IMU_data[0:3]=IMU_data[0:3]-imu_init[0:3]
        #else:
        #    IMU_data[0:3]=IMU_data[0:3]-imu_init[0:3]
        
             
        #IMU_data=np.array([0,0,0,0,0,0,])
        IMU_data_angles=IMU_data[0:3]-IMU_data_init[0:3] # 绕着 xy z轴转动  单独是度
        IMU_data_1=IMU_data_angles/180*math.pi # 观察中的imu是以弧度为单位
        (roll ,pitch,yaw)=IMU_data[0:3]/180*math.pi # 绕着 xy z轴转动
        
        # roll  绕着x轴旋转 x 轴是身体横向  
        # 仿真中roll  和实际中roll的方向相同
        print("roll pitch yaw: ",roll ,pitch,yaw)
        print("imu time: ",(time.time()-start_time_t)*1000)
        start_t0=time.time()
        position_Read=servos.read_all_positions() # np array 18
        
        end_t=time.time()
        print("read time: ",(end_t-start_t0)*1000)
        start_t1=time.time()
        theta_sim=real_angles_to_sim(position_Read)
    
        
        length_12=0.126
        length_13=0.266
        dz1=-length_12*math.tan(roll)
        dz2=-length_13*math.tan(roll)
        foot_z=np.zeros(6)
        
        print("cal time: ",(time.time()-start_time_t)*1000)
        
        
        start_t111=time.time()
        for agent_index in range(6):
            #start_t0=time.time()
            
            robot_joint_positions_agenti=theta_sim[agent_index]
            if agent_index>=3:
                end_pos,end_ori=get_forward_pos(-robot_joint_positions_agenti)
            else:
                end_pos,end_ori=get_forward_pos(robot_joint_positions_agenti)
                
            foot_z[agent_index]=end_pos[2]
            if agent_index%3==1:
                foot_z[agent_index]=end_pos[2]+dz1            
            if agent_index%3==2:
                foot_z[agent_index]=end_pos[2]+dz2
                
        relative_foot_z_all=foot_z-foot_z[0]
        print("loop time: ",(time.time()-start_t111)*1000)
        
        print("last time: ",(time.time()-start_time_t)*1000)
        observation_agenti=np.concatenate((position_Read,IMU_data_1,relative_foot_z_all),).tolist()
        data_array=struct.pack('<27f',*observation_agenti)	
        print("concatenate time: ",(time.time()-start_time_t)*1000)
        ser.write(data_array)
        while (time.time()-start_time_t)*1000<20.0:
            1
        print("end time: ",(time.time()-start_time_t)*1000)
        
        
        
            
        
        
           
            
        
        
        
        
            
        
        
       
    



import numpy as np

from Servos import *

if __name__ == '__main__':
    #global IMU_data 
    
    
    #IMU_data =np.array([1,1,1,0,0,0,1,1,1,])
    with open('force_real7.json', 'r') as f:
    
        data_read = json.load(f)
        force_r = np.asarray(data_read['contact_force'])
        phase_r = np.asarray(data_read['phase'])
        cpg_r = np.asarray(data_read['cpg'])
        theta_r = np.asarray(data_read['theta'])
        theta_ini_r=np.asarray(data_read['theta_ini'])
        ini_index_r=np.asarray(data_read['ini_index'])
        #torque_n=np.asarray(data_read['torque_n'])


    step= 0
    cpg_index=ini_index_r[0]
    
    #q_imu=queue.LifoQueue()
    q_imu= Queue()
    q_act=Queue()
    q_servo_obs_now= Queue()
    q_servo_obs_next=Queue()
    q_pos_read= Queue()
    q_obs1=Queue()
    
    
    model1_dir="/home/fast3/Desktop/DynamixelSDK-3.7.31/python/tests/protocol2_0/distill_model/action_net_one_BC_mlp_new2_1_70.pt"
    # socket

    
    Read_IMU = Process(target=read_imu,args=(q_imu,) )
    Read_IMU.start()
    time.sleep(6)
    ser = serial.Serial("/dev/ttyAMA1",115200*16,parity='E',timeout=0.00001)

    if not ser.isOpen():
        print("open failed")
    else:
        print("open success: ")
        print(ser)
    

    reflex_(q_imu,ser)
    #Reflex_ = Process(target=reflex_,args=(q_imu,) )
    #Reflex_.start()
    
    
    
    
        
