
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







def read_servos_only(servos,data_read,cpg_index,q_imu,socket_tcp,step):
    global imu_init
    #global position_Read
    #Timer(0.04,read_servos,args=(servos,data_read,cpg_index,q_imu,socket_tcp)).start()
    print("read servo:",time.time())
    agent_num=7
    force_r = np.asarray(data_read['contact_force'])
    phase_r = np.asarray(data_read['phase'])
    cpg_r = np.asarray(data_read['cpg'])
    theta_r = np.asarray(data_read['theta'])
    theta_ini_r=np.asarray(data_read['theta_ini'])
    ini_index_r=np.asarray(data_read['ini_index'])
    torque_r=np.asarray(data_read['torque_n'])
    end_pos_r=np.asarray(data_read['end_pos'])
    
    
    start_t=time.time()
    position_Read=servos.read_all_positions() # np array 18
    #while(flag==0):
    #    position_Read=servos.read_all_positions()
    #current_Read=servos.read_all_current() # np array 18
    #while(flag==0):
    #    current_Read=servos.read_all_current() # np array 18
        
    
    end_t=time.time()
    print("read time: ",end_t-start_t)
    
    theta_sim=real_angles_to_sim(position_Read)
    #torque_sim=real_current_to_sim_torque(current_Read)
    torque_sim=np.zeros_like(theta_sim)
    
    sim_angles_to_real(theta_sim)
    ## imu data
    IMU_data=q_imu.get(True,10)
    while( not q_imu.empty()):
        IMU_data=q_imu.get(True,10)
    if step==0:
        imu_init=IMU_data
        IMU_data[0:3]=IMU_data[0:3]-imu_init[0:3]
    else:
        IMU_data[0:3]=IMU_data[0:3]-imu_init[0:3]
        
    #IMU_data=np.array([0,0,0,0,0,0,])
    (roll ,pitch,yaw)=IMU_data[0:3]/180*math.pi # ���� xy z��ת��
    length_12=0.126
    length_13=0.266
    dz1=-length_12*math.tan(roll)
    dz2=-length_13*math.tan(roll)
    
    comtact_force_leg=np.zeros(agent_num-1)
    foot_z=np.zeros(agent_num-1)
    phase=phase_r[cpg_index,:]
    observation_temp=[]
    
    
    
    
    torque_n=torque_r[cpg_index]
    for agent_index in range(agent_num):

        if agent_index<6:
            robot_joint_positions_agenti=theta_sim[agent_index]
            robot_joint_Torque_agenti=torque_sim[agent_index]
            #robot_joint_Torque_agenti = torque_sim[3*agent_index:3*agent_index+3]
            #world_end_position=p.getLinkState(robot,3*agent_index+2,computeForwardKinematics=True)[0]
            #zworld_end_ori=p.getLinkState(robot,3*agent_index+2,computeForwardKinematics=True)[0]
            if agent_index>=3:
                end_pos,end_ori=get_forward_pos(-robot_joint_positions_agenti)
            else:
                end_pos,end_ori=get_forward_pos(robot_joint_positions_agenti)
            
            foot_z[agent_index]=end_pos[2]
            if agent_index%3==1:
                foot_z[agent_index]=end_pos[2]+dz1
                
            if agent_index%3==2:
                foot_z[agent_index]=end_pos[2]+dz2
        
            # ����error
            
            torque_target=torque_n[agent_index,:]
            torque_error=torque_target-robot_joint_Torque_agenti
            
            expeted_end_pos=end_pos_r[cpg_index,agent_index,:]
            end_pos_error=expeted_end_pos-end_pos



            # ��λ��Ϣ
            
            phase_continus=cpg_r[agent_index,2:4,cpg_index]
            
            ## ��ȫ֪��Ϣ
            contact_force_v=0
            contact_force_diff=0
            contact_force_l=0
            contact_point_num=0
            height_now_n_next=np.array([0,0])
            step_width=0
            #print("pos error",end_pos_error,"torque_error",torque_error,"torque_target",torque_target,"\n")
            
            
            observation_agenti = np.append(robot_joint_positions_agenti, robot_joint_Torque_agenti)
            observation_agenti = np.append(observation_agenti,contact_force_v/10)
            observation_agenti = np.append(observation_agenti,contact_force_diff/10)
            observation_agenti = np.append(observation_agenti,contact_force_l/10)
            observation_agenti = np.append(observation_agenti,contact_point_num)
            observation_agenti = np.append(observation_agenti,height_now_n_next*10)
            observation_agenti = np.append(observation_agenti,step_width*10)
            observation_agenti = np.append(observation_agenti,end_pos*10)
            observation_agenti = np.append(observation_agenti,end_pos_error[0:2]*100)
            observation_agenti = np.append(observation_agenti,phase_continus)
            observation_agenti = np.append(observation_agenti,torque_error)
            observation_temp.append(observation_agenti)
                
        else:
            IMU=IMU_data[0:3]*1/180*math.pi
            print("IMU",IMU_data[0:3])
            contact_force_flag=comtact_force_leg*phase*-1
            relative_foot_z=foot_z-foot_z[0]
            observation_agenti = np.append(IMU*10, contact_force_flag/10)
            observation_agenti = np.append(observation_agenti,relative_foot_z*10)
            observation_agenti = np.append(observation_agenti,IMU*10)
            observation_agenti = np.append(observation_agenti,np.zeros(2))
            observation_agenti = np.append(observation_agenti,np.zeros(3))
           
            observation_temp.append(observation_agenti)
            
    
    
    #return observation_temp
    observation = np.array(np.vstack(observation_temp),dtype=np.float32).flatten()
    #data_obs=struct.pack('<161f',*observation)
    
    
    #socket_tcp.send(data_obs)
    #print("send time:",time.time())
    return observation,position_Read



def onUpdate(deviceModel):
    """
    ���ݸ����¼�  Data update event
    :param deviceModel: �豸ģ��    Device model
    :return:
    """
    global IMU_data
    '''
    print("оƬʱ��:" + str(deviceModel.getDeviceData("Chiptime"))
         , " �¶�:" + str(deviceModel.getDeviceData("temperature"))
         , " ���ٶȣ�" + str(deviceModel.getDeviceData("accX")) +","+  str(deviceModel.getDeviceData("accY")) +","+ str(deviceModel.getDeviceData("accZ"))
         ,  " ���ٶ�:" + str(deviceModel.getDeviceData("gyroX")) +","+ str(deviceModel.getDeviceData("gyroY")) +","+ str(deviceModel.getDeviceData("gyroZ"))
         , " �Ƕ�:" + str(deviceModel.getDeviceData("angleX")) +","+ str(deviceModel.getDeviceData("angleY")) +","+ str(deviceModel.getDeviceData("angleZ"))
        , " �ų�:" + str(deviceModel.getDeviceData("magX")) +","+ str(deviceModel.getDeviceData("magY"))+","+ str(deviceModel.getDeviceData("magZ"))
        , " ����:" + str(deviceModel.getDeviceData("lon")) + " γ��:" + str(deviceModel.getDeviceData("lat"))
        , " �����:" + str(deviceModel.getDeviceData("Yaw")) + " ����:" + str(deviceModel.getDeviceData("Speed"))
         , " ��Ԫ��:" + str(deviceModel.getDeviceData("q1")) + "," + str(deviceModel.getDeviceData("q2")) + "," + str(deviceModel.getDeviceData("q3"))+ "," + str(deviceModel.getDeviceData("q4"))
          )
    '''
    
    IMU_data=np.array([deviceModel.getDeviceData("angleX"),deviceModel.getDeviceData("angleY"),deviceModel.getDeviceData("angleZ"),deviceModel.getDeviceData("gyroX"),
                       deviceModel.getDeviceData("gyroY"),deviceModel.getDeviceData("gyroZ"),deviceModel.getDeviceData("accX"),deviceModel.getDeviceData("accY"),deviceModel.getDeviceData("accZ")])
    #q_imu_1.append(q_imu)
    q_imu_1.put(IMU_data)
   


    
    global _IsWriteF
    _IsWriteF = False             # ��ǲ���д���ʶ    Tag cannot write the identity
    #_writeF.close()               #�ر��ļ� Close file
     


def read_imu(q_imu):
     #init servos
    global  q_imu_1
    q_imu_1=q_imu
    device = deviceModel.DeviceModel(
        "�ҵ�JY901",
        WitProtocolResolver(),
        JY901SDataProcessor(),
        "51_0"
    )

    if (platform.system().lower() == 'linux'):
        device.serialConfig.portName = "/dev/ttyUSB0"   #���ô���   Set serial port
    else:
        device.serialConfig.portName = "COM39"          #���ô���   Set serial port
    device.serialConfig.baud = 230400                     #���ò�����  Set baud rate
    device.ADDR=0x50
    device.openDevice()                                 #�򿪴���   Open serial port
    #setConfig()
    
    readConfig(device)                                  #��ȡ������Ϣ Read configuration information
    
    device.dataProcessor.onVarChanged.append(onUpdate)  #���ݸ����¼� Data update event
    #q_imu.put(IMU_data)
    

              

def reflex_(q_imu):
    


    #with open('pos_0_5_10_1.json', 'r') as f:
    with open(tick_file_name, 'r') as f:
        
        data_read = json.load(f)
        positions_tick = np.asarray(data_read['positions_tick'])
        current_pos_tick = np.asarray(data_read['current_pos_tick'])
        goal_pos_sim = np.asarray(data_read['goal_pos_sim'])
        phase = np.asarray(data_read['phase'])
        
    step= 0

    # init servos
    servos=Servos()
    voltage=servos.read_voltage(1)
    servos.set_position_control()
    #position_Read=servos.read_position_loop()
    position_all=range(18)
    print("Press any key to enable legs! (or press ESC to escape!)")
    #if getch() != chr(0x1b):
        #servos.enable_torque(position_all)
    servos.enable_torque(position_all)
    position_Read=servos.read_all_positions()
    print("read position:",position_Read)
    
    theta_sim=goal_pos_sim[step]    
    angles_real=sim_angles_to_real(theta_sim)
    servos.Robot_initialize(angles_real)
    goal_theta_tick=angles_to_tick(angles_real)
    time.sleep(5)
    position_Read=servos.read_all_positions()
    position_Read_tick=angles_to_tick(position_Read)
    flat_cpg_tick=current_pos_tick[step] #18
    

    theta_tick=angles_to_tick(angles_real)
    servos.write_all_positions(theta_tick)
    servos.write_all_positions(theta_tick)
    servos.write_all_positions(theta_tick)



    # init vaiables
    positions=[]
    traj_error_buf=np.zeros((4,18))
    on_reflex=np.zeros(6)
    reflex_index=np.ones(6)*(-10)
    on_reflex_stance=np.zeros(6)
    reflex_index_stance=np.ones(6)*(-10)
    stance_step_per_reflex=np.ones(6)*(0)
    swing_step_per_reflex=np.ones(6)*(0)
    sum_leg_all=[]
    reflex_real_all=[]
    on_reflex_all=[]
    traj_error_all=[]
    count_all=[]
    csv_rows=[]
    
    
    step=0
    T=240
    T_count=0
    coef=1
    coef_stance=1
    step=0
    reflex=np.zeros(6)
    # ������ÿ��swing or reflex�еĲ���
    swing_step_count=0



    
    for count in range(int(T*4)):
        start_time_t=time.time()
        
        # ����imu������
        IMU_data=q_imu.get(True,10)
        
        while( not q_imu.empty()):
            IMU_data=q_imu.get(True,10)
        if count==0:
            imu_init=IMU_data
            IMU_data[0:3]=IMU_data[0:3]-imu_init[0:3]
        else:
            IMU_data[0:3]=IMU_data[0:3]-imu_init[0:3]
                
        #IMU_data=np.array([0,0,0,0,0,0,])
        (roll ,pitch,yaw)=IMU_data[0:3]/180*math.pi # ���� xy z��ת��
        print("roll pitch yaw: ",roll ,pitch,yaw)
        
        theta_sim=goal_pos_sim[step]    
        angles_real=sim_angles_to_real(theta_sim)
        theta_tick=angles_to_tick(angles_real)
        servos.write_all_positions(theta_tick)
        # read feedback
        position_Read=servos.read_all_positions()
        position_Read_tick=angles_to_tick(position_Read)
        #flat_cpg_tick=current_pos_tick[count] #18
        
            
 
            
        csv_row=[]
        csv_row=[count,T_count,theta_sim,theta_tick,position_Read_tick,IMU_data,voltage,]
        csv_rows.append(csv_row)
            
        
        while (time.time()-start_time_t)*1000<20.00:
            1
        end_time_t=time.time()
        print("last time",time.time()-start_time_t,"count:",count)
           
            
        
        if step==239:
            step=0
        else:
            step=step+1
        if step%T==int(T/4-1):
            T_count+=1

    
    # csv
    with open(output_file_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['count','T_count','theta_sim','theta_tick','position_Read_tick','IMU_data','voltage'])
        writer.writerows(csv_rows)
        print("save file!")

    data = {'positions': positions}
    data_json = json.dumps(data, cls=NumpyArrayEncoder)
    DXLn_ID=range(18)
    #servos.disable_torque(DXLn_ID)
    



import numpy as np

from Servos import *

if __name__ == '__main__':
    #global IMU_data 
    file_first_name='force_real_wave_'
    file_last_name='17_1'
    file_name=file_first_name+file_last_name+'.json'
    output_file_first_name='pos_cpg_5_'
    record_file_first_name='record_cpg_5_'
    output_file_name=record_file_first_name+file_last_name+'.csv'
    tick_file_name=output_file_first_name+file_last_name+'.json'
    
    #file_name="force_real17.json"
    #tick_file_name="pos_20_17_1.json"
    #output_file_name='pos_cpg_3_2.csv'
    
        
    
    #IMU_data =np.array([1,1,1,0,0,0,1,1,1,])
    with open(file_name, 'r') as f:
    
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
    if cpg_index>240:
        cpg_index=cpg_index-240*int(cpg_index/240)
    
    #q_imu=queue.LifoQueue()
    q_imu= Queue()
    q_act=Queue()
    # socket

    
    Read_IMU = Process(target=read_imu,args=(q_imu,) )
    Read_IMU.start()
    time.sleep(2)
    reflex_(q_imu)
    #Reflex_ = Process(target=reflex_,args=(q_imu,) )
    #Reflex_.start()
    
    
    
    
        
