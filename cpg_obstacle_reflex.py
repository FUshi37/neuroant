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
from distill_model.action_net import *


def set_obs(q_obs1,q_obs2,q_obs3,q_obs4,q_obs5,q_obs6,):
    while(1):
        obs=torch.rand((1,66))
        q_obs1.put(obs)
        q_obs2.put(obs)
        q_obs3.put(obs)
        q_obs4.put(obs)
        q_obs5.put(obs)
        q_obs6.put(obs)
        print("put obs")
        time.sleep(0.0199)
        
    


def load_local_model():
    obs_dim_t = 11

    act_dim = 1
    action_type = 'Continuous'
    num_agents=6
    n_embd=64
    device = torch.device("cpu")
    obs_dim=obs_dim_t*num_agents
        
        
    action_net2=ActionNet(obs_dim, n_embd, act_dim,device)
    action_net3=ActionNet(obs_dim, n_embd, act_dim,device)
    action_net4=ActionNet(obs_dim, n_embd, act_dim,device)
    action_net5=ActionNet(obs_dim, n_embd, act_dim,device)
    action_net6=ActionNet(obs_dim, n_embd, act_dim,device)
        
    action_net1=ActionNet(obs_dim, n_embd*1, act_dim,device)

    model1_dir="distill_model/action_net1_BC_mlp_1_20.pt"
    model2_dir="distill_model/action_net2_BC_mlp_1_20.pt"
    model3_dir="distill_model/action_net3_BC_mlp_1_20.pt"
    model4_dir="distill_model/action_net4_BC_mlp_1_20.pt"
    model5_dir="distill_model/action_net5_BC_mlp_1_20.pt"
    model6_dir="distill_model/action_net6_BC_mlp_1_20.pt"

    saved_model_1 = torch.load(model1_dir,map_location=torch.device('cpu'))
    saved_model_2 = torch.load(model2_dir,map_location=torch.device('cpu'))
    saved_model_3 = torch.load(model3_dir,map_location=torch.device('cpu'))
    saved_model_4 = torch.load(model4_dir,map_location=torch.device('cpu'))
    saved_model_5 = torch.load(model5_dir,map_location=torch.device('cpu'))
    saved_model_6 = torch.load(model6_dir,map_location=torch.device('cpu'))
    action_net1.load_state_dict(saved_model_1)
    action_net2.load_state_dict(saved_model_2)
    action_net3.load_state_dict(saved_model_3)
    action_net4.load_state_dict(saved_model_4)
    action_net5.load_state_dict(saved_model_5)
    action_net6.load_state_dict(saved_model_6)

    return action_net1,action_net2,action_net3,action_net4,action_net5,action_net6


def actor_act(q_obs,action_net):
    while 1:
        print("q_obs empty  ",q_obs.empty())
        obs=q_obs.get(True,10)
        print("obs",obs)
        
        #while( not q_obs.empty()):
        #    obs=q_obs.get(True,10)
        print("action net",action_net)
        start_t=time.time()
        action=action_net(obs)
        print("action",action)
        end_t=time.time()
        pid = os.getpid()
        # 显示进程号
        
        print("id:",pid,"action:",action,"last time",end_t-start_t)
    return action

def load_and_actor(model1_dir,q_obs):
    obs_dim_t = 11

    act_dim = 1
    action_type = 'Continuous'
    num_agents=6
    n_embd=32
    device = torch.device("cpu")
    obs_dim=obs_dim_t*num_agents
    
    action_net1=ActionNet(obs_dim, n_embd*1, act_dim*1,device)
    saved_model_1 = torch.load(model1_dir,map_location=torch.device('cpu'))
    #action_net1.load_state_dict(saved_model_1)
    
    while 1:
        print("q_obs empty  ",q_obs.empty())
        obs=q_obs.get(True,10)
        #print("obs",obs)
        
        #while( not q_obs.empty()):
        #    obs=q_obs.get(True,10)
        #print("action net",action_net1)
        start_t=time.time()
        action=action_net1(obs)
        #action=action_net1(obs)
        #action=action_net1(obs)
        
        #print("action",action)
        end_t=time.time()
        pid = os.getpid()
        # 显示进程号
        
        print("id:",pid,"action:",action,"last time",(end_t-start_t)*1000)
    
    
    
    
    
        


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
    (roll ,pitch,yaw)=IMU_data[0:3]/180*math.pi # 绕着 xy z轴转动
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
        
            # 误差的error
            
            torque_target=torque_n[agent_index,:]
            torque_error=torque_target-robot_joint_Torque_agenti
            
            expeted_end_pos=end_pos_r[cpg_index,agent_index,:]
            end_pos_error=expeted_end_pos-end_pos



            # 相位信息
            
            phase_continus=cpg_r[agent_index,2:4,cpg_index]
            
            ## 不全知信息
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
    

              

def reflex_(q_imu):
    


    with open('pos_0_5_10_1.json', 'r') as f:
        
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
    T=480 
    T_count=0
    coef=2
    coef_stance=1.5
    step=0
    reflex=np.zeros(6)
    # 计数在每个swing or reflex中的步数
    swing_step_count=0



    
    for count in range(int(T*6)):
        start_time_t=time.time()
        
        # 接收imu的数据
        IMU_data=q_imu.get(True,10)
        
        while( not q_imu.empty()):
            IMU_data=q_imu.get(True,10)
        if count==0:
            imu_init=IMU_data
            IMU_data[0:3]=IMU_data[0:3]-imu_init[0:3]
        else:
            IMU_data[0:3]=IMU_data[0:3]-imu_init[0:3]
                
        #IMU_data=np.array([0,0,0,0,0,0,])
        (roll ,pitch,yaw)=IMU_data[0:3]/180*math.pi # 绕着 xy z轴转动
        print("roll pitch yaw: ",roll ,pitch,yaw)
            
        
        if count<4:
            # just give command
            theta_sim=goal_pos_sim[count]    
            angles_real=sim_angles_to_real(theta_sim)
            theta_tick=angles_to_tick(angles_real)
            servos.write_all_positions(theta_tick)
            # read feedback
            position_Read=servos.read_all_positions()
            position_Read_tick=angles_to_tick(position_Read)
            flat_cpg_tick=current_pos_tick[count] #18
            traj_error_buf[count]=flat_cpg_tick-position_Read_tick
            on_reflex=np.zeros(6)
            
        else:
            # judhe whwther or not is a reflex
            # 在判断一开始的reflex 之前就清零
            # judge reflex 
            phase_now=phase[step]
            last_phase=phase[step-1]
            reflex,sum_leg=judge_reflex(traj_error_buf)
            reflex_sim=real_reflex_leg_to_sim(reflex)
            reflex_sim=swing_reflex(reflex_sim,phase_now)
            reflex_stance,sum_leg_stance=judge_reflex_stance(traj_error_buf)
            reflex_stance_sim0=real_reflex_leg_to_sim(reflex_stance)
            reflex_stance_sim=stance_reflex(reflex_stance_sim0,phase_now)
            print("sum_leg",sum_leg,'sum_leg_stance',sum_leg_stance)
            
            #print("count:",count,"reflex",reflex,"on reflex",on_reflex)
            # on_reflex 在sim的基础上
            
            #on_reflex=swing_reflex(on_reflex,phase_now)
            #on_reflex_stance=stance_reflex(on_reflex_stance,phase_now)
            if reflex_sim.any():
                reflex_index[reflex_sim>0]=T_count
                on_reflex[reflex_sim>0]=1
                
            if reflex_stance_sim.any():
                reflex_index_stance[reflex_stance_sim>0]=T_count
                on_reflex_stance[reflex_stance_sim>0]=1
                on_reflex_stance[on_reflex>0]=0
            if on_reflex_stance.any():
                stance_step_per_reflex[on_reflex_stance>0]+=1
            if on_reflex_stance.any():
                swing_step_per_reflex[on_reflex>0]+=1
                
            swing_step_per_reflex[swing_step_per_reflex>239]=0
            stance_step_per_reflex[stance_step_per_reflex>239]=0
            
            
            # swing 和stance  交界处 error 清零
            swing_step_per_reflex[last_phase*phase_now<0]=0
            stance_step_per_reflex[last_phase*phase_now<0]=0
            swing_step_count+=1
            if (last_phase*phase_now)[0]<1:
                traj_error_buf=np.zeros_like(traj_error_buf)
                swing_step_count=0
                
            for i in range(6):# 当进入下一周期后 默认不在reflex范围内
                if (T_count-reflex_index[i])==1 and on_reflex[i]>0:
                    on_reflex[i]=0
                    reflex_index[i]=-2
                    swing_step_per_reflex[i]=0
                if (T_count-reflex_index_stance[i])==1 and on_reflex_stance[i]>0:
                    on_reflex_stance[i]=0
                    reflex_index_stance[i]=-2
                    stance_step_per_reflex[i]=0
            # 将reflex  都转换到仿真的维度中  使得能够与phase 相关
            #on_reflex_sim=real_reflex_leg_to_sim(on_reflex)
            #on_reflex_stance_sim=real_reflex_leg_to_sim(on_reflex_stance)     
            
            #theta_new_sim=get_reflex_theta(theta_sim,coef,reflex_sim,on_reflex_sim,phase_now,)
            
            
            
            
            
            
            
            
            
            
            if swing_step_count<10:
                coef=1
                #coef_stance=1
            else:
                coef=2
                #coef_stance=1.5
            theta_sim=goal_pos_sim[step] 
            theta_new_sim=get_reflex_theta_all(theta_sim,coef,coef_stance,reflex_sim,on_reflex,reflex_stance_sim,on_reflex_stance,phase_now,stance_step_per_reflex,swing_step_per_reflex)
            angles_real=sim_angles_to_real(theta_new_sim)
            theta_tick=angles_to_tick(angles_real)
            servos.write_all_positions(theta_tick)
            
            # read feedback
            position_Read=servos.read_all_positions()
            position_Read_tick=angles_to_tick(position_Read)
            flat_cpg_tick=current_pos_tick[step] #18
            
            on_reflex_real=sim_reflex_leg_to_real(on_reflex)
            on_reflex_stance_real=sim_reflex_leg_to_real(on_reflex_stance)
            
            
            for i in range(6):
                if on_reflex_real[i]==1:
                    flat_cpg_tick[3*i:3*i+3]=theta_tick[3*i:3*i+3]
                elif on_reflex_stance_real[i]==1:
                    flat_cpg_tick[3*i:3*i+3]=theta_tick[3*i:3*i+3]
         
            ## buffer 升级
            for i_index in range(3):
                traj_error_buf[i_index]=traj_error_buf[i_index+1]
            traj_error_buf[3]=flat_cpg_tick-position_Read_tick
            
            # csv writer
            #traj_error_all.append(flat_cpg_tick-position_Read_tick)
            #count_all.append(count)
            #reflex_real_all.append(reflex)
            #on_reflex_all.append(on_reflex)
            #sum_leg_all.append(sum_leg)
            on_reflex2=copy.copy(on_reflex)
            on_reflex_stance2=copy.copy(on_reflex_stance)
        
            reflex_index2=copy.copy(reflex_index)
            reflex_index_stance2=copy.copy(reflex_index_stance)
            
            
            
            csv_row=[]
            csv_row=[count,T_count,sum_leg,reflex_sim,reflex_stance_sim,on_reflex2,on_reflex_stance2,reflex_index2,reflex_index_stance2,swing_step_count,flat_cpg_tick,position_Read_tick,IMU_data,voltage]
            csv_rows.append(csv_row)
            
            
            
            
            
            
            
            
            
            
        
            #print("current",current_read,"\n")
            
            print("reflex_stance",reflex_stance_sim,"on reflex",on_reflex_stance,'reflex_index_stance',reflex_index_stance,'T_count',)
            while (time.time()-start_time_t)*1000<10.00:
                1
            end_time_t=time.time()
            print("last time",time.time()-start_time_t,"count:",count,"reflex",reflex_sim,"on reflex",on_reflex,'reflex_index',reflex_index,'T_count',T_count)
           
            
        
        
        
        
            
        
        
        if step==479:
            step=0
        else:
            step=step+1
        if step%T==int(T/4-1):
            T_count+=1

    
    # csv
    with open('data_reflex_imu_stairs_4_7_5_7.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['count','T_count','sum_leg','reflex_sim','reflex_stance_sim','on_reflex2','on_reflex_stance2','reflex_index2','reflex_index_stance2','swing_step_count','flat_cpg_tick','position_Read_tick','IMU_data'])
        writer.writerows(csv_rows)

    data = {'positions': positions}
    data_json = json.dumps(data, cls=NumpyArrayEncoder)
    



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
    q_obs1=Queue()
    q_obs2=Queue()
    q_obs3=Queue()
    q_obs4=Queue()
    q_obs5=Queue()
    q_obs6=Queue()
    
    
    
    model1_dir="/home/fast3/Desktop/DynamixelSDK-3.7.31/python/tests/protocol2_0/distill_model/action_net_one_BC_mlp_new2_1_70.pt"
    model2_dir="distill_model/action_net2_BC_mlp_1_20.pt"
    model3_dir="distill_model/action_net3_BC_mlp_1_20.pt"
    model4_dir="distill_model/action_net4_BC_mlp_1_20.pt"
    model5_dir="distill_model/action_net5_BC_mlp_1_20.pt"
    model6_dir="distill_model/action_net6_BC_mlp_1_20.pt"

    
    # socket
    #action_net1,action_net2,action_net3,action_net4,action_net5,action_net6=load_local_model()
    
    Read_IMU = Process(target=read_imu,args=(q_imu,) )
    Read_IMU.start()
    
    
    
    set_obs1 = Process(target=set_obs,args=(q_obs1,q_obs2,q_obs3,q_obs4,q_obs5,q_obs6,) )
    set_obs1.start()
    
    #actor1 = Process(target=actor_act,args=(q_obs1,action_net1,) )
    #actor1.start()
    
    
    actor1 = Process(target=load_and_actor,args=(model2_dir,q_obs1,) )
    actor1.start()
    #actor2 = Process(target=load_and_actor,args=(model2_dir,q_obs2,) )
    #actor2.start()
    #actor3 = Process(target=load_and_actor,args=(model3_dir,q_obs3,) )
    #actor3.start()
    
    #actor2 = Process(target=actor_act,args=(q_obs2,action_net2,) )
    #actor2.start()
    #actor3 = Process(target=actor_act,args=(q_obs3,action_net3,) )
    #actor3.start()
    #actor4 = Process(target=actor_act,args=(q_obs4,action_net4,) )
    #actor4.start()
    #actor5 = Process(target=actor_act,args=(q_obs5,action_net5,) )
    #actor5.start()
    #actor6 = Process(target=actor_act,args=(q_obs6,action_net6,) )
    #actor6.start()
    
    #
    #time.sleep(2)
    #reflex_(q_imu)
    #Reflex_ = Process(target=reflex_,args=(q_imu,) )
    #Reflex_.start()
    '''
    while 1:
        start_t=time.time()
    
        a1=torch.rand((1,66))
        #ac1=action_net1(a1)
        ac2=action_net2(a1)
        #ac3=action_net3(a1)
        #ac4=action_net4(a1)
        #ac5=action_net5(a1)
        #ac6=action_net6(a1)
        end_t=time.time()
        print("last time",(end_t-start_t)*1000)
    '''
    
        
