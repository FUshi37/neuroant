import json
from sys import path
path.append("../../")
import math
import numpy as np
import random
#import Servos
from scipy.signal import savgol_filter


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)




def sim_angles_to_real0(theta):
    # theta [6,3]
    real_angles=np.zeros_like(theta).flatten()
    for i in range(6):
        if i<3:
            
            hip=180+theta[i,0]/math.pi*180 
            
            knee=180-theta[i,1]/math.pi*180
            ankle=62.69+theta[i,2]/math.pi*180
            
            real_angles[3+i*6+0]=hip
            real_angles[3+i*6+1]=knee
            real_angles[3+i*6+2]=ankle
            
        else:
            hip=180+theta[i,0]/math.pi*180 
            knee=180+theta[i,1]/math.pi*180
            ankle=62.69-theta[i,2]/math.pi*180
            
            real_angles[0+(i-3)*6+0]=hip
            real_angles[0+(i-3)*6+1]=knee
            real_angles[0+(i-3)*6+2]=ankle
        
    return real_angles

def angles_to_tick(angles):
    theta_tick=np.zeros_like(angles)
    for i in range(18):
        theta_tick[i]=int(angles[i]/180*2048)
    return theta_tick
            

def real_angles_to_sim0(real_angles):
    # theta [6,3]
    theta=np.zeros((6,3))
    for i in range(6):
        if i<3:
            theta[i,0]= (real_angles[3+i*6+0]-180)/180*math.pi
            theta[i,1]= (-real_angles[3+i*6+1]+180)/180*math.pi
            theta[i,2]= (real_angles[3+i*6+2]-62.69)/180*math.pi
            
            
            
        else:
            theta[i,0]= (real_angles[0+(i-3)*6+0]-180)/180*math.pi
            theta[i,1]= (real_angles[0+(i-3)*6+1]-180)/180*math.pi
            theta[i,2]= (-real_angles[0+(i-3)*6+2]+62.69)/180*math.pi
            
           
        
    return theta      

def real_angles_to_sim(real_angles):
    # theta [6,3]
    
    theta=np.zeros((6,3))
    for i in range(6):
        if i<3:
            theta[i,0]= -(real_angles[0+i*6+0]-180)/180.0*math.pi
            theta[i,1]= (-real_angles[0+i*6+1]+180)/180.0*math.pi
            theta[i,2]= (real_angles[0+i*6+2]-62.69)/180.0*math.pi
            
            
            
        else:
            theta[i,0]= -(real_angles[3+(i-3)*6+0]-180)/180.0*math.pi
            theta[i,1]= (real_angles[3+(i-3)*6+1]-180)/180.0*math.pi
            theta[i,2]= (-real_angles[3+(i-3)*6+2]+62.69)/180.0*math.pi
            
           
        
    return theta

def sim_angles_to_real(theta):
    # theta [6,3]  杈撳�? np array [18]
    real_angles=np.zeros_like(theta).flatten()*1.00000
    for i in range(6):
        if i<3:
            
            hip=(180-theta[i,0]/math.pi*180)
            
            knee=180-theta[i,1]/math.pi*180+17
            ankle=62.69+theta[i,2]/math.pi*180
            
            real_angles[0+i*6+0]=hip
            real_angles[0+i*6+1]=knee
            real_angles[0+i*6+2]=ankle
            
        else:
            hip=(180-theta[i,0]/math.pi*180) 
            knee=180+theta[i,1]/math.pi*180+17
            ankle=62.69-theta[i,2]/math.pi*180
            
            real_angles[3+(i-3)*6+0]=hip
            real_angles[3+(i-3)*6+1]=knee
            real_angles[3+(i-3)*6+2]=ankle
        
    return real_angles    

def real_current_to_sim_torque(real_current):
    # theta [6,3]
    torque_sim=np.zeros((6,3))
    real_torque1=real_current.reshape(6,3)
    for i in range(6):
        if i<3:
            torque_sim[i,0]= real_current[3+i*6+0]*2.69/1000*1.82-0.2576
            torque_sim[i,1]= -(real_current[3+i*6+1]*2.69/1000*1.82-0.2576)
            torque_sim[i,2]= real_current[3+i*6+2]*2.69/1000*1.82-0.2576
            
            
            
        else:
            torque_sim[i,0]= real_current[0+(i-3)*6+0]*2.69/1000*1.82-0.2576
            torque_sim[i,1]= real_current[0+(i-3)*6+1]*2.69/1000*1.82-0.2576
            torque_sim[i,2]= -(real_current[0+(i-3)*6+2]*2.69/1000*1.82-0.2576)
            
           
        
    return torque_sim     

def real_torque_to_sim_torque(real_current):
    # theta [6,3]
    torque_sim=np.zeros((6,3))
    
    for i in range(6):
        if i<3:
            torque_sim[i,0]= real_current[3+i*6+0]
            torque_sim[i,1]= -(real_current[3+i*6+1])
            torque_sim[i,2]= real_current[3+i*6+2]
            
            
            
        else:
            torque_sim[i,0]= real_current[0+(i-3)*6+0]
            torque_sim[i,1]= real_current[0+(i-3)*6+1]
            torque_sim[i,2]= -(real_current[0+(i-3)*6+2])
            
           
        
    return torque_sim            



folder_name='data_real_robot'
file_name='record_data/mine_fault1_no2_3.json'
#file_first_name='mine_fault_'
file_last_name='test.json'
#file_name=folder_name+file_first_name+file_last_name


output_file_first_name='pos_cpg_5_'
output_file_name=output_file_first_name+file_last_name
#with open('force_real17.json', 'r') as f:
#with open('force_real_four_30_1.json','r') as f:
with open(file_name,'r') as f:    
    data_read = json.load(f)
    theta_target = np.asarray(data_read['theta'])
    theta_sim_r_old = np.asarray(data_read['theta_real_n'])
    IMU_r = np.asarray(data_read['IMU_n'])
theta_sim_r=np.zeros_like(theta_sim_r_old)
for i in range(6):
    for j in range(3):
        theta_sim_r[:,i,j]=savgol_filter(theta_sim_r_old[:,i,j],13,3)     
theta_sim_r=theta_sim_r_old
test_length=theta_sim_r.shape[0]
step= 0

    

import numpy as np

from Servos import *


servos=Servos()
#servos.light_LED()
#goal_position=2048*np.ones(1,18)
servos.read_voltage(1
                    )
servos.set_position_control()
goal_position=np.array([180,204,85,180,204,85])
#position_Read=servos.read_position_loop()
#print("read position:",position_Read)
DXLn_ID=[0,1,2,3,4,5]
servos.enable_torque(DXLn_ID)
#print("Press any key to continue! (or press ESC to move leg2!)")
#if getch() == chr(0x1b):
#    servos.write_some_positions(goal_position,DXLn_ID)
time.sleep(1)
DXLn_ID=[6,7,8,9,10,11]
servos.enable_torque(DXLn_ID)
#print("Press any key to continue! (or press ESC to move leg2!)")
#if getch() == chr(0x1b):
#    servos.write_some_positions(goal_position,DXLn_ID)


DXLn_ID=[12,13,14,15,16,17]
servos.enable_torque(DXLn_ID)
#print("Press any key to continue! (or press ESC to move leg2!)")
#if getch() == chr(0x1b):
#    servos.write_some_positions(goal_position,DXLn_ID)

position_Read=servos.read_all_positions()
print("read position:",position_Read)




positions=[]
phase_all=[]
goal_pos_sim=[]
current_pos_tick=[]
# 初始化
theta_sim=theta_sim_r[1,:,:]
angles_real=sim_angles_to_real(theta_sim)
servos.Robot_initialize(angles_real)
theta_tick=angles_to_tick(angles_real)
servos.write_all_positions(theta_tick)
time.sleep(0.5)
servos.write_all_positions(theta_tick)
Interpolation_num=50
# -0.3 0.33
for step in range(test_length-1):
    
   
    
    


    start_time_t=time.time()
    theta_sim=theta_sim_r[step+1,:,:]
    angles_real=sim_angles_to_real(theta_sim)
    theta_tick=angles_to_tick(angles_real)
    positions.append(theta_tick)
    goal_pos_sim.append(theta_sim)
    
    servos.write_all_positions_smooth(theta_tick,Interpolation_num)
    #servos.write_all_positions_angles(angles_real)
    #position_goal = np.trunc(angles_real / 360 * 4096)  # tick
    
    
    position_Read=servos.read_all_positions()
    position_tick=angles_to_tick(position_Read)
    current_pos_tick.append(position_tick)
    position_error=angles_real-position_Read
    #print("position_error",position_error)
    #print("position target",angles_real)
    
    
    #current_read=servos.read_all_torque()
    #torque_sim=real_torque_to_sim_torque(current_read)
    #print("torque sim",torque_read_sim,"\n")
    #print("torque real",torque_sim,"\n")
    #print("current",current_read,"\n")
    #while (time.time()-start_time_t)*1000<20.00:
    #    1
    step+=1
    end_time_t=time.time()
    print("last time",(end_time_t-start_time_t)*1000)
    
    #print("Press any key to continue! (or press ESC to escape)")
    #if getch() != chr(0x1b):
        #servos.write_some_positions(angles_real[0:6],DXLn_ID)
        #servos.write_all_positions_angles(angles_real)
    #time.sleep(0.07)
        
    
    
    

str1="pos_cpg_test"+".json"
print(output_file_name)
#data = {'positions': positions,'current_pos_tick':current_pos_tick,'goal_pos_sim':goal_pos_sim,'phase':phase_all,}
data = {'positions_tick': positions,'current_pos_tick':current_pos_tick,'goal_pos_sim':goal_pos_sim,}
data_json = json.dumps(data, cls=NumpyArrayEncoder)
with open(output_file_name, 'w') as f:
    json.dump(data, f, cls=NumpyArrayEncoder)
    print("save data done!")
#print("Press any key to disable legs")
DXLn_ID=range(18)
#servos.disable_torque(DXLn_ID)
        
