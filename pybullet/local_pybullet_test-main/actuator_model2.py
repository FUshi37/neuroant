import pybullet as p
import pybullet_data as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from math import *
import math
import csv

from pdControllerExplicit import*
from pdControllerStable import*

from plt_traj_stairs import *



def __rpy2quaternion(roll, pitch, yaw):
        x=math.sin(pitch/2)*math.sin(yaw/2)*math.cos(roll/2)+math.cos(pitch/2)*math.cos(yaw/2)*math.sin(roll/2)
        y=math.sin(pitch/2)**math.cos(yaw/2)*math.cos(roll/2)+math.cos(pitch/2)*math.sin(yaw/2)*math.sin(roll/2)
        z=math.cos(pitch/2)*math.sin(yaw/2)*math.cos(roll/2)-math.sin(pitch/2)*math.cos(yaw/2)*math.sin(roll/2)
        w=math.cos(pitch/2)*math.cos(yaw/2)*math.cos(roll/2)-math.sin(pitch/2)*math.sin(yaw/2)*math.sin(roll/2)
        ori=[x,y,z,w]
        return ori


def get_obseration(robot):
     
    robot_joints_states_agenti = p.getJointState(robot, 0,)
    robot_joint_positions =robot_joints_states_agenti[0]
    return robot_joint_positions

def get_obserations_multi(robot):

    JointIndicies_agenti=range(18)
    robot_joints_states_agenti = p.getJointStates(robot, jointIndices=JointIndicies_agenti,)
    robot_joint_positions = np.array([x[0] for x in robot_joints_states_agenti])
    robot_joint_positions=np.array(robot_joint_positions).reshape((6,3))
     
    return robot_joint_positions


def get_contactPoints(robot,floor):
    result = np.zeros((6))
    lateral_contacts=np.zeros((6))
    
    for j in range(6):
        foot_contacts = []
        contact_force_v = 0
        contact_force_l = 0

        for body_1 in floor:
            foot_contact = p.getContactPoints(bodyA=robot, bodyB=body_1, linkIndexA=3 * j + 2)
            if len(foot_contact):
                foot_contacts.append(foot_contact)
        if not len(foot_contacts):
            link_name = (p.getJointInfo(robot, 3 * j + 2)[12]).decode()
            contact_force_v = 0
            contact_force_l = 0
        else:
            for f_contacts in foot_contacts:
                for f_contact in f_contacts:
                    link_index = f_contact[3]
                    contact_normal = f_contact[7]
                    contact_force_v = f_contact[9] * contact_normal[2] / np.linalg.norm(contact_normal)
                    contact_force_l = f_contact[9] * np.linalg.norm(contact_normal[0:1]) / np.linalg.norm(
                        contact_normal)
                    # print(contact_normal)
                    if link_index >= 0:
                        link_name = (p.getJointInfo(robot, link_index)[12]).decode()
                    else:
                        link_name = 'base'

        # result.append((link_name, f_contact[9]))
        result[ j] =  contact_force_v
        lateral_contacts[j]=contact_force_l
        
    return result,lateral_contacts


def real_angles_to_sim(real_angles):
    # theta [6,3]
    real_angles_2=np.zeros_like(real_angles)*1.0000
    for i in range(18):
        real_angles_2[i]=real_angles[i]*1.0000/2048*180
    theta=np.zeros((6,3))
    for i in range(6):
        if i<3:
            theta[i,0]= -(real_angles_2[0+i*6+0]-180)/180.0*math.pi
            theta[i,1]= (-real_angles_2[0+i*6+1]+180)/180.0*math.pi
            theta[i,2]= (real_angles_2[0+i*6+2]-62.69)/180.0*math.pi
            
            
            
        else:
            theta[i,0]= -(real_angles_2[3+(i-3)*6+0]-180)/180.0*math.pi
            theta[i,1]= (real_angles_2[3+(i-3)*6+1]-180)/180.0*math.pi
            theta[i,2]= (-real_angles_2[3+(i-3)*6+2]+62.69)/180.0*math.pi
            
           
        
    return theta

def sim_angles_to_real(theta):
    # theta [6,3]  输出 np array [18]
    real_angles=np.zeros_like(theta).flatten()*1.00000
    for i in range(6):
        if i<3:
            
            hip=(180-theta[i,0]/math.pi*180) #向前为增加
            
            knee=180-theta[i,1]/math.pi*180
            ankle=62.69+theta[i,2]/math.pi*180
            
            real_angles[0+i*6+0]=hip
            real_angles[0+i*6+1]=knee
            real_angles[0+i*6+2]=ankle
            
        else:
            hip=(180-theta[i,0]/math.pi*180) #向前为增加
            knee=180+theta[i,1]/math.pi*180
            ankle=62.69-theta[i,2]/math.pi*180
            
            real_angles[3+(i-3)*6+0]=hip
            real_angles[3+(i-3)*6+1]=knee
            real_angles[3+(i-3)*6+2]=ankle
    for i in range(18):
        real_angles[i]=real_angles[i]/180*2048
        
    return real_angles

def set_force(tau):
    1
def set_positions_multi(theta,robot):
    # theta (6,3)
    theta_list=theta.flatten().tolist()
    joint_Indice=range(18)
    p.setJointMotorControlArray(bodyUniqueId=robot, 
                jointIndices=joint_Indice, 
                controlMode=p.POSITION_CONTROL,
                targetPositions=theta_list,
                forces = [4.1]*18)



def get_current_pos_and_goal(row):
    current_pos=[]
    goal_pos=[]
    for i in range(18):
        current_pos.append(eval(row[6*i+1]))
        goal_pos.append(eval(row[6*i+2]))
    return current_pos,goal_pos




def actuator_control_pd_force_multi_support_swing(P_gain,D_gain,friction):
    #p.connect(p.DIRECT )  # 连接到仿真服务器
    p.connect(p.GUI )  # 连接到仿真服务器
    p.setGravity(0, 0, -9.8)  # 设置重力值
    p.setAdditionalSearchPath(pd.getDataPath())  # 设置pybullet_data的文件路径

    # 加载地面
    floor = p.loadURDF("plane.urdf")
    floors=[]
    floors.append(floor)

    # mini cheetah的初始位置
    startPos = [0, 0, 0.1]
    startOri=__rpy2quaternion(math.pi/2,0,0)
    # read from csv

    #21.72978

    # 加载urdf文件
    robot = p.loadURDF("hexapod_34/urdf/hexapod_34.urdf", startPos,)


    dt=1/1000
    p.setTimeStep(dt)
    Force = 4.1
    
    p.stepSimulation()
    
    '''
    p.setJointMotorControl2(bodyUniqueId=robot, 
    jointIndex=0, 
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity = targetVel,
    force = Force)
    '''
    #21.129780689865218
    step=0
    error_sum=0
    DynamicsInfo=p.getDynamicsInfo(robot,0)
    p.changeDynamics(robot,0,lateralFriction=10)
    explicitPD=PDControllerExplicit(p)
    stablePD=PDControllerStable(p)
    explicitPD_mul=PDControllerExplicitMultiDof(p)
    step=0

    with open('/home/xu-ye/Downloads/dynamixel_workbench_master-main(1)/dynamixel_workbench_master-main/dynamixel_workbench_toolbox/test/build/servo_all_fre_cpg_3.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        skip_flag=0
        for row in csv_reader:
            
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif line_count == 1 or line_count%1000==0:
                real_position=get_obserations_multi(robot)
                real_position_tick_first=sim_angles_to_real(real_position)
                current_pos,goal_pos=get_current_pos_and_goal(row)
                current_pos_sim=real_angles_to_sim(current_pos)
                goal_pos_sim=real_angles_to_sim(goal_pos)
                set_positions_multi(current_pos_sim,robot)
                for i in range(300):
                    p.stepSimulation()
                real_position=get_obserations_multi(robot)
                real_position_tick=sim_angles_to_real(real_position)
                #print("position now",real_position_tick)
                line_count += 1
                first_one=1
            else:
                if eval(row[1])>4096:
                    skip_flag=1
                    continue

                if skip_flag==1:
                    current_pos,goal_pos=get_current_pos_and_goal(row)
                    current_pos_sim=real_angles_to_sim(current_pos)
                    goal_pos_sim=real_angles_to_sim(goal_pos)
                    set_positions_multi(current_pos_sim,robot)
                    for i in range(300):
                        p.stepSimulation()
                    skip_flag=0
                    first_one=1

                    
                sim_position=get_obserations_multi(robot)
                sim_position_tick=sim_angles_to_real(sim_position)
                current_pos,goal_pos=get_current_pos_and_goal(row)
                goal_pos_sim=real_angles_to_sim(goal_pos).flatten()
                sim_real_error=current_pos-sim_position_tick
                print("sim_real_error",sim_real_error.reshape((6,3)))
                print("goal position",(np.array(goal_pos)).reshape((6,3)),)
                
                print("real position",(np.array(current_pos)).reshape((6,3)))
                print("position now",sim_position_tick.reshape((6,3)),)
                error_sum+=abs(sim_real_error)
                step+=1



                contacts,lateral=get_contactPoints(robot,floors)
                support_leg=contacts>0
                Pgain_support=5.843*0.9
                Dgain_support=0.1316
                Pgain_swing=P_gain
                Dgain_swing=D_gain
                Kp=np.ones((6,3))*Pgain_swing
                Kd=np.ones((6,3))*Dgain_swing
                Kp[support_leg]=np.ones(3)*Pgain_support
                Kd[support_leg]=np.ones(3)*Dgain_support
                Kp=Kp.flatten()
                Kd=Kd.flatten()
                
                #tau=explicitPD_mul.computePD(robot,range(18),goal_pos_sim,[0]*18,P_gain,D_gain,[Force]*18,dt)
                tau=explicitPD.computePD(robot,range(18),goal_pos_sim,[0]*18,Kp,Kd,[Force]*18,dt,first_one)
                #tau=stablePD.computePD(robot,range(18),goal_pos_sim,[0]*18,[P_gain]*18,[D_gain]*18,[Force]*18,dt)
                '''
                tau_mul=[]
                for i_index in range(18):
                    tau=explicitPD.computePD(robot,[i_index],goal_pos_sim[i_index],0,P_gain,D_gain,Force,dt,first_one)
                    tau_mul.append(tau)
                '''

                first_one=0
                real_tau=1.82*2.68*eval(row[3])/1000

                #print("cal tau",tau,)
                p.setJointMotorControlArray(robot, range(18), p.VELOCITY_CONTROL, targetVelocities=[0]*18, forces=[friction]*18)
                # 直接应用real tau 完全没有按照既定轨迹运行
                p.setJointMotorControlArray(bodyUniqueId=robot,jointIndices=range(18),controlMode=p.TORQUE_CONTROL,
                forces = tau)
                p.stepSimulation()



                # 第二个stick

                contacts,lateral=get_contactPoints(robot,floors)
                support_leg=contacts>0
                
                Pgain_support=P_gain
                Dgain_support=D_gain
                Pgain_swing=5.843
                Dgain_swing=0.1316
                Kp=np.ones((6,3))*Pgain_swing
                Kd=np.ones((6,3))*Dgain_swing
                Kp[support_leg]=np.ones(3)*Pgain_support
                Kd[support_leg]=np.ones(3)*Dgain_support
                Kp=Kp.flatten()
                Kd=Kd.flatten()


                tau=explicitPD.computePD(robot,range(18),goal_pos_sim,[0]*18,Kp,Kd,[Force]*18,dt,first_one)
                #tau=stablePD.computePD(robot,range(18),goal_pos_sim,[0]*18,[P_gain]*18,[D_gain]*18,[Force]*18,dt)
                p.setJointMotorControlArray(robot, range(18), p.VELOCITY_CONTROL, targetVelocities=[0]*18, forces=[friction]*18)
                # 直接应用real tau 完全没有按照既定轨迹运行
                p.setJointMotorControlArray(bodyUniqueId=robot,jointIndices=range(18),controlMode=p.TORQUE_CONTROL,
                forces = tau)
                p.stepSimulation()

                #sim_position=get_obseration(robot)
                #sim_position_tick=sim_position/math.pi*2048+2048
                #print("goal position",eval(row[2]),"position now",sim_position_tick)



                line_count += 1
    with open('/home/xu-ye/Downloads/dynamixel_workbench_master-main(1)/dynamixel_workbench_master-main/dynamixel_workbench_toolbox/test/build/servo_all_fre_cpg_2.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        skip_flag=0
        for row in csv_reader:
            
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif line_count == 1 or line_count%1000==0:
                real_position=get_obserations_multi(robot)
                real_position_tick_first=sim_angles_to_real(real_position)
                current_pos,goal_pos=get_current_pos_and_goal(row)
                current_pos_sim=real_angles_to_sim(current_pos)
                goal_pos_sim=real_angles_to_sim(goal_pos)
                set_positions_multi(current_pos_sim,robot)
                for i in range(300):
                    p.stepSimulation()
                real_position=get_obserations_multi(robot)
                real_position_tick=sim_angles_to_real(real_position)
                #print("position now",real_position_tick)
                line_count += 1
                first_one=1
            else:
                if eval(row[1])>4096:
                    skip_flag=1
                    continue

                if skip_flag==1:
                    current_pos,goal_pos=get_current_pos_and_goal(row)
                    current_pos_sim=real_angles_to_sim(current_pos)
                    goal_pos_sim=real_angles_to_sim(goal_pos)
                    set_positions_multi(current_pos_sim,robot)
                    for i in range(300):
                        p.stepSimulation()
                    skip_flag=0
                    first_one=1

                    
                sim_position=get_obserations_multi(robot)
                sim_position_tick=sim_angles_to_real(sim_position)
                current_pos,goal_pos=get_current_pos_and_goal(row)
                goal_pos_sim=real_angles_to_sim(goal_pos).flatten()
                sim_real_error=current_pos-sim_position_tick
                print("sim_real_error",sim_real_error.reshape((6,3)))
                print("goal position",(np.array(goal_pos)).reshape((6,3)),)
                
                print("real position",(np.array(current_pos)).reshape((6,3)))
                print("position now",sim_position_tick.reshape((6,3)),)
                error_sum+=abs(sim_real_error)
                step+=1



                contacts,lateral=get_contactPoints(robot,floors)
                support_leg=contacts>0
                Pgain_support=5.843*0.9
                Dgain_support=0.1316
                Pgain_swing=P_gain
                Dgain_swing=D_gain
                Kp=np.ones((6,3))*Pgain_swing
                Kd=np.ones((6,3))*Dgain_swing
                Kp_support=np.array([P_gain,33.36,33.36])
                Kd_support=np.array([D_gain,0.6841,0.6841])
                #Kp[support_leg]=np.ones(3)*Pgain_support
                #Kd[support_leg]=np.ones(3)*Dgain_support
                Kp[support_leg]=Kp_support
                Kd[support_leg]=Kd_support
                Kp=Kp.flatten()
                Kd=Kd.flatten()
                
                #tau=explicitPD_mul.computePD(robot,range(18),goal_pos_sim,[0]*18,P_gain,D_gain,[Force]*18,dt)
                tau=explicitPD.computePD(robot,range(18),goal_pos_sim,[0]*18,Kp,Kd,[Force]*18,dt,first_one)
                #tau=stablePD.computePD(robot,range(18),goal_pos_sim,[0]*18,[P_gain]*18,[D_gain]*18,[Force]*18,dt)
                '''
                tau_mul=[]
                for i_index in range(18):
                    tau=explicitPD.computePD(robot,[i_index],goal_pos_sim[i_index],0,P_gain,D_gain,Force,dt,first_one)
                    tau_mul.append(tau)
                '''

                first_one=0
                real_tau=1.82*2.68*eval(row[3])/1000

                #print("cal tau",tau,)
                p.setJointMotorControlArray(robot, range(18), p.VELOCITY_CONTROL, targetVelocities=[0]*18, forces=[friction]*18)
                # 直接应用real tau 完全没有按照既定轨迹运行
                p.setJointMotorControlArray(bodyUniqueId=robot,jointIndices=range(18),controlMode=p.TORQUE_CONTROL,
                forces = tau)
                p.stepSimulation()



                # 第二个stick

                contacts=get_contactPoints(robot,floors)
                support_leg=contacts>0
                
                Pgain_support=P_gain
                Dgain_support=D_gain
                Pgain_swing=5.843
                Dgain_swing=0.1316
                Kp=np.ones((6,3))*Pgain_swing
                Kd=np.ones((6,3))*Dgain_swing
                Kp[support_leg]=np.ones(3)*Pgain_support
                Kd[support_leg]=np.ones(3)*Dgain_support
                Kp=Kp.flatten()
                Kd=Kd.flatten()


                tau=explicitPD.computePD(robot,range(18),goal_pos_sim,[0]*18,Kp,Kd,[Force]*18,dt,first_one)
                #tau=stablePD.computePD(robot,range(18),goal_pos_sim,[0]*18,[P_gain]*18,[D_gain]*18,[Force]*18,dt)
                p.setJointMotorControlArray(robot, range(18), p.VELOCITY_CONTROL, targetVelocities=[0]*18, forces=[friction]*18)
                # 直接应用real tau 完全没有按照既定轨迹运行
                p.setJointMotorControlArray(bodyUniqueId=robot,jointIndices=range(18),controlMode=p.TORQUE_CONTROL,
                forces = tau)
                p.stepSimulation()

                #sim_position=get_obseration(robot)
                #sim_position_tick=sim_position/math.pi*2048+2048
                #print("goal position",eval(row[2]),"position now",sim_position_tick)



                line_count += 1
            
    average_sum=error_sum/step
    print("average_sum",average_sum)
    p.disconnect()
    return -sum(average_sum)




def actuator_control_pd_force_multi_obstacle(P_gain,D_gain,friction):
    #p.connect(p.DIRECT )  # 连接到仿真服务器
    p.connect(p.GUI )  # 连接到仿真服务器
    p.setGravity(0, 0, -9.8)  # 设置重力值
    p.setAdditionalSearchPath(pd.getDataPath())  # 设置pybullet_data的文件路径

    # 加载地面
    floor = p.loadURDF("plane.urdf")
    floors=[]
    floors.append(floor)

    # mini cheetah的初始位置
    startPos = [0, 0, 0.1]
    startOri=__rpy2quaternion(math.pi/2,0,0)
    # read from csv

    #21.72978

    # 加载urdf文件
    robot = p.loadURDF("hexapod_34/urdf/hexapod_34.urdf", startPos,)


    # 加载障碍物

    length=0.3
    delta=-0.014064173723102645-(-0.02594413848924254)+0.001
    base_position=[-0.21,0.155+delta,0.015]
    hal_geo=[0.05, 0.05, 0.015]

    
    colBoxId = p.createCollisionShape(p.GEOM_BOX,halfExtents=hal_geo)
    visualID=p.createVisualShape(p.GEOM_BOX,halfExtents=hal_geo)                                  
    box=p.createMultiBody(0, colBoxId,visualID,base_position)
    floors.append(box)    
    length=0.3
    base_position=[-0.21,0.145+delta,0.041]
    hal_geo=[0.05, 0.04, 0.011]
    colBoxId = p.createCollisionShape(p.GEOM_BOX,halfExtents=hal_geo)
    visualID=p.createVisualShape(p.GEOM_BOX,halfExtents=hal_geo)                                  
    box2=p.createMultiBody(0, colBoxId,visualID,base_position)
    floors.append(box2)    
    
    


    dt=1/1000
    p.setTimeStep(dt)
    Force = 4.1
    
    p.stepSimulation()
    
    '''
    p.setJointMotorControl2(bodyUniqueId=robot, 
    jointIndex=0, 
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity = targetVel,
    force = Force)
    '''
    #21.129780689865218
    step=0
    error_sum=0
    DynamicsInfo=p.getDynamicsInfo(robot,0)
    p.changeDynamics(robot,0,lateralFriction=0.1)
    p.changeDynamics(floor,0,lateralFriction=0.1)
    explicitPD=PDControllerExplicit(p)
    stablePD=PDControllerStable(p)
    explicitPD_mul=PDControllerExplicitMultiDof(p)
    step=0

    with open('/home/xu-ye/Downloads/dynamixel_workbench_master-main(1)/dynamixel_workbench_master-main/dynamixel_workbench_toolbox/test/build/servo_all_fre_cpg_obstable_16_10_5_1.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        skip_flag=0
        for row in csv_reader:
            
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif line_count == 1 or line_count%1000==0:
                real_position=get_obserations_multi(robot)
                real_position_tick_first=sim_angles_to_real(real_position)
                current_pos,goal_pos=get_current_pos_and_goal(row)
                current_pos_sim=real_angles_to_sim(current_pos)
                goal_pos_sim=real_angles_to_sim(goal_pos)
                set_positions_multi(current_pos_sim,robot)
                for i in range(300):
                    p.stepSimulation()
                real_position=get_obserations_multi(robot)
                real_position_tick=sim_angles_to_real(real_position)
                #print("position now",real_position_tick)
                line_count += 1
                first_one=1
            else:
                if eval(row[1])>4096:
                    skip_flag=1
                    continue

                if skip_flag==1:
                    current_pos,goal_pos=get_current_pos_and_goal(row)
                    current_pos_sim=real_angles_to_sim(current_pos)
                    goal_pos_sim=real_angles_to_sim(goal_pos)
                    set_positions_multi(current_pos_sim,robot)
                    for i in range(300):
                        p.stepSimulation()
                    skip_flag=0
                    first_one=1

                    
                sim_position=get_obserations_multi(robot)
                sim_position_tick=sim_angles_to_real(sim_position)
                current_pos,goal_pos=get_current_pos_and_goal(row)
                goal_pos_sim=real_angles_to_sim(goal_pos).flatten()
                sim_real_error=current_pos-sim_position_tick
                print("sim_real_error",sim_real_error.reshape((6,3)))
                print("goal position",(np.array(goal_pos)).reshape((6,3)),)
                
                print("real position",(np.array(current_pos)).reshape((6,3)))
                print("position now",sim_position_tick.reshape((6,3)),)
                error_sum+=abs(sim_real_error)
                step+=1
                basePos, baseOri = p.getBasePositionAndOrientation(robot,)
                (x, y, z, w) = baseOri
                #(0.001106937767314761, -0.014064173723102645, 0.1084525570062837)
                #(0.0014899830066186857, -0.02594413848924254, 0.10858217092560875)  with obs
                contacts,lateral=get_contactPoints(robot,floors)
                support_leg=contacts>0
                Pgain_support=5.843*0.9
                Dgain_support=0.1316
                Pgain_swing=P_gain
                Dgain_swing=D_gain
                Kp=np.ones((6,3))*Pgain_swing
                Kd=np.ones((6,3))*Dgain_swing
                Kp[support_leg]=np.ones(3)*Pgain_support
                Kd[support_leg]=np.ones(3)*Dgain_support
                Kp=Kp.flatten()
                Kd=Kd.flatten()
                
                #tau=explicitPD_mul.computePD(robot,range(18),goal_pos_sim,[0]*18,P_gain,D_gain,[Force]*18,dt)
                tau=explicitPD.computePD(robot,range(18),goal_pos_sim,[0]*18,Kp,Kd,[Force]*18,dt,first_one)
                #tau=stablePD.computePD(robot,range(18),goal_pos_sim,[0]*18,[P_gain]*18,[D_gain]*18,[Force]*18,dt)
                '''
                tau_mul=[]
                for i_index in range(18):
                    tau=explicitPD.computePD(robot,[i_index],goal_pos_sim[i_index],0,P_gain,D_gain,Force,dt,first_one)
                    tau_mul.append(tau)
                '''

                first_one=0
                real_tau=1.82*2.68*eval(row[3])/1000

                print("cal tau",tau,)
                p.setJointMotorControlArray(robot, range(18), p.VELOCITY_CONTROL, targetVelocities=[0]*18, forces=[friction]*18)
                # 直接应用real tau 完全没有按照既定轨迹运行
                p.setJointMotorControlArray(bodyUniqueId=robot,jointIndices=range(18),controlMode=p.TORQUE_CONTROL,
                forces = tau)
                p.stepSimulation()



                # 第二个stick

                contacts,lateral=get_contactPoints(robot,floors)
                support_leg=contacts>0
                
                Pgain_support=P_gain
                Dgain_support=D_gain
                Pgain_swing=5.843
                Dgain_swing=0.1316
                Kp=np.ones((6,3))*Pgain_swing
                Kd=np.ones((6,3))*Dgain_swing
                Kp[support_leg]=np.ones(3)*Pgain_support
                Kd[support_leg]=np.ones(3)*Dgain_support
                Kp=Kp.flatten()
                Kd=Kd.flatten()


                tau=explicitPD.computePD(robot,range(18),goal_pos_sim,[0]*18,Kp,Kd,[Force]*18,dt,first_one)
                #tau=stablePD.computePD(robot,range(18),goal_pos_sim,[0]*18,[P_gain]*18,[D_gain]*18,[Force]*18,dt)
                p.setJointMotorControlArray(robot, range(18), p.VELOCITY_CONTROL, targetVelocities=[0]*18, forces=[friction]*18)
                # 直接应用real tau 完全没有按照既定轨迹运行
                p.setJointMotorControlArray(bodyUniqueId=robot,jointIndices=range(18),controlMode=p.TORQUE_CONTROL,
                forces = tau)
                p.stepSimulation()

                #sim_position=get_obseration(robot)
                #sim_position_tick=sim_position/math.pi*2048+2048
                #print("goal position",eval(row[2]),"position now",sim_position_tick)



                line_count += 1
    
            
    average_sum=error_sum/step
    print("average_sum",average_sum)
    p.disconnect()
    return -sum(average_sum)




def actuator_control_pd_force_2(P_gain,D_gain,friction):
    #p.connect(p.DIRECT )  # 连接到仿真服务器
    p.connect(p.GUI )  # 连接到仿真服务器
    p.setGravity(0, 0, -9.8)  # 设置重力值
    p.setAdditionalSearchPath(pd.getDataPath())  # 设置pybullet_data的文件路径

    # 加载地面
    floor = p.loadURDF("plane.urdf")

    # mini cheetah的初始位置
    startPos = [0, 0, 0.15]
    startOri=__rpy2quaternion(math.pi/2,0,0)
    # read from csv

    #21.72978

    # 加载urdf文件
    #robot = p.loadURDF("actuators/actuators.urdf", startPos,startOri,useFixedBase=True)
    robot =p.loadURDF("hexapod_34/urdf/hexapod_34.urdf", startPos,useFixedBase=True)


    dt=1/1000
    p.setTimeStep(dt)
    Force = 4.1
    
    p.stepSimulation()
    
    '''
    p.setJointMotorControl2(bodyUniqueId=robot, 
    jointIndex=0, 
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity = targetVel,
    force = Force)
    '''
    #21.129780689865218
    step=0
    error_sum=0
    DynamicsInfo=p.getDynamicsInfo(robot,0)
    p.changeDynamics(robot,0,lateralFriction=10)
    explicitPD=PDControllerExplicit(p)
    stablePD=PDControllerStable(p)
    step=0

    p.setJointMotorControl2(bodyUniqueId=robot, 
                jointIndex=2, 
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.5,
                force = Force)
    p.setJointMotorControl2(bodyUniqueId=robot, 
                jointIndex=0, 
                controlMode=p.POSITION_CONTROL,
                targetPosition=0,
                force = Force)
    for i in range(300):
        p.stepSimulation()

    with open('/home/xu-ye/Downloads/record-main/servo_all_fre_10_2.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        skip_flag=0
        for row in csv_reader:
            
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif line_count == 1 or line_count%10000==0:
                current_position=-(eval(row[1])-2048)/2048*math.pi
                #print("initial position",eval(row[1]))
                p.setJointMotorControl2(bodyUniqueId=robot, 
                jointIndex=1, 
                controlMode=p.POSITION_CONTROL,
                targetPosition=current_position,
                force = Force)
                for i in range(300):
                    p.stepSimulation()
                real_position=get_obseration(robot)
                real_position_tick=-real_position/math.pi*2048+2048
                #print("position now",real_position_tick)
                line_count += 1
                first_one=1
            else:
                if eval(row[1])>4096:
                    skip_flag=1
                    continue
                if line_count%2==1:
                    line_count += 1
                    continue
                

                if skip_flag==1:
                    current_position=-(eval(row[1])-2048)/2048*math.pi
                #print("initial position",eval(row[1]))
                    p.setJointMotorControl2(bodyUniqueId=robot, 
                    jointIndex=1, 
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=current_position,
                    force = Force)
                    for i in range(300):
                        p.stepSimulation()
                    real_position=get_obseration(robot)
                    real_position_tick=-real_position/math.pi*2048+2048
                    skip_flag=0
                    first_one=1

                    
                sim_position=get_obseration(robot)
                sim_position_tick=-sim_position/math.pi*2048+2048
                real_position_tick=eval(row[1])
                sim_real_error=real_position_tick-sim_position_tick
                print("sim_real_error",sim_real_error)
                print("goal position",eval(row[2]),"position now",sim_position_tick,"real position",real_position_tick)
                error_sum+=abs(sim_real_error)
                goal_position=-(eval(row[2])-2048)/2048*math.pi

                tau=explicitPD.computePD_single(robot,[1],goal_position,0,P_gain,D_gain,Force,dt,first_one)
                first_one=0
                real_tau=1.82*2.68*eval(row[3])/1000

                print("cal tau",tau,"    real tau",real_tau)
                p.setJointMotorControl2(robot, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=friction)
                # 直接应用real tau 完全没有按照既定轨迹运行
                p.setJointMotorControl2(bodyUniqueId=robot,jointIndex=1,controlMode=p.TORQUE_CONTROL,
                force = tau)
                p.stepSimulation()

                tau=explicitPD.computePD_single(robot,[1],goal_position,0,P_gain,D_gain,Force,dt,first_one)
                p.setJointMotorControl2(robot, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=friction)
                # 直接应用real tau 完全没有按照既定轨迹运行
                p.setJointMotorControl2(bodyUniqueId=robot,jointIndex=1,controlMode=p.TORQUE_CONTROL,
                force = tau)
                p.stepSimulation()
                #sim_position=get_obseration(robot)
                #sim_position_tick=sim_position/math.pi*2048+2048
                #print("goal position",eval(row[2]),"position now",sim_position_tick)



                line_count += 1
            
    average_sum=error_sum/line_count*2
    print("average_sum",average_sum)
    p.disconnect()
    return -average_sum
    

def actuator_control_pd__multi_obstacle(P_gain,D_gain,max_speed):
    #p.connect(p.DIRECT )  # 连接到仿真服务器
    p.connect(p.GUI )  # 连接到仿真服务器
    p.setGravity(0, 0, -9.8)  # 设置重力值
    p.setAdditionalSearchPath(pd.getDataPath())  # 设置pybullet_data的文件路径

    # 加载地面
    floor = p.loadURDF("plane.urdf")
    floors=[]
    floors.append(floor)

    # mini cheetah的初始位置
    startPos = [0, 0, 0.1]
    startOri=__rpy2quaternion(math.pi/2,0,0)
    # read from csv

    #21.72978

    # 加载urdf文件
    robot = p.loadURDF("hexapod_34/urdf/hexapod_34.urdf", startPos,)
    length=0.3
    delta=-0.014064173723102645-(-0.02594413848924254)+0.001+0.0003
    base_position=[-0.21,0.155+delta,0.015]
    hal_geo=[0.05, 0.05, 0.015]

    
    colBoxId = p.createCollisionShape(p.GEOM_BOX,halfExtents=hal_geo)
    visualID=p.createVisualShape(p.GEOM_BOX,halfExtents=hal_geo)                                  
    box=p.createMultiBody(2, colBoxId,visualID,base_position)
    floors.append(box)    
    length=0.3
    base_position=[-0.21,0.145+delta,0.041]
    hal_geo=[0.05, 0.04, 0.011]
    colBoxId = p.createCollisionShape(p.GEOM_BOX,halfExtents=hal_geo)
    visualID=p.createVisualShape(p.GEOM_BOX,halfExtents=hal_geo)                                  
    #box2=p.createMultiBody(2, colBoxId,visualID,base_position)
    #floors.append(box2)  


    dt=1/1000
    p.setTimeStep(dt)
    Force = 4.1
    
    p.stepSimulation()
    
    '''
    p.setJointMotorControl2(bodyUniqueId=robot, 
    jointIndex=0, 
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity = targetVel,
    force = Force)
    '''
    #21.129780689865218
    step=0
    error_sum=0
    DynamicsInfo=p.getDynamicsInfo(robot,0)
    p.changeDynamics(robot,0,lateralFriction=0.8)
    explicitPD=PDControllerExplicit(p)
    stablePD=PDControllerStable(p)
    explicitPD_mul=PDControllerExplicitMultiDof(p)
    step=0

    with open('/home/xu-ye/Downloads/dynamixel_workbench_master-main(1)/dynamixel_workbench_master-main/dynamixel_workbench_toolbox/test/build/servo_all_fre_cpg_obstable_16_10_5_0_5.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        skip_flag=0
        for row in csv_reader:
            
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif line_count == 1 or line_count%1000==0:
                real_position=get_obserations_multi(robot)
                real_position_tick_first=sim_angles_to_real(real_position)
                current_pos,goal_pos=get_current_pos_and_goal(row)
                current_pos_sim=real_angles_to_sim(current_pos)
                goal_pos_sim=real_angles_to_sim(goal_pos)
                set_positions_multi(current_pos_sim,robot)
                for i in range(300):
                    p.stepSimulation()
                real_position=get_obserations_multi(robot)
                real_position_tick=sim_angles_to_real(real_position)
                #print("position now",real_position_tick)
                line_count += 1
                first_one=1
            else:
                if eval(row[1])>4096:
                    skip_flag=1
                    continue

                if skip_flag==1:
                    current_pos,goal_pos=get_current_pos_and_goal(row)
                    current_pos_sim=real_angles_to_sim(current_pos)
                    goal_pos_sim=real_angles_to_sim(goal_pos)
                    set_positions_multi(current_pos_sim,robot)
                    for i in range(300):
                        p.stepSimulation()
                    skip_flag=0
                    first_one=1

                    
                sim_position=get_obserations_multi(robot)
                sim_position_tick=sim_angles_to_real(sim_position)
                current_pos,goal_pos=get_current_pos_and_goal(row)
                goal_pos_sim=real_angles_to_sim(goal_pos).flatten()
                sim_real_error=current_pos-sim_position_tick
                print("sim_real_error",sim_real_error.reshape((6,3)))
                print("goal position",(np.array(goal_pos)).reshape((6,3)),)
                
                print("real position",(np.array(current_pos)).reshape((6,3)))
                print("position now",sim_position_tick.reshape((6,3)),)
                error_sum+=abs(sim_real_error)
                contacts,lateral=get_contactPoints(robot,floors)
                print("contacts",contacts)
                
                
                frictionForce=0.9
                p.setJointMotorControlArray(robot, range(18), p.VELOCITY_CONTROL, targetVelocities=[0]*18, forces=[frictionForce]*18)
                # 直接应用real tau 完全没有按照既定轨迹运行
                
                for actuator_index in range(18):
                    p.setJointMotorControl2(bodyUniqueId=robot,jointIndex=actuator_index,controlMode=p.POSITION_CONTROL,targetPosition=goal_pos_sim[actuator_index],
                    positionGain=P_gain,
                    velocityGain=D_gain,
                    maxVelocity=max_speed,
                    force = Force)    

                p.stepSimulation()

                # 第二个stick
                
                p.setJointMotorControlArray(robot, range(18), p.VELOCITY_CONTROL, targetVelocities=[0]*18, forces=[frictionForce]*18)
                # 直接应用real tau 完全没有按照既定轨迹运行
                for actuator_index in range(18):
                    p.setJointMotorControl2(bodyUniqueId=robot,jointIndex=actuator_index,controlMode=p.POSITION_CONTROL,targetPosition=goal_pos_sim[actuator_index],
                    positionGain=P_gain,
                    velocityGain=D_gain,
                    maxVelocity=max_speed,
                    force = Force)    

                p.stepSimulation()

                #sim_position=get_obseration(robot)
                #sim_position_tick=sim_position/math.pi*2048+2048
                #print("goal position",eval(row[2]),"position now",sim_position_tick)



                line_count += 1
            
    average_sum=error_sum/line_count
    print("average_sum",average_sum)
    p.disconnect()
    return -average_sum

def __rpy2quaternion(roll, pitch, yaw):
        x=math.sin(pitch/2)*math.sin(yaw/2)*math.cos(roll/2)+math.cos(pitch/2)*math.cos(yaw/2)*math.sin(roll/2)
        y=math.sin(pitch/2)**math.cos(yaw/2)*math.cos(roll/2)+math.cos(pitch/2)*math.sin(yaw/2)*math.sin(roll/2)
        z=math.cos(pitch/2)*math.sin(yaw/2)*math.cos(roll/2)-math.sin(pitch/2)*math.cos(yaw/2)*math.sin(roll/2)
        w=math.cos(pitch/2)*math.cos(yaw/2)*math.cos(roll/2)-math.sin(pitch/2)*math.sin(yaw/2)*math.sin(roll/2)
        ori=[x,y,z,w]
        return ori

def actuator_control_pd_10ms(P_gain,D_gain,max_speed):
    p.connect(p.DIRECT )  # 连接到仿真服务器
    #p.connect(p.GUI )  # 连接到仿真服务器
    p.setGravity(0, 0, -9.8)  # 设置重力值
    p.setAdditionalSearchPath(pd.getDataPath())  # 设置pybullet_data的文件路径

    # 加载地面
    floor = p.loadURDF("plane.urdf")
    floors=[]
    floors.append(floor)

    # mini cheetah的初始位置
    startPos = [0, 0, 0.5]
    startOri=__rpy2quaternion(math.pi/3,0,0)
    # read from csv
    
    #21.72978

    # 加载urdf文件
    robot = p.loadURDF("hexapod_34/urdf/hexapod_34.urdf", startPos,startOri,useFixedBase=True,)
    length=0.3
    delta=-0.014064173723102645-(-0.02594413848924254)+0.001+0.0003
    base_position=[-0.21,0.155+delta,0.015]
    hal_geo=[0.05, 0.05, 0.015]
    
    basePos, baseOri = p.getBasePositionAndOrientation(robot)
    (x, y, z, w) = baseOri
    # print(basePos)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    IMU = np.array([roll, pitch, yaw])
    print("roll pitch yaw!",roll,pitch,yaw)

    
    colBoxId = p.createCollisionShape(p.GEOM_BOX,halfExtents=hal_geo)
    visualID=p.createVisualShape(p.GEOM_BOX,halfExtents=hal_geo)                                  
    box=p.createMultiBody(2, colBoxId,visualID,base_position)
    floors.append(box)    
    length=0.3
    base_position=[-0.21,0.145+delta,0.041]
    hal_geo=[0.05, 0.04, 0.011]
    colBoxId = p.createCollisionShape(p.GEOM_BOX,halfExtents=hal_geo)
    visualID=p.createVisualShape(p.GEOM_BOX,halfExtents=hal_geo)                                  
    #box2=p.createMultiBody(2, colBoxId,visualID,base_position)
    #floors.append(box2)  


    dt=1/1000
    p.setTimeStep(dt)
    Force = 4.1
    
    p.stepSimulation()
    
    '''
    p.setJointMotorControl2(bodyUniqueId=robot, 
    jointIndex=0, 
    controlMode=p.VELOCITY_CONTROL,
    targetVelocity = targetVel,
    force = Force)
    '''
    #21.129780689865218
    step=0
    error_sum=0
    DynamicsInfo=p.getDynamicsInfo(robot,0)
    p.changeDynamics(robot,0,lateralFriction=0.8)
    explicitPD=PDControllerExplicit(p)
    stablePD=PDControllerStable(p)
    explicitPD_mul=PDControllerExplicitMultiDof(p)
    step=0
    
    with open('/home/fast3/Desktop/DynamixelSDK-3.7.31/python/tests/protocol2_0/data_reflex_imu_flat_20.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        skip_flag=0
        for row in csv_reader:
            
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif line_count == 1 or line_count%1000==0:
                real_position=get_obserations_multi(robot)
                real_position_tick_first=sim_angles_to_real(real_position)
                count,T_count,sum_leg,reflex_sim,reflex_stance_sim,on_reflex2,on_reflex_stance2,\
    reflex_index2,reflex_index_stance2,swing_step_count,goal_pos,current_pos,IMU_data=eval_row2(row)
                #current_pos,goal_pos=get_current_pos_and_goal(row)
                current_pos_sim=real_angles_to_sim(current_pos)
                goal_pos_sim=real_angles_to_sim(goal_pos)
                set_positions_multi(current_pos_sim,robot)
                for i in range(300):
                    p.stepSimulation()
                real_position=get_obserations_multi(robot)
                real_position_tick=sim_angles_to_real(real_position)
                #print("position now",real_position_tick)
                line_count += 1
                first_one=1
            else:
                if eval(row[1])>4096:
                    skip_flag=1
                    continue

                if skip_flag==1:
                    count,T_count,sum_leg,reflex_sim,reflex_stance_sim,on_reflex2,on_reflex_stance2,\
    reflex_index2,reflex_index_stance2,swing_step_count,goal_pos,current_pos,IMU_data=eval_row2(row)
                    current_pos_sim=real_angles_to_sim(current_pos)
                    goal_pos_sim=real_angles_to_sim(goal_pos)
                    set_positions_multi(current_pos_sim,robot)
                    for i in range(300):
                        p.stepSimulation()
                    skip_flag=0
                    first_one=1

                    
                sim_position=get_obserations_multi(robot)
                sim_position_tick=sim_angles_to_real(sim_position)
                count,T_count,sum_leg,reflex_sim,reflex_stance_sim,on_reflex2,on_reflex_stance2,\
    reflex_index2,reflex_index_stance2,swing_step_count,goal_pos,current_pos,IMU_data=eval_row2(row)
                goal_pos_sim=real_angles_to_sim(goal_pos).flatten()
                sim_real_error=current_pos-sim_position_tick
                #print("sim_real_error",sim_real_error.reshape((6,3)))
                #print("goal position",(np.array(goal_pos)).reshape((6,3)),)
                
                #print("real position",(np.array(current_pos)).reshape((6,3)))
                #print("position now",sim_position_tick.reshape((6,3)),)
                error_sum+=abs(sim_real_error)
                contacts,lateral=get_contactPoints(robot,floors)
                #print("contacts",contacts)
                
                start_time=time.time()
                #for actuator_index in range(18):
                #    p.resetJointState(robot,actuator_index,1)
                #sim_position2=get_obserations_multi(robot)
                #print("sim position after reset",sim_position2)
                frictionForce=0.9
                for tick_num in range(20):
                    p.setJointMotorControlArray(robot, range(18), p.VELOCITY_CONTROL, targetVelocities=[0]*18, forces=[frictionForce]*18)
                    # 直接应用real tau 完全没有按照既定轨迹运行
                    
                    for actuator_index in range(18):
                        p.setJointMotorControl2(bodyUniqueId=robot,jointIndex=actuator_index,controlMode=p.POSITION_CONTROL,targetPosition=goal_pos_sim[actuator_index],
                        positionGain=P_gain,
                        velocityGain=D_gain,
                        maxVelocity=max_speed,
                        force = Force)    

                    p.stepSimulation()
                end_time=time.time()
                print("last time",(end_time-start_time)*1000)
                

                
                #sim_position=get_obseration(robot)
                #sim_position_tick=sim_position/math.pi*2048+2048
                #print("goal position",eval(row[2]),"position now",sim_position_tick)



                line_count += 1
            
    average_sum=error_sum/line_count
    print("average_sum",average_sum)
    p.disconnect()
    return -average_sum