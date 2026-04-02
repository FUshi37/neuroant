
import argparse
import json
from sys import path
path.append("../../")
import math
import numpy as np
import torch
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
import queue as _queue_std
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



def read_imu(q_imu):
    """JY901 IMU reader (runs in a child process). Pushes 9-float rows to ``q_imu``.

    Uses callback ``onUpdate`` that calls ``q_imu.put`` directly (no global queue).

    **Serial port (Linux):** set env ``IMU_SERIAL_PORT`` (e.g. ``/dev/ttyUSB0``).
    Default is ``/dev/ttyUSB0`` to match ``test_rwm_real_robot.read_imu``.
    The old hard-coded ``by-id`` path often does not exist on every machine, which
    yields an empty queue and all-zero IMU in logs.
    """
    def onUpdate(dm):
        try:
            row = np.array(
                [
                    dm.getDeviceData("angleX"),
                    dm.getDeviceData("angleY"),
                    dm.getDeviceData("angleZ"),
                    dm.getDeviceData("gyroX"),
                    dm.getDeviceData("gyroY"),
                    dm.getDeviceData("gyroZ"),
                    dm.getDeviceData("accX"),
                    dm.getDeviceData("accY"),
                    dm.getDeviceData("accZ"),
                ],
                dtype=np.float64,
            )
            q_imu.put(row)
        except Exception as e:
            print(f"[read_imu] onUpdate error: {e}")

    try:
        device = deviceModel.DeviceModel(
            "JY901",
            WitProtocolResolver(),
            JY901SDataProcessor(),
            "51_0",
        )
        if platform.system().lower() == "linux":
            port = os.environ.get("IMU_SERIAL_PORT", "/dev/ttyUSB0")
        else:
            port = os.environ.get("IMU_SERIAL_PORT", "COM39")
        device.serialConfig.portName = port
        device.serialConfig.baud = 230400
        device.ADDR = 0x50
        print(f"[read_imu] Opening IMU serial: {port}")
        device.openDevice()
        readConfig(device)
        device.dataProcessor.onVarChanged.append(onUpdate)
        print("[read_imu] IMU device opened, callback registered.")
    except Exception as e:
        print(f"[read_imu] FAILED to open IMU: {e}")
        raise

              

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
    # 将initialize角度保存到文件
    with open('./validation_outputs/initialize_angles_CPGs_sim_2_real_IMU_new.txt', 'a') as f:
        f.write(str(angles_real) + '\n')
    servos.Robot_initialize(angles_real)
    goal_theta_tick=angles_to_tick(angles_real)
    time.sleep(1)
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
        # 将angles_real写入文件
        with open('./validation_outputs/angles_real_CPGs_sim_2_real_IMU_new.txt', 'a') as f:
            f.write(str(angles_real) + '\n')
        theta_tick=angles_to_tick(angles_real)
        servos.write_all_positions(theta_tick)
        # 将write_all_positions的theta_tick写入文件
        with open('./validation_outputs/theta_tick_CPGs_sim_2_real_IMU_new.txt', 'a') as f:
            f.write(str(theta_tick) + '\n')
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
           
        velocity=servos.read_all_velocity()
        print("velocity:",velocity)
        
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
    



# ============================================================================
# System Identification Data Collection Module
# For parameter identification per Table 1 of:
#   "Sim-to-Real Transfer of Compliant Bipedal Locomotion on Torque Sensor-Less
#    Gear-Driven Humanoid"
#
# Target parameters: Kt, Rter, motor inertia, eta_fw, eta_bw,
#                    fs, fc, kv, base mass/CoM offsets
# ============================================================================

import pickle as _sysid_pickle

# XM430-W350 unit conversions
DXL_CURRENT_UNIT_MA = 2.69           # mA per raw unit
DXL_VELOCITY_UNIT_RPM = 0.229        # rev/min per raw unit
DXL_POS_DEG_PER_TICK = 360.0 / 4096.0
DXL_PWM_MAX_RAW = 885                # raw value = 100% duty cycle
DXL_VOLTAGE_UNIT_V = 0.1             # V per raw unit (addr 144)

# XM430-W350 datasheet constants for Table 1 identification
XM430_GEAR_RATIO = 353.5
XM430_STALL_TORQUE_NM = 4.1          # at 12.0V / 2.3A
XM430_KT_DATASHEET = 1.783           # Nm/A (stall_torque / stall_current)

SYSID_SAFE_POS_MIN_TICK = 200
SYSID_SAFE_POS_MAX_TICK = 3900
SYSID_SAFE_CURRENT_ABS_RAW = 1200    # ~3228 mA


def _sysid_s16(v):
    return v - 65536 if v > 32767 else v


def _sysid_s32(v):
    return v - 4294967296 if v > 2147483647 else v


def sysid_read_all_sensors(servos):
    """Single SyncRead of addr 124..145 (22 bytes per servo).

    XM430 control table layout in this range:
        124-125  Present PWM          (2B, signed)  → V_pwm proxy
        126-127  Present Current      (2B, signed)  → motor current I
        128-131  Present Velocity     (4B, signed)  → joint velocity q̇
        132-135  Present Position     (4B, unsigned) → joint position q
        136-139  Velocity Trajectory  (4B, unsigned) → internal profile
        140-143  Position Trajectory  (4B, unsigned) → internal profile
        144-145  Present Input Voltage(2B, unsigned) → V_battery

    These fields cover all real-robot observables needed for Table 1
    parameter identification (eq. 1-4 & eq. 7 in the paper).
    """
    START_ADDR = 124
    READ_LEN = 22
    gsr = GroupSyncRead(servos.portHandler, servos.packetHandler,
                        START_ADDR, READ_LEN)
    for i in servos.DXLn_ID:
        gsr.addParam(i)

    N = 18
    pos_deg   = np.zeros(N, dtype=np.float64)
    vel_rpm   = np.zeros(N, dtype=np.float64)
    cur_mA    = np.zeros(N, dtype=np.float64)
    pos_raw   = np.zeros(N, dtype=np.int32)
    vel_raw   = np.zeros(N, dtype=np.int32)
    cur_raw   = np.zeros(N, dtype=np.int32)
    pwm_raw   = np.zeros(N, dtype=np.int32)
    voltage_V = np.zeros(N, dtype=np.float64)

    rc = gsr.txRxPacket()
    if rc != COMM_SUCCESS:
        print(f"[SysID] SyncRead error: "
              f"{servos.packetHandler.getTxRxResult(rc)}")
        gsr.clearParam()
        return (pos_deg, vel_rpm, cur_mA,
                pos_raw, vel_raw, cur_raw, pwm_raw, voltage_V)

    for i in servos.DXLn_ID:
        if not gsr.isAvailable(i, START_ADDR, READ_LEN):
            continue
        pw = _sysid_s16(gsr.getData(i, 124, 2))
        c  = _sysid_s16(gsr.getData(i, 126, 2))
        v  = _sysid_s32(gsr.getData(i, 128, 4))
        p  = gsr.getData(i, 132, 4)
        vt = gsr.getData(i, 144, 2)

        pwm_raw[i]   = pw
        cur_raw[i]   = c
        vel_raw[i]   = v
        pos_raw[i]   = p
        cur_mA[i]    = c * DXL_CURRENT_UNIT_MA
        vel_rpm[i]   = v * DXL_VELOCITY_UNIT_RPM
        pos_deg[i]   = p * DXL_POS_DEG_PER_TICK
        voltage_V[i] = vt * DXL_VOLTAGE_UNIT_V

    gsr.clearParam()
    return (pos_deg, vel_rpm, cur_mA,
            pos_raw, vel_raw, cur_raw, pwm_raw, voltage_V)


def sysid_read_pid_gains(servos):
    """Read position PID gains (D:addr80, I:addr82, P:addr84) from all joints."""
    gsr = GroupSyncRead(servos.portHandler, servos.packetHandler, 80, 6)
    for i in servos.DXLn_ID:
        gsr.addParam(i)
    gains = np.zeros((18, 3), dtype=np.int32)
    if gsr.txRxPacket() == COMM_SUCCESS:
        for i in servos.DXLn_ID:
            if gsr.isAvailable(i, 80, 6):
                gains[i, 0] = gsr.getData(i, 80, 2)   # D
                gains[i, 1] = gsr.getData(i, 82, 2)   # I
                gains[i, 2] = gsr.getData(i, 84, 2)   # P
    gsr.clearParam()
    return gains


def sysid_generate_trajectory(traj_type, neutral_deg, duration, dt, **kw):
    """Generate excitation trajectory: shape (T, 18) in degrees (real servo space).

    Supported types:
        cpg        – replay existing CPG gait from JSON
        sinusoidal – single-frequency sine on selected joints
        chirp      – frequency sweep (f0→f1) on selected joints
        multi_sine – sum of multiple sines for richer spectrum
    """
    T = max(1, int(duration / dt))
    neutral = np.asarray(neutral_deg, dtype=np.float64).reshape(18)

    if traj_type == "cpg":
        tick_json = kw.get("tick_json", "pos_cpg_4_new_36_1.json")
        with open(tick_json, "r") as f:
            d = json.load(f)
        gps = np.asarray(d["goal_pos_sim"])
        traj = np.zeros((T, 18))
        for t in range(T):
            traj[t] = sim_angles_to_real(gps[t % gps.shape[0]])
        return traj

    ji = kw.get("joint_indices", list(range(18)))
    if isinstance(ji, str) and ji == "all":
        ji = list(range(18))
    traj = np.tile(neutral, (T, 1))
    ts = np.arange(T, dtype=np.float64) * dt

    if traj_type == "sinusoidal":
        f = kw.get("freq_hz", 0.5)
        a = kw.get("amplitude_deg", 15.0)
        wave = a * np.sin(2.0 * np.pi * f * ts)
        for j in ji:
            traj[:, j] += wave

    elif traj_type == "chirp":
        f0, f1 = kw.get("freq_start", 0.1), kw.get("freq_end", 3.0)
        a = kw.get("amplitude_deg", 10.0)
        phase = 2.0 * np.pi * (f0 * ts + (f1 - f0) / (2.0 * duration) * ts ** 2)
        wave = a * np.sin(phase)
        for j in ji:
            traj[:, j] += wave

    elif traj_type == "multi_sine":
        freqs = kw.get("frequencies", [0.2, 0.5, 1.0, 2.0])
        amps  = kw.get("amplitudes", [10.0, 7.0, 5.0, 3.0])
        wave = sum(a * np.sin(2.0 * np.pi * f * ts) for f, a in zip(freqs, amps))
        for j in ji:
            traj[:, j] += wave

    elif traj_type == "squat":
        freq = kw.get("freq_hz", 0.5)
        knee_amp = kw.get("amplitude_deg", 25.0)
        wave = knee_amp * (1.0 - np.cos(2.0 * np.pi * freq * ts)) / 2.0
        knee_ids = kw.get("knee_ids", [1, 4, 7, 10, 13, 16])
        hip_ids = kw.get("hip_ids", [0, 3, 6, 9, 12, 15])
        for j in knee_ids:
            traj[:, j] += wave
        for j in hip_ids:
            traj[:, j] -= wave * 0.3

    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    return traj


def load_sysid_data(path):
    """Load system identification data from .npz or .pkl file."""
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return _sysid_pickle.load(f)
    data = np.load(path, allow_pickle=True)
    result = {k: data[k] for k in data.files}
    if "metadata" in result:
        result["metadata"] = json.loads(str(result["metadata"][0]))
    return result


def _sysid_wait_imu_ready(q_imu, timeout_sec=15.0):
    """Block until at least one non-trivial IMU sample arrives (or timeout)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_sec:
        try:
            sample = q_imu.get(timeout=0.5)
            if np.max(np.abs(sample)) > 1e-3:
                return sample
        except _queue_std.Empty:
            continue
    return None


def collect_sysid_data(
    duration=30.0,
    dt=0.02,
    trajectory_type="cpg",
    tick_json="pos_cpg_4_new_36_1.json",
    output_path="sysid_data",
    freq_hz=0.5,
    amplitude_deg=15.0,
    freq_start=0.1,
    freq_end=3.0,
    joint_indices="all",
    safety_enabled=True,
    metadata_extra=None,
):
    """Execute excitation trajectory on the real robot and record sensor data.

    Recorded per timestep (all 18 joints):
        q          – joint position (deg + raw ticks)
        qdot       – joint velocity (RPM + raw)
        current    – motor current (mA + raw)
        q_target   – commanded target position (deg)
        imu        – [roll, pitch, yaw, gx, gy, gz, ax, ay, az]
        timestamp  – seconds since start

    Saves <output_path>.npz  (numpy compressed)
          <output_path>.pkl  (pickle with metadata dict)
    """
    import signal as _sig

    # --- Hardware init ---
    servos = Servos()
    voltage = servos.read_voltage(1)
    servos.set_position_control()
    servos.enable_torque(range(18))

    pid_gains = sysid_read_pid_gains(servos)
    print(f"[SysID] PID gains (joint 0, D/I/P): {pid_gains[0]}")

    # --- IMU init ---
    q_imu = Queue()
    imu_proc = Process(target=read_imu, args=(q_imu,))
    imu_proc.daemon = True
    imu_proc.start()
    time.sleep(1.5)
    imu_warm = _sysid_wait_imu_ready(q_imu, timeout_sec=15.0)
    if imu_warm is None:
        print(
            "[SysID] ERROR: IMU queue empty after 15s. Check USB cable, "
            "set IMU_SERIAL_PORT (e.g. export IMU_SERIAL_PORT=/dev/ttyUSB0), "
            "and that no other process holds the IMU serial port."
        )
        try:
            imu_proc.terminate()
            imu_proc.join(timeout=1.0)
        except Exception:
            pass
        try:
            servos.disable_torque(range(18))
            servos.portHandler.closePort()
        except Exception:
            pass
        raise RuntimeError("SysID: IMU did not produce data (all-zero risk).")

    print(f"[SysID] IMU OK (warm sample max|.| = {np.max(np.abs(imu_warm)):.4f})")

    # --- Neutral pose from CPG first frame ---
    with open(tick_json, "r") as f:
        d = json.load(f)
    gps = np.asarray(d["goal_pos_sim"])
    neutral_deg = sim_angles_to_real(gps[0])
    servos.Robot_initialize(neutral_deg)
    time.sleep(1.0)
    neutral_read = servos.read_all_positions()
    print(f"[SysID] Neutral target : {neutral_deg}")
    print(f"[SysID] Neutral readback: {neutral_read}")

    # --- Joint indices ---
    if joint_indices == "all":
        ji = list(range(18))
    else:
        ji = [int(x) for x in str(joint_indices).split(",")]

    # --- Generate trajectory ---
    traj = sysid_generate_trajectory(
        trajectory_type, neutral_deg, duration, dt,
        tick_json=tick_json, joint_indices=ji,
        freq_hz=freq_hz, amplitude_deg=amplitude_deg,
        freq_start=freq_start, freq_end=freq_end,
    )
    T = traj.shape[0]
    print(f"[SysID] Trajectory: type={trajectory_type}  steps={T}  "
          f"dt={dt}s  total={T * dt:.1f}s")

    # --- Pre-allocate buffers ---
    timestamps    = np.zeros(T, dtype=np.float64)
    q_deg_buf     = np.zeros((T, 18), dtype=np.float64)
    q_raw_buf     = np.zeros((T, 18), dtype=np.int32)
    qdot_rpm_buf  = np.zeros((T, 18), dtype=np.float64)
    qdot_raw_buf  = np.zeros((T, 18), dtype=np.int32)
    cur_mA_buf    = np.zeros((T, 18), dtype=np.float64)
    cur_raw_buf   = np.zeros((T, 18), dtype=np.int32)
    pwm_raw_buf   = np.zeros((T, 18), dtype=np.int32)
    voltage_V_buf = np.zeros((T, 18), dtype=np.float64)
    q_tgt_buf     = np.zeros((T, 18), dtype=np.float64)
    imu_buf       = np.zeros((T, 9), dtype=np.float64)
    loop_ms_buf   = np.zeros(T, dtype=np.float64)

    # --- SIGINT handler ---
    _stop = [False]
    def _on_sigint(sn, fr):
        _stop[0] = True
        print("\n[SysID] SIGINT received, stopping safely...")
    old_handler = _sig.signal(_sig.SIGINT, _on_sigint)

    # Drain stale IMU data
    while not q_imu.empty():
        try:
            q_imu.get_nowait()
        except Exception:
            break

    n = 0
    imu_init = None
    last_imu = np.asarray(imu_warm, dtype=np.float64).copy()
    print("[SysID] Starting collection...")
    t_origin = time.perf_counter()

    try:
        for step in range(T):
            if _stop[0]:
                break
            t0 = time.perf_counter()
            timestamps[step] = t0 - t_origin

            # --- Write target ---
            tgt = traj[step].copy()
            q_tgt_buf[step] = tgt
            tgt_tick = angles_to_tick(tgt)

            if safety_enabled:
                clamped = np.clip(tgt_tick, SYSID_SAFE_POS_MIN_TICK,
                                  SYSID_SAFE_POS_MAX_TICK)
                if not np.allclose(clamped, tgt_tick):
                    print(f"[SysID] SAFETY: clamping target at step {step}")
                    tgt_tick = clamped

            servos.write_all_positions(tgt_tick)

            # --- Read all sensors in one bus transaction ---
            pd, vr, cm, pr, vrw, crw, pw, vv = sysid_read_all_sensors(servos)
            q_deg_buf[step]     = pd
            q_raw_buf[step]     = pr
            qdot_rpm_buf[step]  = vr
            qdot_raw_buf[step]  = vrw
            cur_mA_buf[step]    = cm
            cur_raw_buf[step]   = crw
            pwm_raw_buf[step]   = pw
            voltage_V_buf[step] = vv

            # --- Current safety ---
            if safety_enabled:
                mc = int(np.max(np.abs(crw)))
                if mc > SYSID_SAFE_CURRENT_ABS_RAW:
                    print(f"[SysID] SAFETY: current {mc} > limit "
                          f"{SYSID_SAFE_CURRENT_ABS_RAW}, aborting")
                    n = step + 1
                    break

            # --- IMU (keep last sample if queue momentarily empty) ---
            try:
                imu_data = q_imu.get(timeout=0.2)
                while not q_imu.empty():
                    imu_data = q_imu.get_nowait()
                last_imu = np.asarray(imu_data, dtype=np.float64).copy()
            except _queue_std.Empty:
                imu_data = last_imu.copy()
            if step == 0:
                imu_init = imu_data.copy()
            imu_buf[step] = imu_data

            n = step + 1
            elapsed = time.perf_counter() - t0
            loop_ms_buf[step] = elapsed * 1000.0

            if step % 50 == 0:
                v_bat = np.mean(vv[vv > 0]) if np.any(vv > 0) else 0.0
                print(f"  [{step:05d}/{T}] loop={elapsed*1000:.1f}ms  "
                      f"|I|={np.max(np.abs(cm)):.0f}mA  "
                      f"|PWM|={np.max(np.abs(pw))}  "
                      f"V_bat={v_bat:.1f}V  "
                      f"err={np.max(np.abs(pd - tgt)):.2f}deg")

            wait = dt - elapsed
            if wait > 0:
                time.sleep(wait)

    finally:
        _sig.signal(_sig.SIGINT, old_handler)

        sl = slice(0, n)
        timestamps    = timestamps[sl]
        q_deg_buf     = q_deg_buf[sl]
        q_raw_buf     = q_raw_buf[sl]
        qdot_rpm_buf  = qdot_rpm_buf[sl]
        qdot_raw_buf  = qdot_raw_buf[sl]
        cur_mA_buf    = cur_mA_buf[sl]
        cur_raw_buf   = cur_raw_buf[sl]
        pwm_raw_buf   = pwm_raw_buf[sl]
        voltage_V_buf = voltage_V_buf[sl]
        q_tgt_buf     = q_tgt_buf[sl]
        imu_buf       = imu_buf[sl]
        loop_ms_buf   = loop_ms_buf[sl]

        meta = {
            "trajectory_type": trajectory_type,
            "tick_json": tick_json,
            "duration_planned_s": duration,
            "dt_s": dt,
            "actual_steps": n,
            "actual_duration_s": float(timestamps[-1]) if n > 0 else 0.0,
            "freq_hz": freq_hz,
            "amplitude_deg": amplitude_deg,
            "freq_start": freq_start,
            "freq_end": freq_end,
            "joint_indices": ji,
            "voltage_initial_raw": float(voltage) if voltage else 0.0,
            "pid_gains_DIP": pid_gains.tolist(),
            "neutral_target_deg": neutral_deg.tolist(),
            "neutral_readback_deg": neutral_read.tolist(),
            "imu_init": imu_init.tolist() if imu_init is not None else [],
            "safety_enabled": safety_enabled,
            "num_joints": 18,
            "servo_model": "XM430-W350",
            "gear_ratio": XM430_GEAR_RATIO,
            "Kt_datasheet_Nm_per_A": XM430_KT_DATASHEET,
            "stall_torque_Nm": XM430_STALL_TORQUE_NM,
            "units": {
                "current_mA_per_raw": DXL_CURRENT_UNIT_MA,
                "velocity_RPM_per_raw": DXL_VELOCITY_UNIT_RPM,
                "position_deg_per_tick": DXL_POS_DEG_PER_TICK,
                "pwm_100pct_raw": DXL_PWM_MAX_RAW,
                "voltage_V_per_raw": DXL_VOLTAGE_UNIT_V,
                "timestamp": "seconds",
                "imu_angles": "degrees",
                "imu_gyro": "deg/s",
                "imu_acc": "m/s^2",
            },
            "table1_data_mapping": {
                "Kt":       "Identify from current_mA, qdot_rpm, pwm_raw, voltage_V via eq(1)",
                "Rter":     "Identify from pwm_raw → V_pwm = (pwm/885)*V_bat, current, back-EMF via eq(1)",
                "armature": "Identify from q̈ (diff of qdot_rpm) vs torque (from current) via dynamics",
                "eta_fw":   "Identify from motor torque vs load torque in forward-drive regimes via eq(3)",
                "eta_bw":   "Identify from motor torque vs load torque in backward-drive regimes via eq(3)",
                "fc":       "Identify from current/torque at constant velocity (Stribeck eq(4))",
                "fs":       "Identify from current/torque at near-zero velocity transitions (eq(4))",
                "kv":       "Identify from slope of torque vs velocity (eq(4))",
                "base_mass_offset": "Identify from imu orientation dynamics",
                "base_com_x":      "Identify from imu orientation dynamics",
                "base_com_z":      "Identify from imu orientation dynamics",
            },
            "collection_time": datetime.datetime.now().isoformat(),
            "extra": metadata_extra or {},
        }

        if n > 0:
            npz_path = output_path if output_path.endswith(".npz") \
                       else output_path + ".npz"
            np.savez_compressed(
                npz_path,
                timestamps=timestamps,
                q_deg=q_deg_buf,
                q_raw=q_raw_buf,
                qdot_rpm=qdot_rpm_buf,
                qdot_raw=qdot_raw_buf,
                current_mA=cur_mA_buf,
                current_raw=cur_raw_buf,
                pwm_raw=pwm_raw_buf,
                voltage_V=voltage_V_buf,
                q_target_deg=q_tgt_buf,
                imu=imu_buf,
                loop_dt_ms=loop_ms_buf,
                metadata=np.array([json.dumps(meta)]),
            )
            print(f"[SysID] Saved {n} steps -> {npz_path}")

            pkl_path = npz_path.replace(".npz", ".pkl")
            with open(pkl_path, "wb") as f:
                _sysid_pickle.dump({
                    "timestamps": timestamps,
                    "q_deg": q_deg_buf,
                    "q_raw": q_raw_buf,
                    "qdot_rpm": qdot_rpm_buf,
                    "qdot_raw": qdot_raw_buf,
                    "current_mA": cur_mA_buf,
                    "current_raw": cur_raw_buf,
                    "pwm_raw": pwm_raw_buf,
                    "voltage_V": voltage_V_buf,
                    "q_target_deg": q_tgt_buf,
                    "imu": imu_buf,
                    "loop_dt_ms": loop_ms_buf,
                    "metadata": meta,
                }, f)
            print(f"[SysID] Saved pickle -> {pkl_path}")

        # --- Safe shutdown ---
        try:
            servos.disable_torque(range(18))
        except Exception:
            pass
        try:
            imu_proc.terminate()
            imu_proc.join(timeout=1.0)
        except Exception:
            pass
        try:
            servos.portHandler.closePort()
        except Exception:
            pass

        if n > 0:
            print(f"[SysID] Done: {n} steps in {timestamps[-1]:.2f}s")
        else:
            print("[SysID] No data collected")


# ============================================================================
# End System Identification Module
# ============================================================================

import numpy as np

from Servos import *

ENABLE_COLLECT_DATASET = False

def collect_dataset_with_cpg(
    tick_json: str,
    output_dir: str,
    model_path: str = "",
    total_steps: int = 1200,
    data_flush_every: int = 200,
    flush_at_end_only: bool = False,
    log_every: int = 50,
    wm_update_stride: int = 5,
    cmd_x: float = 0.0,
    cmd_y: float = 0.5,
    cmd_yaw: float = 1.57,
    use_model: bool = False,
    wm_feature_dim: int = 512,
    wm_action_dim: int = 90,
    vel_proxy_source: str = "model",
    vel_proxy_sign_x: float = 1.0,
    vel_proxy_sign_y: float = -1.0,
    vel_proxy_sign_z: float = 1.0,
):
    # Reuse the same observation/WM/data schema as collect_dataset_from_test.py
    import test_rwm_real_robot as rrm
    from test_rwm_real_robot import (
        ROBOT_CONFIG,
        TARGET_DT,
        RealRobotRWMInference,
        create_observation_from_real_robot,
        get_action_limits,
        remove_dof_vel,
        cpg_reward,
        sim_angles_to_real,
        angles_to_tick,
        servo_angles_to_sim_angles,
    )
    from real_robot_sim2real.collect_dataset_from_test import RealVelocityEstimator

    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.join(output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "collect_log.tsv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write(
                "step\treward\tr_upright\tr_cmd\tr_smooth\tr_cpg\t"
                "v_proxy_x\tv_proxy_y\tv_proxy_z\tloop_ms\n"
            )

    with open(tick_json, "r") as f:
        d = json.load(f)
    goal_pos_sim = np.asarray(d["goal_pos_sim"], dtype=np.float32)
    # Keep original CPG data compatibility:
    # - legacy file format is often [T, 6, 3]
    # - some files may already be flattened [T, 18]
    if goal_pos_sim.ndim == 3 and goal_pos_sim.shape[1:] == (6, 3):
        goal_pos_sim = goal_pos_sim.reshape(goal_pos_sim.shape[0], 18)
    elif goal_pos_sim.ndim == 2 and goal_pos_sim.shape[1] == 18:
        pass
    else:
        raise RuntimeError(f"goal_pos_sim shape invalid: {goal_pos_sim.shape}, expected (T,6,3) or (T,18)")

    # Keep collection smooth by disabling compile in collection mode.
    rrm.WM_OPT_TORCH_COMPILE = False
    rrm.POLICY_OPT_TORCH_COMPILE = False
    rrm.WM_OPT_COMPILE_OBS_STEP = False

    rwm = None
    actor_critic = None
    if use_model:
        if not model_path:
            raise RuntimeError("use_model=True 时必须提供 --model-path")
        rwm = RealRobotRWMInference(model_path=model_path, device="cpu", remove_dof_vel=remove_dof_vel)
        actor_critic = rwm.actor_critic
        actor_critic.eval()

    servos = Servos()
    servos.set_position_control()
    servos.enable_torque(range(18))
    servos.Robot_initialize(ROBOT_CONFIG["neutral_angles"])
    time.sleep(0.8)

    q_imu = Queue()
    imu_process = Process(target=rrm.read_imu, args=(q_imu,))
    imu_process.daemon = True
    imu_process.start()
    time.sleep(0.5)

    action_limits = get_action_limits()
    vel_estimator = RealVelocityEstimator(dt=TARGET_DT)
    history_length = 5
    obs_without_command_dim = (42 if remove_dof_vel else 60) + (6 if cpg_reward else 0)
    trajectory_history = [np.zeros(obs_without_command_dim, dtype=np.float32) for _ in range(history_length)]
    prev_action_for_obs = np.zeros(18, dtype=np.float32)
    executed_prev_action = np.zeros(18, dtype=np.float32)
    dataset_cache = []
    chunk_id = 0
    last_wm_feat = None
    vel_proxy_ma_buf = []
    vel_proxy_reward_ma_window = 1

    def _flush(force: bool = False):
        nonlocal dataset_cache, chunk_id
        if flush_at_end_only and (not force):
            return
        if (not force) and len(dataset_cache) < data_flush_every:
            return
        if len(dataset_cache) == 0:
            return
        chunk_file = os.path.join(dataset_dir, f"train_chunk_{chunk_id:05d}.npz")
        np.savez_compressed(
            chunk_file,
            obs=np.stack([x["obs"] for x in dataset_cache], axis=0).astype(np.float32),
            history=np.stack([x["history"] for x in dataset_cache], axis=0).astype(np.float32),
            wm_feature=np.stack([x["wm_feature"] for x in dataset_cache], axis=0).astype(np.float32),
            action=np.stack([x["action"] for x in dataset_cache], axis=0).astype(np.float32),
            reward=np.stack([x["reward"] for x in dataset_cache], axis=0).astype(np.float32),
            imu=np.stack([x["imu"] for x in dataset_cache], axis=0).astype(np.float32),
            vel_est=np.stack([x["vel_est"] for x in dataset_cache], axis=0).astype(np.float32),
            vel_proxy=np.stack([x["vel_proxy"] for x in dataset_cache], axis=0).astype(np.float32),
            wm_action=np.stack([x["wm_action"] for x in dataset_cache], axis=0).astype(np.float32),
        )
        dataset_cache.clear()
        chunk_id += 1

    def _compute_reward(obs_prop, prev_action, action, vel_proxy):
        base_ang = obs_prop[0:3] / 0.015
        gravity = obs_prop[3:6]
        command = obs_prop[6:9]
        upright = float(np.exp(-3.0 * (gravity[0] ** 2 + gravity[1] ** 2)))
        vel_track = float(np.exp(-2.5 * ((vel_proxy[0] - command[0]) ** 2 + (vel_proxy[1] - command[1]) ** 2)))
        yaw_track = float(np.exp(-2.0 * (base_ang[2] - command[2]) ** 2))
        smooth = float(np.exp(-5.0 * np.mean((action - prev_action) ** 2)))
        total = 0.26 * upright + 0.24 * vel_track + 0.18 * yaw_track + 0.12 * smooth + 0.10 * float(
            np.exp(-0.1 * np.sum(base_ang ** 2))
        )
        return total, upright, 0.5 * (vel_track + yaw_track), smooth

    print(f"[CPG-Collect] start total_steps={total_steps} output={output_dir}")
    log_f = open(log_path, "a", buffering=1)
    try:
        for step in range(total_steps):
            t0 = time.perf_counter()
            obs_prop, obs_wo_cmd, _, imu_data = create_observation_from_real_robot(
                servos,
                q_imu,
                step,
                history_length,
                cpg_reward=cpg_reward,
                previous_actions=prev_action_for_obs,
                imu_timeout_sec=0.05,
                imu_drain_max=2,
                imu_reinit_period_sec=None,
            )

            trajectory_history.pop(0)
            trajectory_history.append(obs_wo_cmd.astype(np.float32))
            history_flat = np.concatenate(trajectory_history, axis=0)

            obs_t = torch.tensor(obs_prop, dtype=torch.float32).unsqueeze(0)
            obs_t[0, 6] = float(cmd_x)
            obs_t[0, 7] = float(cmd_y)
            obs_t[0, 8] = float(cmd_yaw)
            obs_prop = obs_t[0].detach().cpu().numpy().astype(np.float32)
            hist_t = torch.tensor(history_flat, dtype=torch.float32).unsqueeze(0)

            if use_model:
                do_wm_update = (step % max(1, int(wm_update_stride)) == 0) or (last_wm_feat is None)
                if do_wm_update:
                    rwm._prop_buffer[0].copy_(obs_t[0])
                    wm_feat_t = rwm.update_world_model(
                        {"prop": rwm._prop_buffer, "is_first": rwm.wm_is_first},
                        prev_action=prev_action_for_obs,
                    )
                    last_wm_feat = wm_feat_t.detach().clone()
                else:
                    wm_feat_t = last_wm_feat
                wm_feature_np = wm_feat_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
                wm_action_np = rwm.wm_action.detach().cpu().numpy().reshape(-1).astype(np.float32)
            else:
                wm_feature_np = np.zeros((wm_feature_dim,), dtype=np.float32)
                wm_action_np = np.zeros((wm_action_dim,), dtype=np.float32)
                wm_feat_t = None

            action_cpg = goal_pos_sim[step % goal_pos_sim.shape[0]].astype(np.float32)
            action_for_obs = np.clip(action_cpg, action_limits["min"], action_limits["max"]).astype(np.float32)
            angles_real = sim_angles_to_real(action_for_obs.reshape(6, 3))
            theta_tick = angles_to_tick(angles_real)
            servos.write_all_positions(theta_tick)
            executed_action = servo_angles_to_sim_angles(servos.read_all_positions()).astype(np.float32)

            use_model_v = bool(use_model and vel_proxy_source == "model")
            if use_model_v:
                with torch.no_grad():
                    vel_proxy = actor_critic.get_linear_vel(obs_t, hist_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
            else:
                vel_proxy = vel_estimator.v.copy().astype(np.float32)
            vel_proxy_ma_buf.append(vel_proxy.copy())
            if len(vel_proxy_ma_buf) > vel_proxy_reward_ma_window:
                vel_proxy_ma_buf = vel_proxy_ma_buf[-vel_proxy_reward_ma_window:]
            vel_proxy_reward = np.mean(np.stack(vel_proxy_ma_buf, axis=0), axis=0).astype(np.float32)

            ang_vel_norm = float(np.linalg.norm(obs_t[0, 0:3].detach().cpu().numpy() / 0.015))
            action_norm = float(np.linalg.norm(executed_action))
            vel_est = vel_estimator.update(imu_data, action_norm=action_norm, ang_vel_norm=ang_vel_norm).astype(np.float32)
            if not use_model_v:
                vel_proxy = vel_est.copy()
            signs = np.array([vel_proxy_sign_x, vel_proxy_sign_y, vel_proxy_sign_z], dtype=np.float32)
            vel_proxy = (vel_proxy * signs).astype(np.float32)

            reward, r_upright, r_cmd, r_smooth = _compute_reward(obs_prop, executed_prev_action, executed_action, vel_proxy_reward)

            dataset_cache.append(
                {
                    "obs": obs_prop.astype(np.float32),
                    "history": history_flat.astype(np.float32),
                    "wm_feature": wm_feature_np,
                    "action": executed_action.astype(np.float32),
                    "reward": np.array(float(reward), dtype=np.float32),
                    "imu": imu_data.astype(np.float32),
                    "vel_est": vel_est.astype(np.float32),
                    "vel_proxy": vel_proxy.astype(np.float32),
                    "wm_action": wm_action_np,
                }
            )
            _flush(force=False)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            log_f.write(
                f"{step}\t{reward:.6f}\t{r_upright:.6f}\t{r_cmd:.6f}\t{r_smooth:.6f}\t0.000000\t"
                f"{vel_proxy[0]:.6f}\t{vel_proxy[1]:.6f}\t{vel_proxy[2]:.6f}\t{dt_ms:.3f}\n"
            )
            if step % max(1, int(log_every)) == 0:
                print(
                    f"[CPG-Collect][{step:05d}] R={reward:.3f} "
                    f"v_proxy=({vel_proxy[0]:.2f},{vel_proxy[1]:.2f},{vel_proxy[2]:.2f}) "
                    f"loop={dt_ms:.1f}ms"
                )

            prev_action_for_obs = action_for_obs.copy()
            executed_prev_action = executed_action.copy()
            dt = time.perf_counter() - t0
            if dt < TARGET_DT:
                time.sleep(TARGET_DT - dt)
    finally:
        log_f.close()
        _flush(force=True)
        try:
            servos.disable_torque(range(18))
        except Exception:
            pass
        try:
            imu_process.terminate()
            imu_process.join(timeout=1.0)
        except Exception:
            pass
        try:
            if hasattr(servos, "portHandler"):
                servos.portHandler.closePort()
        except Exception:
            pass
        print("[CPG-Collect] done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-dataset", action="store_true", help="Run CPG gait and save dataset in train_chunk_*.npz schema")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--use-model", action="store_true", help="启用模型推理，写入真实wm_feature/wm_action并用model vel_proxy")
    parser.add_argument("--wm-feature-dim", type=int, default=512, help="不启用模型时写入零向量的wm_feature维度")
    parser.add_argument("--wm-action-dim", type=int, default=90, help="不启用模型时写入零向量的wm_action维度")
    parser.add_argument("--vel-proxy-source", type=str, default="model", choices=["model", "imu"], help="v_proxy来源")
    parser.add_argument("--vel-proxy-sign-x", type=float, default=1.0)
    parser.add_argument("--vel-proxy-sign-y", type=float, default=-1.0)
    parser.add_argument("--vel-proxy-sign-z", type=float, default=1.0)
    parser.add_argument("--tick-json", type=str, default="pos_cpg_4_new_36_1.json")
    parser.add_argument("--output-dir", type=str, default="real_robot_sim2real/outputs/cpg_dataset_collect")
    parser.add_argument("--total-steps", type=int, default=1200)
    parser.add_argument("--data-flush-every", type=int, default=200)
    parser.add_argument("--flush-at-end-only", action="store_true", help="仅在结束时保存一次数据")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--wm-update-stride", type=int, default=5)
    parser.add_argument("--cmd-x", type=float, default=0.0)
    parser.add_argument("--cmd-y", type=float, default=0.5)
    parser.add_argument("--cmd-yaw", type=float, default=1.57)
    # --- System Identification args ---
    parser.add_argument("--collect-sysid", action="store_true",
                        help="Run system identification data collection")
    parser.add_argument("--sysid-duration", type=float, default=30.0,
                        help="Collection duration in seconds")
    parser.add_argument("--sysid-dt", type=float, default=0.02,
                        help="Timestep (default 20ms = 50Hz)")
    parser.add_argument("--sysid-trajectory", type=str, default="cpg",
                        choices=["cpg", "sinusoidal", "chirp", "multi_sine", "squat"],
                        help="Excitation trajectory type")
    parser.add_argument("--sysid-output", type=str, default="sysid_data",
                        help="Output file path (without extension)")
    parser.add_argument("--sysid-freq-hz", type=float, default=0.5,
                        help="Frequency for sinusoidal trajectory")
    parser.add_argument("--sysid-amplitude", type=float, default=15.0,
                        help="Amplitude in degrees for non-CPG trajectories")
    parser.add_argument("--sysid-freq-start", type=float, default=0.1,
                        help="Start frequency for chirp trajectory")
    parser.add_argument("--sysid-freq-end", type=float, default=3.0,
                        help="End frequency for chirp trajectory")
    parser.add_argument("--sysid-joints", type=str, default="all",
                        help="Joint indices (comma-separated) or 'all'")
    parser.add_argument("--sysid-no-safety", action="store_true",
                        help="Disable safety checks (use with caution)")
    args, _ = parser.parse_known_args()

    # --- System Identification ---
    if args.collect_sysid:
        collect_sysid_data(
            duration=args.sysid_duration,
            dt=args.sysid_dt,
            trajectory_type=args.sysid_trajectory,
            tick_json=args.tick_json,
            output_path=args.sysid_output,
            freq_hz=args.sysid_freq_hz,
            amplitude_deg=args.sysid_amplitude,
            freq_start=args.sysid_freq_start,
            freq_end=args.sysid_freq_end,
            joint_indices=args.sysid_joints,
            safety_enabled=not args.sysid_no_safety,
        )
        sys.exit(0)

    # --- Legacy dataset collection (disabled by default) ---
    if args.collect_dataset:
        if not ENABLE_COLLECT_DATASET:
            print("[WARNING] --collect-dataset is disabled by default.")
            print("  Set ENABLE_COLLECT_DATASET = True in this file to enable.")
            print("  Use --collect-sysid for system identification instead.")
            sys.exit(1)
        collect_dataset_with_cpg(
            model_path=args.model_path,
            tick_json=args.tick_json,
            output_dir=args.output_dir,
            total_steps=args.total_steps,
            data_flush_every=args.data_flush_every,
            flush_at_end_only=args.flush_at_end_only,
            log_every=args.log_every,
            wm_update_stride=args.wm_update_stride,
            cmd_x=args.cmd_x,
            cmd_y=args.cmd_y,
            cmd_yaw=args.cmd_yaw,
            use_model=args.use_model,
            wm_feature_dim=args.wm_feature_dim,
            wm_action_dim=args.wm_action_dim,
            vel_proxy_source=args.vel_proxy_source,
            vel_proxy_sign_x=args.vel_proxy_sign_x,
            vel_proxy_sign_y=args.vel_proxy_sign_y,
            vel_proxy_sign_z=args.vel_proxy_sign_z,
        )
        sys.exit(0)

    #global IMU_data 
    file_first_name='force_real_four_'
    file_last_name='new_36_1'
    file_name=file_first_name+file_last_name+'.json'
    output_file_first_name='pos_cpg_4_'
    record_file_first_name='record_fix_cpg_4_'
    output_file_name=record_file_first_name+file_last_name+'_test_mbrl_supergrass.csv'
    tick_file_name=output_file_first_name+file_last_name+'.json'
    
    # file_name="force_real17.json"
    #tick_file_name="pos_20_17_3.json"
    tick_file_name="pos_cpg_4_new_36_1.json"
    # tick_file_name="pos_cpg_5_new_17_3.json"
    #output_file_name='record_fix_new_cpg_3_3.csv'
    
        
    
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
    
    
    
    
        
