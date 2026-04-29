import argparse
import json
import socket
import time
from collections import deque
from multiprocessing import Process, Queue

import numpy as np

from test_rwm_real_robot_wm import (
    ACTION_SCALE_PER_DIM,
    AdmittanceFilter,
    HARDWARE_IMPORT_ERROR,
    MAX_STEPS,
    ROBOT_CONFIG,
    TARGET_DT,
    USE_ADMITTANCE,
    action_to_servo_angles,
    angles_to_ticks,
    apply_asymmetric_ankle_mapping_rad,
    create_observation_from_real_robot,
    get_action_limits,
    radians_to_degrees,
    read_imu,
    servo_angles_to_sim_angles,
)
from Servos import Servos


USE_ASYMMETRIC_ANKLE_MAPPING = True
ASYM_ANKLE_LIFT_RANGE_RAD = 1.0
ASYM_ANKLE_SINK_RANGE_RAD = 0.10


def _request_action(sock, server_addr, step, obs, history, prev_action, timeout_s=0.015):
    msg = {
        "type": "obs",
        "step": int(step),
        "obs": [float(x) for x in obs],
        "history": [float(x) for x in history],
        "prev_action": None if prev_action is None else [float(x) for x in prev_action],
        "ts": time.time(),
    }
    sock.sendto(json.dumps(msg).encode("utf-8"), server_addr)
    sock.settimeout(timeout_s)
    data, _ = sock.recvfrom(1024 * 1024)
    return json.loads(data.decode("utf-8"))


def run_client(server_ip, server_port):
    if HARDWARE_IMPORT_ERROR is not None:
        raise RuntimeError(f"hardware import failed: {HARDWARE_IMPORT_ERROR}")

    servos = Servos()
    voltage = servos.read_voltage(1)
    if voltage < ROBOT_CONFIG["control"]["voltage_threshold"]:
        print(f"[RPI] WARNING low voltage: {voltage:.2f}V")
    servos.set_position_control()
    position_all = range(18)
    servos.enable_torque(position_all)

    q_imu = Queue()
    imu_process = Process(target=read_imu, args=(q_imu,))
    imu_process.daemon = True
    imu_process.start()
    time.sleep(1.0)

    neutral_angles = ROBOT_CONFIG["neutral_angles"]
    real_angles = radians_to_degrees(neutral_angles * np.pi / 180.0)
    servos.Robot_initialize(real_angles)

    history_length = 5
    remove_dof_vel = True
    cpg_reward = True
    obs_without_command_dim = (42 if remove_dof_vel else 60) + (6 if cpg_reward else 0)
    trajectory_history = deque(maxlen=history_length)
    for _ in range(history_length):
        trajectory_history.append(np.zeros(obs_without_command_dim, dtype=np.float32))

    action_scale = np.asarray(ACTION_SCALE_PER_DIM, dtype=np.float32)
    action_limits = get_action_limits()
    admittance_filter = AdmittanceFilter(m=0.5, d=15.0, k=80.0, dt=TARGET_DT, num_joints=18)
    admittance_needs_init = True

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_addr = (server_ip, server_port)

    prev_action_for_obs = None
    last_safe_action = np.zeros(18, dtype=np.float32)

    try:
        for step in range(MAX_STEPS):
            t_start = time.time()
            real_obs, obs_wo_cmd, position_read, _imu = create_observation_from_real_robot(
                servos, q_imu, step, history_length, cpg_reward, prev_action_for_obs
            )
            history_flat = np.concatenate(list(trajectory_history)).astype(np.float32)

            try:
                resp = _request_action(
                    sock=sock,
                    server_addr=server_addr,
                    step=step,
                    obs=real_obs,
                    history=history_flat,
                    prev_action=prev_action_for_obs,
                    timeout_s=0.015,
                )
                if resp.get("ok") and int(resp.get("step", -1)) == step:
                    action_raw = np.asarray(resp["action_raw"], dtype=np.float32)
                    last_safe_action = action_raw.copy()
                else:
                    action_raw = last_safe_action.copy()
            except Exception:
                # network timeout / decode error fallback
                action_raw = last_safe_action.copy()

            action_for_obs = np.clip(action_raw, action_limits["min"], action_limits["max"])
            prev_action_for_obs = action_for_obs.copy()

            action_exec = action_for_obs * action_scale
            if USE_ASYMMETRIC_ANKLE_MAPPING:
                action_exec = apply_asymmetric_ankle_mapping_rad(
                    action_exec,
                    lift_range_rad=ASYM_ANKLE_LIFT_RANGE_RAD,
                    sink_range_rad=ASYM_ANKLE_SINK_RANGE_RAD,
                )

            action_exec = np.clip(action_exec, action_limits["min"], action_limits["max"])
            if USE_ADMITTANCE:
                if admittance_needs_init:
                    current_sim_angles = servo_angles_to_sim_angles(position_read)
                    admittance_filter.reset(np.clip(current_sim_angles, action_limits["min"], action_limits["max"]))
                    admittance_needs_init = False
                action_exec = admittance_filter.update(action_exec)
                action_exec = np.clip(action_exec, action_limits["min"], action_limits["max"])

            servo_angles = action_to_servo_angles(action_exec)
            angle_limits = ROBOT_CONFIG["angle_limits"]
            servo_angles = np.clip(servo_angles, angle_limits["min"], angle_limits["max"])
            ticks = angles_to_ticks(servo_angles)
            servos.write_all_positions(ticks)

            trajectory_history.append(obs_wo_cmd.copy())
            while (time.time() - t_start) < TARGET_DT:
                pass

            if step % 50 == 0:
                print(f"[RPI] step={step} net+ctrl ok")
    finally:
        try:
            servos.disable_torque(position_all)
        except Exception:
            pass
        try:
            imu_process.terminate()
            imu_process.join(timeout=1.0)
        except Exception:
            pass
        try:
            servos.portHandler.closePort()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi robot control UDP client")
    parser.add_argument("--server-ip", required=True, help="PC IP running wm server")
    parser.add_argument("--server-port", type=int, default=9876)
    args = parser.parse_args()
    run_client(args.server_ip, args.server_port)


if __name__ == "__main__":
    main()
