#!/usr/bin/env python3
"""
Raspberry Pi-side real robot client for remote PC training.

Run this on the Pi.  It keeps the 50Hz hardware loop local, sends observations
to the PC, receives actions, and handles human reset prompts when the PC says a
reset is required.
"""

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


ACTION_DIM = 18
USE_ASYMMETRIC_ANKLE_MAPPING = True
ASYM_ANKLE_LIFT_RANGE_RAD = 1.0
ASYM_ANKLE_SINK_RANGE_RAD = 0.10


def _send_json(sock, server_addr, payload):
    sock.sendto(json.dumps(payload).encode("utf-8"), server_addr)


def _request(sock, server_addr, payload, step, timeout_s):
    # Drop stale packets before sending the current request.
    sock.settimeout(0.0)
    try:
        while True:
            sock.recvfrom(1024 * 1024)
    except Exception:
        pass

    _send_json(sock, server_addr, payload)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        sock.settimeout(max(0.001, deadline - time.time()))
        data, _ = sock.recvfrom(1024 * 1024)
        resp = json.loads(data.decode("utf-8"))
        resp_type = resp.get("type")
        if resp_type in ("reset_required", "battery_reset_required"):
            return resp
        if int(resp.get("step", -1)) == int(step):
            return resp
    raise TimeoutError(f"no matched PC response for step={step}")


def _send_reset_done(sock, server_addr, step):
    _send_json(sock, server_addr, {"type": "reset_done", "step": int(step), "ts": time.time()})


def _execute_action(servos, action_raw, position_read, admittance_filter, admittance_needs_init):
    action_scale = np.asarray(ACTION_SCALE_PER_DIM, dtype=np.float32)
    action_limits = get_action_limits()

    action_for_obs = np.clip(np.asarray(action_raw, dtype=np.float32), action_limits["min"], action_limits["max"])
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
    return action_for_obs, admittance_needs_init


def _send_safe_pose(servos):
    neutral_angles = ROBOT_CONFIG["neutral_angles"]
    real_angles = radians_to_degrees(neutral_angles * np.pi / 180.0)
    servos.Robot_initialize(real_angles)


def _read_voltage_safe(servos, last_voltage=None):
    try:
        return float(servos.read_voltage(1))
    except Exception as exc:
        if last_voltage is None:
            print(f"[RPI] WARNING failed to read voltage: {exc}")
        return last_voltage


def _extract_yaw_from_imu(imu_data):
    try:
        if imu_data is None:
            return None
        arr = np.asarray(imu_data, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 3:
            return None
        # JY901 callback returns [roll, pitch, yaw, gyroX, ...] in degrees.
        return float(arr[2])
    except Exception:
        return None


def run_client(server_ip, server_port, timeout_s, max_steps, log_every, voltage_check_every):
    if HARDWARE_IMPORT_ERROR is not None:
        raise RuntimeError(f"hardware import failed: {HARDWARE_IMPORT_ERROR}")

    servos = Servos()
    voltage = _read_voltage_safe(servos)
    if voltage is not None and voltage < ROBOT_CONFIG["control"]["voltage_threshold"]:
        print(f"[RPI] WARNING low voltage: {voltage:.2f}V")
    servos.set_position_control()
    position_all = range(ACTION_DIM)
    servos.enable_torque(position_all)

    q_imu = Queue()
    imu_process = Process(target=read_imu, args=(q_imu,))
    imu_process.daemon = True
    imu_process.start()
    time.sleep(1.0)

    print("[RPI] Moving robot to initial safe pose...")
    _send_safe_pose(servos)
    print("[RPI] Initial pose reached. Press ENTER to start remote training.")
    input()
    time.sleep(0.8)

    history_length = 5
    remove_dof_vel = True
    cpg_reward = True
    obs_without_command_dim = (42 if remove_dof_vel else 60) + (6 if cpg_reward else 0)
    trajectory_history = deque(maxlen=history_length)
    for _ in range(history_length):
        trajectory_history.append(np.zeros(obs_without_command_dim, dtype=np.float32))

    admittance_filter = AdmittanceFilter(m=0.5, d=15.0, k=80.0, dt=TARGET_DT, num_joints=ACTION_DIM)
    admittance_needs_init = True

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_addr = (server_ip, server_port)
    prev_action_for_obs = None
    last_safe_action = np.zeros(ACTION_DIM, dtype=np.float32)
    net_ok_count = 0
    fallback_count = 0

    try:
        step = 0
        while max_steps <= 0 or step < max_steps:
            t_start = time.time()
            if step == 0 or (step % max(1, int(voltage_check_every)) == 0):
                voltage = _read_voltage_safe(servos, voltage)
            real_obs, obs_wo_cmd, position_read, _imu = create_observation_from_real_robot(
                servos, q_imu, step, history_length, cpg_reward, prev_action_for_obs
            )
            yaw = _extract_yaw_from_imu(_imu)
            history_flat = np.concatenate(list(trajectory_history)).astype(np.float32)
            payload = {
                "type": "obs",
                "step": int(step),
                "obs": [float(x) for x in real_obs],
                "history": [float(x) for x in history_flat],
                "prev_action": None
                if prev_action_for_obs is None
                else [float(x) for x in prev_action_for_obs],
                "voltage": None if voltage is None else float(voltage),
                "yaw": yaw,
                "ts": time.time(),
            }

            try:
                resp = _request(sock, server_addr, payload, step, timeout_s)
            except Exception:
                resp = {"type": "act", "ok": False, "action_raw": last_safe_action.tolist()}
                fallback_count += 1
                if fallback_count <= 5 or step % max(1, log_every) == 0:
                    print(f"[RPI] network fallback at step={step}; using last safe action")

            if resp.get("type") in ("reset_required", "battery_reset_required"):
                is_battery_reset = resp.get("type") == "battery_reset_required"
                print("\n==============================")
                print("!!! BATTERY RESET REQUIRED !!!" if is_battery_reset else "!!! RESET REQUIRED !!!")
                print(f"Reason from PC: {resp.get('reason', 'unknown')}")
                if is_battery_reset:
                    v = resp.get("voltage", voltage)
                    if v is None:
                        print("Current voltage: unavailable")
                    else:
                        print(f"Current voltage: {float(v):.2f} V")
                    print("Robot will move to safe neutral pose if possible.")
                    print("Please replace the battery with a fully charged one.")
                    print("Then manually reset the robot to a safe standing pose.")
                else:
                    print("Robot will move to safe neutral pose if possible.")
                    print("Please manually reset the robot to a safe standing pose.")
                print("Then press ENTER to continue...")
                print("==============================\n")
                _send_safe_pose(servos)
                input()
                prev_action_for_obs = None
                last_safe_action = np.zeros(ACTION_DIM, dtype=np.float32)
                trajectory_history.clear()
                for _ in range(history_length):
                    trajectory_history.append(np.zeros(obs_without_command_dim, dtype=np.float32))
                admittance_needs_init = True
                voltage = _read_voltage_safe(servos, voltage)
                _send_reset_done(sock, server_addr, step)
                print("[RPI] reset_done sent to PC; resuming control")
                step += 1
                continue

            if resp.get("ok"):
                action_raw = np.asarray(resp["action_raw"], dtype=np.float32)
                last_safe_action = action_raw.copy()
                net_ok_count += 1
            else:
                action_raw = last_safe_action.copy()
                fallback_count += 1

            prev_action_for_obs, admittance_needs_init = _execute_action(
                servos=servos,
                action_raw=action_raw,
                position_read=position_read,
                admittance_filter=admittance_filter,
                admittance_needs_init=admittance_needs_init,
            )
            trajectory_history.append(obs_wo_cmd.copy())

            while (time.time() - t_start) < TARGET_DT:
                pass

            if step % max(1, log_every) == 0:
                print(
                    f"[RPI] step={step} ok={net_ok_count} fallback={fallback_count} "
                    f"voltage={voltage} "
                    f"action[min={action_raw.min():.4f}, max={action_raw.max():.4f}, mean={action_raw.mean():.4f}]"
                )
            step += 1
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
    parser = argparse.ArgumentParser(description="Raspberry Pi remote training client")
    parser.add_argument("--server-ip", required=True, help="PC IP running pc_remote_dreamer_training_server.py")
    parser.add_argument("--server-port", type=int, default=9876)
    parser.add_argument("--timeout-ms", type=float, default=50.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--voltage-check-every", type=int, default=50)
    args = parser.parse_args()
    run_client(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout_s=max(0.001, args.timeout_ms / 1000.0),
        max_steps=args.max_steps,
        log_every=args.log_every,
        voltage_check_every=args.voltage_check_every,
    )


if __name__ == "__main__":
    main()
