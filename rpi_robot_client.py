import argparse
import csv
import json
import os
import socket
import threading
import time
from collections import deque
from multiprocessing import Process, Queue

import numpy as np

from test_rwm_real_robot_wm import (
    AdmittanceFilter,
    HARDWARE_IMPORT_ERROR,
    MAX_STEPS,
    ROBOT_CONFIG,
    TARGET_DT,
    USE_ADMITTANCE,
    action_to_servo_angles,
    angles_to_ticks,
    create_observation_from_real_robot,
    get_action_limits,
    radians_to_degrees,
    read_imu,
    servo_angles_to_sim_angles,
)
from deployment_safety import (
    RiskLevelEstimator,
    SafetyActionFilter,
    RIGHT_FRONT_ANKLE_ACTION_OFFSET,
    RIGHT_FRONT_KNEE_ACTION_OFFSET,
    apply_right_front_action_offset,
    default_action_scale_per_dim,
    policy_action_to_exec_rad,
)
from Servos import Servos


USE_ASYMMETRIC_ANKLE_MAPPING = True
ASYM_ANKLE_LIFT_RANGE_RAD = 1.0
ASYM_ANKLE_SINK_RANGE_RAD = 0.10
LOG_FLUSH_EVERY_N_STEPS = 20
ASYNC_CONTROL_ENABLED = True
ASYNC_MAX_ACTION_AGE_MS = 250.0
ASYNC_IDLE_SLEEP_S = 0.001


class AsyncActionClient:
    def __init__(self, server_addr, timeout_s=0.05):
        self.server_addr = server_addr
        self.timeout_s = float(timeout_s)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.request_seq = 0
        self.latest_request = None
        self.latest_response = None
        self.latest_response_time = 0.0
        self.latest_request_ms = 0.0
        self.latest_error = "no_response"
        self.worker_ok_count = 0
        self.worker_fallback_count = 0
        self.thread = threading.Thread(target=self._run, name="AsyncActionClient", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=max(0.5, self.timeout_s + 0.2))

    def submit(self, step, obs, history, prev_action):
        req = {
            "step": int(step),
            "obs": np.asarray(obs, dtype=np.float32).copy(),
            "history": np.asarray(history, dtype=np.float32).copy(),
            "prev_action": None if prev_action is None else np.asarray(prev_action, dtype=np.float32).copy(),
        }
        with self.lock:
            self.request_seq += 1
            self.latest_request = req

    def snapshot(self):
        now = time.perf_counter()
        with self.lock:
            resp = None if self.latest_response is None else dict(self.latest_response)
            age_ms = float("inf") if resp is None else float((now - self.latest_response_time) * 1000.0)
            return {
                "resp": resp,
                "action_age_ms": age_ms,
                "request_ms": float(self.latest_request_ms),
                "net_error": str(self.latest_error),
                "worker_ok_count": int(self.worker_ok_count),
                "worker_fallback_count": int(self.worker_fallback_count),
                "request_seq": int(self.request_seq),
            }

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        last_sent_seq = 0
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    seq = int(self.request_seq)
                    req = self.latest_request
                if req is None or seq == last_sent_seq:
                    self.stop_event.wait(ASYNC_IDLE_SLEEP_S)
                    continue
                last_sent_seq = seq
                request_t0 = time.perf_counter()
                try:
                    resp = _request_action(
                        sock=sock,
                        server_addr=self.server_addr,
                        step=req["step"],
                        obs=req["obs"],
                        history=req["history"],
                        prev_action=req["prev_action"],
                        timeout_s=self.timeout_s,
                    )
                    request_ms = float((time.perf_counter() - request_t0) * 1000.0)
                    resp_step = int(resp.get("step", -1))
                    ok = bool(resp.get("ok")) and resp_step == int(req["step"])
                    with self.lock:
                        self.latest_request_ms = request_ms
                        if ok:
                            self.latest_response = resp
                            self.latest_response_time = time.perf_counter()
                            self.latest_error = ""
                            self.worker_ok_count += 1
                        else:
                            self.latest_error = f"bad_response_ok={resp.get('ok')}_step={resp_step}"
                            self.worker_fallback_count += 1
                except Exception as e:
                    request_ms = float((time.perf_counter() - request_t0) * 1000.0)
                    with self.lock:
                        self.latest_request_ms = request_ms
                        self.latest_error = type(e).__name__
                        self.worker_fallback_count += 1
        finally:
            try:
                sock.close()
            except Exception:
                pass


def _request_action(sock, server_addr, step, obs, history, prev_action, timeout_s=0.05):
    msg = {
        "type": "obs",
        "step": int(step),
        "obs": [float(x) for x in obs],
        "history": [float(x) for x in history],
        "prev_action": None if prev_action is None else [float(x) for x in prev_action],
        "ts": time.time(),
    }
    # Drop any stale UDP packets in recv buffer before sending current request.
    sock.settimeout(0.0)
    try:
        while True:
            sock.recvfrom(1024 * 1024)
    except Exception:
        pass

    sock.sendto(json.dumps(msg).encode("utf-8"), server_addr)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        remain = max(0.001, deadline - time.time())
        sock.settimeout(remain)
        data, _ = sock.recvfrom(1024 * 1024)
        resp = json.loads(data.decode("utf-8"))
        # Accept only response for current step; ignore stale delayed packets.
        if int(resp.get("step", -1)) == int(step):
            return resp
    raise TimeoutError(f"no matched response for step={step}")


def run_client(server_ip, server_port, timeout_s=0.05, log_every=50, enable_rate_limiter=True):
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
    # Match local mode UX: reach initial pose first, then wait for explicit start.
    print("[RPI] Robot moved to initial pose. Press Enter to start control, or Ctrl+C to abort.")
    input()
    time.sleep(0.8)

    history_length = 5
    remove_dof_vel = True
    cpg_reward = True
    obs_without_command_dim = (42 if remove_dof_vel else 60) + (6 if cpg_reward else 0)
    trajectory_history = deque(maxlen=history_length)
    for _ in range(history_length):
        trajectory_history.append(np.zeros(obs_without_command_dim, dtype=np.float32))

    action_scale = default_action_scale_per_dim().astype(np.float32)
    action_limits = get_action_limits()
    safety_filter = SafetyActionFilter(action_limits=action_limits)
    risk_estimator = RiskLevelEstimator()
    admittance_filter = AdmittanceFilter(m=0.5, d=15.0, k=80.0, dt=TARGET_DT, num_joints=18)
    admittance_needs_init = True
    print(
        "[RPI] Right-front action offset: "
        f"knee[10]+={RIGHT_FRONT_KNEE_ACTION_OFFSET:.2f}, "
        f"ankle[11]+={RIGHT_FRONT_ANKLE_ACTION_OFFSET:.2f}"
    )
    print(f"[RPI] Rate limiter: {'ENABLED' if enable_rate_limiter else 'DISABLED'}")
    print(
        "[RPI] Remote control mode: "
        f"{'ASYNC' if ASYNC_CONTROL_ENABLED else 'SYNC'} "
        f"target_dt={TARGET_DT * 1000.0:.1f}ms "
        f"max_action_age={ASYNC_MAX_ACTION_AGE_MS:.1f}ms"
    )

    server_addr = (server_ip, server_port)
    sock = None
    async_client = None
    if ASYNC_CONTROL_ENABLED:
        async_client = AsyncActionClient(server_addr, timeout_s=timeout_s)
        async_client.start()
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    prev_action_for_obs = None
    last_safe_action = np.zeros(18, dtype=np.float32)
    last_used_resp_step = -1
    net_ok_count = 0
    net_fallback_count = 0
    last_event_key = None

    # Keep remote-client logs consistent with local mode: always write under
    # this script directory so cwd differences do not affect output location.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "validation_outputs")
    os.makedirs(output_dir, exist_ok=True)
    actions_file = os.path.join(output_dir, "actions_output.txt")
    observations_file = os.path.join(output_dir, "observations_output.txt")
    safety_debug_file = os.path.join(output_dir, "safety_debug.csv")
    print(f"[RPI] Actions will be logged to: {actions_file}")
    print(f"[RPI] Observations will be logged to: {observations_file}")
    print(f"[RPI] Safety debug will be logged to: {safety_debug_file}")

    try:
        with open(actions_file, "w") as f_actions, open(observations_file, "w") as f_obs, open(safety_debug_file, "w", newline="") as f_sd:
            f_actions.write("step\tactions_18\tangles_18\tticks_18\n")
            f_obs.write(
                "step\tbase_ang_vel_3\tprojected_gravity_3\tcommands_3\tdof_pos_18\tdof_vel_18\tobs_action_18\troll\tpitch\tyaw\n"
            )
            sd_writer = csv.DictWriter(
                f_sd,
                fieldnames=[
                    "step", "raw_action_min", "raw_action_max", "raw_action_mean",
                    "exec_action_min", "exec_action_max", "exec_action_mean",
                    "net_ok", "net_fallback", "net_error", "resp_step",
                    "server_ms", "request_ms", "control_dt_ms", "loop_dt_ms", "timeout_ms",
                    "async_mode", "async_new_response", "async_action_age_ms",
                    "async_request_seq", "async_worker_ok_count", "async_worker_fallback_count",
                    "risk_level", "wm_error", "contact_error", "contact_anomaly", "contact_steps",
                    "detected_bad_leg", "bad_leg", "candidate_selected", "candidate_group", "candidate_score",
                    "effective_bad_leg", "effective_risk_level",
                    "forced_lift", "forced_lift_reason",
                    "recovery_latch_active", "recovery_latch_leg", "recovery_latch_hold",
                    "recovery_latch_triggered", "recovery_latch_accepted",
                    "lift_hip_forward_target", "lift_hip_forward_blend",
                    "lift_knee_final_target", "lift_knee_raw_target", "lift_knee_blend",
                    "lift_ankle_final_target", "lift_ankle_raw_target", "lift_ankle_blend",
                    "max_delta_before_filter",
                    "max_delta_after_filter", "ankle_delta_max",
                ],
            )
            sd_writer.writeheader()
            for step in range(MAX_STEPS):
                t_start = time.perf_counter()
                resp = {}
                resp_step = -1
                server_ms = 0.0
                request_ms = 0.0
                net_ok_step = False
                net_fallback_step = False
                net_error = ""
                request_t0 = None
                async_new_response = False
                async_action_age_ms = float("inf")
                async_request_seq = 0
                async_worker_ok_count = 0
                async_worker_fallback_count = 0
                real_obs, obs_wo_cmd, position_read, imu_data = create_observation_from_real_robot(
                    servos, q_imu, step, history_length, cpg_reward, prev_action_for_obs
                )
                history_flat = np.concatenate(list(trajectory_history)).astype(np.float32)

                if ASYNC_CONTROL_ENABLED:
                    async_client.submit(
                        step=step,
                        obs=real_obs,
                        history=history_flat,
                        prev_action=prev_action_for_obs,
                    )
                    async_dbg = async_client.snapshot()
                    resp = async_dbg["resp"] or {}
                    async_action_age_ms = float(async_dbg["action_age_ms"])
                    async_request_seq = int(async_dbg["request_seq"])
                    async_worker_ok_count = int(async_dbg["worker_ok_count"])
                    async_worker_fallback_count = int(async_dbg["worker_fallback_count"])
                    request_ms = float(async_dbg["request_ms"])
                    resp_step = int(resp.get("step", -1))
                    server_ms = float(resp.get("server_ms", 0.0))
                    fresh_resp = bool(resp.get("ok")) and async_action_age_ms <= ASYNC_MAX_ACTION_AGE_MS
                    async_new_response = bool(fresh_resp and resp_step != last_used_resp_step)
                    if fresh_resp:
                        action_raw = np.asarray(resp["action_raw"], dtype=np.float32)
                        last_safe_action = action_raw.copy()
                        last_used_resp_step = resp_step
                        net_ok_count += 1
                        net_ok_step = True
                    else:
                        action_raw = last_safe_action.copy()
                        net_fallback_count += 1
                        net_fallback_step = True
                        if resp:
                            net_error = "stale_async_action"
                        else:
                            net_error = str(async_dbg["net_error"] or "no_async_response")
                        if (net_fallback_count <= 5) or (step % max(1, int(log_every)) == 0):
                            age_text = "inf" if not np.isfinite(async_action_age_ms) else f"{async_action_age_ms:.1f}"
                            print(f"[RPI] fallback step={step}: async action unavailable/stale age={age_text}ms")
                else:
                    try:
                        request_t0 = time.perf_counter()
                        resp = _request_action(
                            sock=sock,
                            server_addr=server_addr,
                            step=step,
                            obs=real_obs,
                            history=history_flat,
                            prev_action=prev_action_for_obs,
                            timeout_s=timeout_s,
                        )
                        request_ms = float((time.perf_counter() - request_t0) * 1000.0)
                        resp_step = int(resp.get("step", -1))
                        server_ms = float(resp.get("server_ms", 0.0))
                        if resp.get("ok") and resp_step == step:
                            action_raw = np.asarray(resp["action_raw"], dtype=np.float32)
                            last_safe_action = action_raw.copy()
                            net_ok_count += 1
                            net_ok_step = True
                        else:
                            action_raw = last_safe_action.copy()
                            net_fallback_count += 1
                            net_fallback_step = True
                            net_error = f"bad_response_ok={resp.get('ok')}_step={resp_step}"
                            if (net_fallback_count <= 5) or (step % max(1, int(log_every)) == 0):
                                print(
                                    f"[RPI] fallback step={step}: resp_ok={resp.get('ok')} resp_step={resp.get('step')}"
                                )
                    except Exception as e:
                        # network timeout / decode error fallback
                        request_ms = float((time.perf_counter() - request_t0) * 1000.0) if request_t0 is not None else 0.0
                        action_raw = last_safe_action.copy()
                        net_fallback_count += 1
                        net_fallback_step = True
                        net_error = type(e).__name__
                        if (net_fallback_count <= 5) or (step % max(1, int(log_every)) == 0):
                            print(f"[RPI] fallback step={step}: timeout/decode, using last_safe_action")

                # Log observation in the same format used by local mode.
                obs_str = "\t".join([f"{x:.6f}" for x in real_obs])
                roll, pitch, yaw = float(imu_data[0]), float(imu_data[1]), float(imu_data[2])
                f_obs.write(f"{step}\t{obs_str}\t{roll:.6f}\t{pitch:.6f}\t{yaw:.6f}\n")
                if (step % LOG_FLUSH_EVERY_N_STEPS) == 0:
                    f_obs.flush()

                action_for_obs = np.clip(action_raw, -1.0, 1.0)
                action_for_exec = apply_right_front_action_offset(action_for_obs)
                prev_action_for_obs = action_for_obs.copy()

                contact_error = float(resp.get("contact_error", 0.0))
                contact_steps = int(resp.get("contact_steps", 0))
                contact_anomaly = bool(resp.get("contact_anomaly", False))
                risk_state = risk_estimator.update(
                    wm_error=contact_error,
                    contact_anomaly=contact_anomaly,
                    contact_steps=contact_steps,
                    bad_leg=int(resp.get("bad_leg", -1)),
                )
                risk_level = int(resp.get("risk_level", risk_state.level))
                risk_level = max(0, min(2, risk_level))
                risk_ema = float(resp.get("risk_ema", risk_state.ema_error))

                action_exec_desired = policy_action_to_exec_rad(
                    action_raw_clipped=action_for_exec,
                    action_scale_per_dim=action_scale,
                    use_asymmetric_ankle_mapping=USE_ASYMMETRIC_ANKLE_MAPPING,
                    asym_lift_range_rad=ASYM_ANKLE_LIFT_RANGE_RAD,
                    asym_sink_range_rad=ASYM_ANKLE_SINK_RANGE_RAD,
                )
                action_exec_desired = np.clip(action_exec_desired, action_limits["min"], action_limits["max"])
                if enable_rate_limiter:
                    action_exec, safety_dbg = safety_filter.filter(action_exec_desired, risk_level)
                else:
                    action_exec, safety_dbg = safety_filter.bypass(action_exec_desired)
                if USE_ADMITTANCE:
                    if admittance_needs_init:
                        current_sim_angles = servo_angles_to_sim_angles(position_read)
                        init_action = np.clip(current_sim_angles, action_limits["min"], action_limits["max"])
                        admittance_filter.reset(init_action)
                        safety_filter.reset(init_action)
                        admittance_needs_init = False
                    action_exec = admittance_filter.update(action_exec)
                    action_exec = np.clip(action_exec, action_limits["min"], action_limits["max"])

                servo_angles = action_to_servo_angles(action_exec)
                angle_limits = ROBOT_CONFIG["angle_limits"]
                servo_angles = np.clip(servo_angles, angle_limits["min"], angle_limits["max"])
                ticks = angles_to_ticks(servo_angles)
                servos.write_all_positions(ticks)

                # Log actions/angles/ticks in the same format used by local mode.
                actions_str = "\t".join([f"{x:.6f}" for x in action_exec])
                servo_angles_str = "\t".join([f"{x:.6f}" for x in servo_angles])
                ticks_str = "\t".join([f"{int(x)}" for x in ticks])
                f_actions.write(f"{step}\t{actions_str}\t{servo_angles_str}\t{ticks_str}\n")
                if (step % LOG_FLUSH_EVERY_N_STEPS) == 0:
                    f_actions.flush()
                    f_sd.flush()

                control_dt_ms = float((time.perf_counter() - t_start) * 1000.0)
                trajectory_history.append(obs_wo_cmd.copy())
                while (time.perf_counter() - t_start) < TARGET_DT:
                    pass
                loop_dt_ms = float((time.perf_counter() - t_start) * 1000.0)

                sd_writer.writerow(
                    {
                        "step": step,
                        "raw_action_min": float(action_for_obs.min()),
                        "raw_action_max": float(action_for_obs.max()),
                        "raw_action_mean": float(action_for_obs.mean()),
                        "exec_action_min": float(action_exec.min()),
                        "exec_action_max": float(action_exec.max()),
                        "exec_action_mean": float(action_exec.mean()),
                        "net_ok": int(net_ok_step),
                        "net_fallback": int(net_fallback_step),
                        "net_error": net_error,
                        "resp_step": int(resp_step),
                        "server_ms": float(server_ms),
                        "request_ms": float(request_ms),
                        "control_dt_ms": float(control_dt_ms),
                        "loop_dt_ms": float(loop_dt_ms),
                        "timeout_ms": float(timeout_s * 1000.0),
                        "async_mode": int(ASYNC_CONTROL_ENABLED),
                        "async_new_response": int(async_new_response),
                        "async_action_age_ms": float(async_action_age_ms if np.isfinite(async_action_age_ms) else -1.0),
                        "async_request_seq": int(async_request_seq),
                        "async_worker_ok_count": int(async_worker_ok_count),
                        "async_worker_fallback_count": int(async_worker_fallback_count),
                        "risk_level": int(risk_level),
                        "wm_error": float(risk_ema),
                        "contact_error": float(contact_error),
                        "contact_anomaly": int(contact_anomaly),
                        "contact_steps": int(contact_steps),
                        "detected_bad_leg": int(resp.get("detected_bad_leg", -1)),
                        "bad_leg": int(resp.get("bad_leg", -1)),
                        "candidate_selected": str(resp.get("candidate_selected", "nominal")),
                        "candidate_group": str(resp.get("candidate_group", "")),
                        "candidate_score": float(resp.get("candidate_score", 0.0)),
                        "effective_bad_leg": int(resp.get("effective_bad_leg", resp.get("bad_leg", -1))),
                        "effective_risk_level": int(resp.get("effective_risk_level", risk_level)),
                        "forced_lift": int(bool(resp.get("forced_lift", False))),
                        "forced_lift_reason": str(resp.get("forced_lift_reason", "")),
                        "recovery_latch_active": int(bool(resp.get("recovery_latch_active", False))),
                        "recovery_latch_leg": int(resp.get("recovery_latch_leg", -1)),
                        "recovery_latch_hold": int(resp.get("recovery_latch_hold", 0)),
                        "recovery_latch_triggered": int(bool(resp.get("recovery_latch_triggered", False))),
                        "recovery_latch_accepted": int(bool(resp.get("recovery_latch_accepted", False))),
                        "lift_hip_forward_target": float(resp.get("lift_hip_forward_target", 0.0)),
                        "lift_hip_forward_blend": float(resp.get("lift_hip_forward_blend", 0.0)),
                        "lift_knee_final_target": float(resp.get("lift_knee_final_target", 0.0)),
                        "lift_knee_raw_target": float(resp.get("lift_knee_raw_target", 0.0)),
                        "lift_knee_blend": float(resp.get("lift_knee_blend", 0.0)),
                        "lift_ankle_final_target": float(resp.get("lift_ankle_final_target", 0.0)),
                        "lift_ankle_raw_target": float(resp.get("lift_ankle_raw_target", 0.0)),
                        "lift_ankle_blend": float(resp.get("lift_ankle_blend", 0.0)),
                        "max_delta_before_filter": safety_dbg.get("max_delta_before_filter", 0.0),
                        "max_delta_after_filter": safety_dbg.get("max_delta_after_filter", 0.0),
                        "ankle_delta_max": safety_dbg.get("ankle_delta_max", 0.0),
                    }
                )

                candidate_selected = str(resp.get("candidate_selected", "nominal"))
                candidate_group = str(resp.get("candidate_group", ""))
                detected_bad_leg = int(resp.get("detected_bad_leg", -1))
                bad_leg = int(resp.get("bad_leg", -1))
                effective_bad_leg = int(resp.get("effective_bad_leg", bad_leg))
                event_key = (int(risk_level), int(contact_anomaly), bad_leg, candidate_selected)
                event_active = (
                    risk_level > 0
                    or contact_anomaly
                    or candidate_group in {"lift", "ankle_protected", "scaled", "blend"}
                )
                should_print = (step % max(1, int(log_every)) == 0) or (event_active and event_key != last_event_key)
                if should_print:
                    last_event_key = event_key
                    print(
                        f"[RPI] step={step} "
                        f"net_ok={net_ok_count} fallback={net_fallback_count} "
                        f"resp_step={resp_step} server_ms={server_ms:.1f} request_ms={request_ms:.1f} "
                        f"loop_dt={loop_dt_ms:.1f} "
                        f"async_age={async_action_age_ms if np.isfinite(async_action_age_ms) else -1.0:.1f} "
                        f"new_resp={int(async_new_response)} "
                        f"worker_ok={async_worker_ok_count} worker_fb={async_worker_fallback_count} "
                        f"raw[min={action_raw.min():.4f}, max={action_raw.max():.4f}, mean={action_raw.mean():.4f}] "
                        f"risk={risk_level} contact_anomaly={int(contact_anomaly)} contact_steps={contact_steps} "
                        f"det_bad_leg={detected_bad_leg} bad_leg={bad_leg} eff_bad_leg={effective_bad_leg} "
                        f"cand={candidate_selected} group={candidate_group} "
                        f"forced_lift={int(bool(resp.get('forced_lift', False)))} "
                        f"latch={int(bool(resp.get('recovery_latch_active', False)))} "
                        f"latch_leg={int(resp.get('recovery_latch_leg', -1))} "
                        f"hold={int(resp.get('recovery_latch_hold', 0))} "
                        f"err={contact_error:.4f} ema={risk_ema:.4f} "
                        f"dq_raw={np.degrees(float(safety_dbg.get('max_delta_before_filter', 0.0))):.2f}deg "
                        f"dq_exec={np.degrees(float(safety_dbg.get('max_delta_after_filter', 0.0))):.2f}deg"
                    )
            print("\n[RPI] Reached MAX_STEPS. Robot keeps holding last pose (torque still enabled).")
            input("[RPI] Press Enter to finish and disable torque...")
    finally:
        try:
            if async_client is not None:
                async_client.close()
        except Exception:
            pass
        try:
            if sock is not None:
                sock.close()
        except Exception:
            pass
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
    parser.add_argument("--timeout-ms", type=float, default=50.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--disable-rate-limiter",
        action="store_true",
        help="Bypass the deployment-side command rate limiter while keeping angle limits and logging.",
    )
    args = parser.parse_args()
    run_client(
        args.server_ip,
        args.server_port,
        timeout_s=max(0.001, args.timeout_ms / 1000.0),
        log_every=args.log_every,
        enable_rate_limiter=not args.disable_rate_limiter,
    )


if __name__ == "__main__":
    main()
