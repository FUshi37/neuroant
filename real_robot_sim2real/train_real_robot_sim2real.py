import argparse
import os
import signal
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# Ensure parent project dir is importable when running this file directly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import test_rwm_real_robot as rrm
from test_rwm_real_robot import (
    ACTION_SCALE_PER_DIM,
    ROBOT_CONFIG,
    TARGET_DT,
    USE_ASYMMETRIC_ANKLE_MAPPING,
    ASYM_ANKLE_LIFT_RANGE_RAD,
    ASYM_ANKLE_SINK_RANGE_RAD,
    RealRobotRWMInference,
    Servos,
    action_to_servo_angles,
    angles_to_ticks,
    apply_asymmetric_ankle_mapping_rad,
    create_observation_from_real_robot,
    get_action_limits,
    read_imu,
    remove_dof_vel,
    cpg_reward,
    servo_angles_to_sim_angles,
)


@dataclass
class TrainConfig:
    stage: str = "wm_only"
    total_steps: int = 3000
    update_horizon: int = 256
    policy_lr: float = 3e-5
    value_coef: float = 0.5
    entropy_coef: float = 2e-4
    vel_predict_coef: float = 0.5
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    lam: float = 0.95
    ppo_clip: float = 0.2
    num_epochs: int = 4
    minibatch_size: int = 64
    # 与 test_rwm_real_robot.py 一致：先逐维 ACTION_SCALE_PER_DIM，再乘全局倍率
    action_scale_multiplier: float = 1.0
    wm_batch_size: int = 8
    wm_batch_length: int = 32
    wm_train_every: int = 128
    wm_grad_steps: int = 2
    wm_use_replay_action: bool = False
    warmup_steps: int = 200
    log_every: int = 50
    save_every: int = 5000
    data_flush_every: int = 200
    auto_stage_switch: bool = True
    switch_min_steps: int = 1200
    switch_wm_loss_threshold: float = 0.25
    battery_mode: bool = False
    battery_run_steps: int = 1200
    battery_rest_seconds: int = 180
    battery_max_cycles: int = 4
    # Safety configs
    soft_limit_margin_deg: float = 4.0
    train_limit_extra_margin_deg: float = 8.0
    train_joint_limits_file: str = "real_robot_sim2real/train_joint_limits.json"
    max_action_delta_rad: float = 0.06
    startup_ramp_steps: int = 200
    imu_roll_pitch_stop_deg: float = 38.0
    imu_gyro_stop_dps: float = 550.0
    voltage_check_every: int = 100
    min_safe_voltage: float = 10.8
    max_consecutive_failures: int = 5
    # CPG-like rewards (real-robot feasible)
    use_cpg_rewards: bool = True
    cpg_ref_joint_file: str = "world_model/joint_angles_tetrapod.csv"
    cpg_track_sigma_rad: float = 0.35
    reward_w_upright: float = 0.26
    reward_w_vel_track: float = 0.24
    reward_w_yaw_track: float = 0.18
    reward_w_smooth: float = 0.12
    reward_w_ang_stability: float = 0.10
    reward_w_cpg_track: float = 0.10
    # Command override (obs[6:9] = [cmd_x, cmd_y, cmd_yaw])
    cmd_x: float = 0.0
    cmd_y: float = 0.1
    cmd_yaw: float = 1.57
    # Compatibility mode: make runtime as close as test_rwm_real_robot.py as possible.
    inference_compatible_mode: bool = True
    # CPU online training default: disable torch.compile to reduce jitter.
    enable_torch_compile: bool = False
    # Velocity proxy configuration (for reward + vel_predict supervision).
    # source:
    # - "model": use actor_critic.get_linear_vel()
    # - "imu": use RealVelocityEstimator integration
    # - "blend": weighted blend of model and imu
    vel_proxy_source: str = "model"
    vel_proxy_blend_alpha: float = 0.8  # only for source="blend"
    # Optional axis-wise calibration (handle sign convention mismatch quickly).
    vel_proxy_sign_x: float = 1.0
    vel_proxy_sign_y: float = 1.0
    vel_proxy_sign_z: float = 1.0
    # Reward-side velocity smoothing (moving average over recent vel_proxy).
    # 1 means no smoothing.
    vel_proxy_reward_ma_window: int = 1
    output_dir: str = "real_robot_sim2real/outputs"


class RealVelocityEstimator:
    """保留类定义用于兼容旧日志流程，当前奖励默认不使用 IMU 积分速度。"""

    def __init__(self, dt: float = 0.02, alpha: float = 0.85, leak: float = 0.995):
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.leak = float(leak)
        self.v = np.zeros(3, dtype=np.float32)
        self._acc_lp = np.zeros(3, dtype=np.float32)
        self._acc_bias = np.zeros(3, dtype=np.float32)

    def reset(self):
        self.v[:] = 0.0
        self._acc_lp[:] = 0.0
        self._acc_bias[:] = 0.0

    @staticmethod
    def _rot_matrix_from_rpy(roll: float, pitch: float, yaw: float):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        # ZYX
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
        return rz @ ry @ rx

    def update(self, imu_data: np.ndarray, action_norm: float, ang_vel_norm: float) -> np.ndarray:
        # imu_data: [roll, pitch, yaw, gyroX, gyroY, gyroZ, accX, accY, accZ]
        roll, pitch, yaw = np.deg2rad(imu_data[0]), np.deg2rad(imu_data[1]), np.deg2rad(imu_data[2])
        acc_body = np.asarray(imu_data[6:9], dtype=np.float32)
        # JY901 常见输出为 g，转 m/s^2
        acc_body = acc_body * 9.81

        r_bw = self._rot_matrix_from_rpy(roll, pitch, yaw)
        acc_world = r_bw @ acc_body
        # world z 向上，减重力
        acc_world[2] -= 9.81
        # 静止时更新加速度偏置，减少积分漂移
        static_like = (action_norm < 0.03) and (ang_vel_norm < 0.25)
        if static_like:
            self._acc_bias = 0.995 * self._acc_bias + 0.005 * acc_world
        acc_world = acc_world - self._acc_bias
        # 低通降噪
        self._acc_lp = self.alpha * self._acc_lp + (1.0 - self.alpha) * acc_world
        self.v = self.leak * self.v + self._acc_lp * self.dt

        # 零速更新：动作和角速度都很小时收敛到静止
        if static_like:
            self.v *= 0.80
            if np.linalg.norm(self.v) < 0.08:
                self.v[:] = 0.0

        # 安全裁剪，防止积分爆炸
        self.v = np.clip(self.v, -1.0, 1.0)
        return self.v.copy()


class RealRobotTrainer:
    def __init__(self, model_path: str, cfg: TrainConfig):
        self.model_path = model_path
        self.cfg = cfg
        self.device = "cpu"

        # In online CPU control loops, torch.compile can introduce first-step stalls
        # and runtime jitter. Keep it opt-in via --enable-torch-compile.
        rrm.WM_OPT_TORCH_COMPILE = bool(self.cfg.enable_torch_compile)
        rrm.POLICY_OPT_TORCH_COMPILE = bool(self.cfg.enable_torch_compile)
        rrm.WM_OPT_COMPILE_OBS_STEP = bool(self.cfg.enable_torch_compile)
        print(
            f"[Sim2Real] torch.compile={'ON' if self.cfg.enable_torch_compile else 'OFF'} "
            f"(wm/policy/obs_step)"
        )

        self.rwm = RealRobotRWMInference(model_path=model_path, device=self.device, remove_dof_vel=remove_dof_vel)
        self.actor_critic = self.rwm.actor_critic
        # Root fix: remove_dof_vel=True 时，实机本体观测就是 33 维，统一到 actor_critic 的 prop_dim 断言。
        if remove_dof_vel:
            self.actor_critic.prop_dim = 33
        self.actor_critic.train()
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=cfg.policy_lr)

        self.servos = None
        self.q_imu = None
        self.imu_process = None
        self.running = True

        self.history_length = 5
        self.obs_without_command_dim = (42 if remove_dof_vel else 60) + (6 if cpg_reward else 0)
        self.history_dim = self.obs_without_command_dim * self.history_length
        self.trajectory_history = deque(maxlen=self.history_length)
        for _ in range(self.history_length):
            self.trajectory_history.append(np.zeros(self.obs_without_command_dim, dtype=np.float32))

        self.action_limits = get_action_limits()
        self.train_angle_limits = self._load_train_joint_limits()
        self.critic_obs_dim = self._infer_critic_obs_dim()

        self.rollout: List[Dict[str, np.ndarray]] = []
        self.wm_buffer: List[Dict[str, np.ndarray]] = []

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.log_path = os.path.join(self.cfg.output_dir, "train_log.tsv")
        self.dataset_dir = os.path.join(self.cfg.output_dir, "dataset")
        os.makedirs(self.dataset_dir, exist_ok=True)
        self._dataset_cache: List[Dict[str, np.ndarray]] = []
        self._dataset_chunk_id = self._infer_next_chunk_id()
        self.vel_estimator = RealVelocityEstimator(dt=TARGET_DT)
        self._wm_loss_ema = None
        self._battery_cycle = 0
        self._battery_step_in_cycle = 0
        self._consecutive_failures = 0
        self._last_safe_action = np.zeros(18, dtype=np.float32)
        # For interrupt-resume: remember last successfully reached step index.
        self._last_step = 0
        self._last_pol_loss = 0.0
        self._last_val_loss = 0.0
        self._last_entropy = 0.0
        self._cpg_ref = self._load_cpg_reference()
        self._wm_shape_debug_printed = False
        self._init_log_file()
        self._vel_proxy_ma_buf = deque(
            maxlen=max(1, int(self.cfg.vel_proxy_reward_ma_window))
        )

    def _infer_next_chunk_id(self) -> int:
        """
        Continue chunk numbering from existing dataset files.
        Prevent overwriting train_chunk_*.npz when running multiple sessions.
        """
        try:
            files = [
                f for f in os.listdir(self.dataset_dir)
                if f.startswith("train_chunk_") and f.endswith(".npz")
            ]
            if not files:
                return 0
            max_id = -1
            for f in files:
                stem = os.path.splitext(f)[0]
                try:
                    idx = int(stem.split("_")[-1])
                    max_id = max(max_id, idx)
                except Exception:
                    continue
            next_id = max_id + 1 if max_id >= 0 else 0
            if next_id > 0:
                print(f"[Data] 检测到历史chunk，自动续写: next_chunk_id={next_id:05d}")
            return next_id
        except Exception as e:
            print(f"[Data] 历史chunk扫描失败，回退到00000: {e}")
            return 0

    def _load_cpg_reference(self):
        if not self.cfg.use_cpg_rewards:
            return None
        path = self.cfg.cpg_ref_joint_file
        if not os.path.isfile(path):
            print(f"[CPG] 未找到参考关节轨迹文件: {path}，关闭 CPG 奖励。")
            return None
        try:
            ref = np.loadtxt(path, delimiter=",").astype(np.float32)
            if ref.ndim != 2 or ref.shape[1] != 18:
                print(f"[CPG] 参考轨迹维度异常 {ref.shape}，关闭 CPG 奖励。")
                return None
            return ref
        except Exception as e:
            print(f"[CPG] 加载参考轨迹失败: {e}，关闭 CPG 奖励。")
            return None

    def _cpg_track_reward(self, action_sim_rad: np.ndarray, step: int) -> float:
        if self._cpg_ref is None:
            return 0.0
        idx = int(step % self._cpg_ref.shape[0])
        target_deg = self._cpg_ref[idx]
        target_sim_rad = servo_angles_to_sim_angles(target_deg).astype(np.float32)
        err = action_sim_rad - target_sim_rad
        sigma = max(1e-4, float(self.cfg.cpg_track_sigma_rad))
        mse = float(np.mean(err * err))
        return float(np.exp(-mse / (2.0 * sigma * sigma)))

    def _load_train_joint_limits(self) -> Dict[str, np.ndarray]:
        """加载训练专用关节限位（与硬件限位分离）。"""
        hw_min = np.asarray(ROBOT_CONFIG["angle_limits"]["min"], dtype=np.float32)
        hw_max = np.asarray(ROBOT_CONFIG["angle_limits"]["max"], dtype=np.float32)
        limit_file = self.cfg.train_joint_limits_file
        if os.path.isfile(limit_file):
            import json

            with open(limit_file, "r") as f:
                cfg = json.load(f)
            train_min = np.asarray(cfg.get("min", hw_min.tolist()), dtype=np.float32)
            train_max = np.asarray(cfg.get("max", hw_max.tolist()), dtype=np.float32)
            if train_min.shape != (18,) or train_max.shape != (18,):
                raise ValueError("[Safety] train_joint_limits.json 的 min/max 必须是长度18数组。")
        else:
            # 若未提供独立文件，默认在硬件限位内再收紧一层
            margin = float(self.cfg.train_limit_extra_margin_deg)
            train_min = hw_min + margin
            train_max = hw_max - margin
            train_min = np.minimum(train_min, train_max - 1.0)
            train_max = np.maximum(train_max, train_min + 1.0)
            print(
                "[Safety] 未找到训练专用限位文件，使用自动收紧限位。"
                f" file={limit_file}, extra_margin={margin:.1f} deg"
            )
        # 最终仍与硬件限位相交，避免越界
        train_min = np.maximum(train_min, hw_min)
        train_max = np.minimum(train_max, hw_max)
        return {"min": train_min, "max": train_max}

    def _infer_critic_obs_dim(self) -> int:
        critic_in = self.actor_critic.critic[0].in_features
        wm_latent_dim = self.actor_critic.critic_wm_feature_encoder[-1].out_features
        return int(critic_in - wm_latent_dim)

    def _init_log_file(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write(
                    "step\treward\tr_upright\tr_cmd\tr_smooth\tr_cpg\tv_proxy_x\tv_proxy_y\tv_proxy_z\twm_loss\tpolicy_loss\tvalue_loss\tentropy\n"
                )

    def _flush_dataset(self, force: bool = False):
        if (not force) and len(self._dataset_cache) < self.cfg.data_flush_every:
            return
        if len(self._dataset_cache) == 0:
            return

        chunk_file = os.path.join(self.dataset_dir, f"train_chunk_{self._dataset_chunk_id:05d}.npz")
        obs = np.stack([x["obs"] for x in self._dataset_cache], axis=0).astype(np.float32)
        hist = np.stack([x["history"] for x in self._dataset_cache], axis=0).astype(np.float32)
        wm_feature = np.stack([x["wm_feature"] for x in self._dataset_cache], axis=0).astype(np.float32)
        action = np.stack([x["action"] for x in self._dataset_cache], axis=0).astype(np.float32)
        reward = np.stack([x["reward"] for x in self._dataset_cache], axis=0).astype(np.float32)
        imu = np.stack([x["imu"] for x in self._dataset_cache], axis=0).astype(np.float32)
        vel_est = np.stack([x["vel_est"] for x in self._dataset_cache], axis=0).astype(np.float32)
        vel_proxy = np.stack([x["vel_proxy"] for x in self._dataset_cache], axis=0).astype(np.float32)
        wm_action = np.stack([x["wm_action"] for x in self._dataset_cache], axis=0).astype(np.float32)

        np.savez_compressed(
            chunk_file,
            obs=obs,
            history=hist,
            wm_feature=wm_feature,
            action=action,
            reward=reward,
            imu=imu,
            vel_est=vel_est,
            vel_proxy=vel_proxy,
            wm_action=wm_action,
        )
        self._dataset_cache.clear()
        self._dataset_chunk_id += 1

    def _safe_shutdown(self):
        self.running = False
        self._flush_dataset(force=True)
        try:
            if self.servos is not None:
                self.servos.disable_torque(range(18))
        except Exception:
            pass
        try:
            if self.imu_process is not None:
                self.imu_process.terminate()
                self.imu_process.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self.servos is not None and hasattr(self.servos, "portHandler"):
                self.servos.portHandler.closePort()
        except Exception:
            pass

    def _signal_handler(self, signum, frame):
        print("\n[Sim2Real] 收到中断信号，执行安全停机...")
        self._safe_shutdown()
        raise KeyboardInterrupt

    def _setup_hardware(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.servos = Servos()
        voltage = float(self.servos.read_voltage(1))
        if voltage < self.cfg.min_safe_voltage:
            raise RuntimeError(
                f"[Safety] 电压过低，当前 {voltage:.2f}V < 最低安全电压 {self.cfg.min_safe_voltage:.2f}V"
            )
        self.servos.set_position_control()
        self.servos.enable_torque(range(18))

        self.q_imu = Queue()
        self.imu_process = Process(target=read_imu, args=(self.q_imu,))
        self.imu_process.daemon = True
        self.imu_process.start()

        time.sleep(0.5)
        neutral_angles = ROBOT_CONFIG["neutral_angles"]
        self.servos.Robot_initialize(neutral_angles)
        time.sleep(0.8)

    def _build_policy_features(self, obs_prop_t: torch.Tensor, history_t: torch.Tensor, wm_feature_t: torch.Tensor):
        latent = self.actor_critic.history_encoder(history_t)
        cmd = obs_prop_t[:, 6:9]
        wm_latent = self.actor_critic.wm_feature_encoder(wm_feature_t)
        return torch.cat((latent, cmd, wm_latent), dim=-1)

    def _compute_reward(
        self,
        obs_prop: np.ndarray,
        prev_action: np.ndarray,
        action: np.ndarray,
        vel_proxy: np.ndarray,
        step: int,
    ) -> Dict[str, float]:
        base_ang = obs_prop[0:3] / 0.015
        gravity = obs_prop[3:6]
        command = obs_prop[6:9]

        upright = float(np.exp(-3.0 * (gravity[0] ** 2 + gravity[1] ** 2)))
        # 命令追踪：vx/vy 使用估计速度，yaw 使用角速度
        # 机器人坐标系约定：
        # - y 轴前进方向
        # - x 轴正向在前进方向右侧
        # 且 command 三维为 [x, y, yaw]，因此 command[0] 对齐 vel_est_x，command[1] 对齐 vel_est_y。
        vel_track = float(np.exp(-2.5 * ((vel_proxy[0] - command[0]) ** 2 + (vel_proxy[1] - command[1]) ** 2)))
        yaw_track = float(np.exp(-2.0 * (base_ang[2] - command[2]) ** 2))
        smooth = float(np.exp(-5.0 * np.mean((action - prev_action) ** 2)))
        ang_stability = float(np.exp(-0.1 * np.sum(base_ang ** 2)))
        cpg_track = self._cpg_track_reward(action, step) if self.cfg.use_cpg_rewards else 0.0
        total = (
            self.cfg.reward_w_upright * upright
            + self.cfg.reward_w_vel_track * vel_track
            + self.cfg.reward_w_yaw_track * yaw_track
            + self.cfg.reward_w_smooth * smooth
            + self.cfg.reward_w_ang_stability * ang_stability
            + self.cfg.reward_w_cpg_track * cpg_track
        )
        return {
            "reward": total,
            "r_upright": upright,
            "r_cmd": 0.5 * (vel_track + yaw_track),
            "r_smooth": smooth,
            "r_cpg": cpg_track,
        }

    def _maybe_switch_stage(self, step: int, wm_loss: float):
        if not self.cfg.auto_stage_switch:
            return
        if self.cfg.stage != "wm_only":
            return
        if wm_loss <= 0.0:
            return
        if self._wm_loss_ema is None:
            self._wm_loss_ema = wm_loss
        else:
            self._wm_loss_ema = 0.9 * self._wm_loss_ema + 0.1 * wm_loss
        if step >= self.cfg.switch_min_steps and self._wm_loss_ema < self.cfg.switch_wm_loss_threshold:
            self.cfg.stage = "policy_finetune"
            print(
                f"[Sim2Real] 自动阶段转换触发: step={step}, wm_loss_ema={self._wm_loss_ema:.4f}, "
                "stage -> policy_finetune"
            )

    def _apply_battery_mode(self, step: int):
        if not self.cfg.battery_mode:
            return
        self._battery_step_in_cycle += 1
        if self._battery_step_in_cycle < self.cfg.battery_run_steps:
            return

        self._battery_cycle += 1
        self._battery_step_in_cycle = 0
        self._save_checkpoint(step)
        print(
            f"[BatteryMode] 完成第 {self._battery_cycle} 个训练周期，进入休息 {self.cfg.battery_rest_seconds}s ..."
        )
        self.servos.Robot_initialize(ROBOT_CONFIG["neutral_angles"])
        time.sleep(max(0, int(self.cfg.battery_rest_seconds)))
        if self._battery_cycle >= self.cfg.battery_max_cycles:
            print("[BatteryMode] 达到最大周期，提前结束训练。")
            raise StopIteration

    def _check_voltage_or_stop(self, step: int):
        if self.cfg.inference_compatible_mode:
            return
        if step % self.cfg.voltage_check_every != 0:
            return
        voltage = float(self.servos.read_voltage(1))
        if voltage < self.cfg.min_safe_voltage:
            raise RuntimeError(
                f"[Safety] 电压过低触发停机: step={step}, {voltage:.2f}V < {self.cfg.min_safe_voltage:.2f}V"
            )

    def _check_imu_or_stop(self, imu_data: np.ndarray):
        if self.cfg.inference_compatible_mode:
            return
        roll = float(imu_data[0])
        pitch = float(imu_data[1])
        gx = float(imu_data[3])
        gy = float(imu_data[4])
        gz = float(imu_data[5])
        gyro_peak = max(abs(gx), abs(gy), abs(gz))
        if abs(roll) > self.cfg.imu_roll_pitch_stop_deg or abs(pitch) > self.cfg.imu_roll_pitch_stop_deg:
            raise RuntimeError(
                f"[Safety] 姿态超限: roll={roll:.1f}, pitch={pitch:.1f}, limit={self.cfg.imu_roll_pitch_stop_deg:.1f} deg"
            )
        if gyro_peak > self.cfg.imu_gyro_stop_dps:
            raise RuntimeError(
                f"[Safety] 角速度超限: peak={gyro_peak:.1f} dps, limit={self.cfg.imu_gyro_stop_dps:.1f} dps"
            )

    def _safe_joint_limits_with_margin(self, angles_deg: np.ndarray) -> np.ndarray:
        if self.cfg.inference_compatible_mode:
            # 与 test_rwm_real_robot.py 更一致：只按硬件限位，不加训练软边界
            clipped = angles_deg.copy()
            for i in range(18):
                lo = float(ROBOT_CONFIG["angle_limits"]["min"][i])
                hi = float(ROBOT_CONFIG["angle_limits"]["max"][i])
                clipped[i] = np.clip(clipped[i], lo, hi)
            return clipped
        clipped = angles_deg.copy()
        margin = float(self.cfg.soft_limit_margin_deg)
        for i in range(18):
            # 训练专用限位 + 软边界缓冲
            lo = float(self.train_angle_limits["min"][i]) + margin
            hi = float(self.train_angle_limits["max"][i]) - margin
            if lo > hi:
                lo = float(self.train_angle_limits["min"][i])
                hi = float(self.train_angle_limits["max"][i])
            clipped[i] = np.clip(clipped[i], lo, hi)
        return clipped

    def _strict_validate_joint_limits(self, angles_deg: np.ndarray):
        if self.cfg.inference_compatible_mode:
            return
        lo = self.train_angle_limits["min"]
        hi = self.train_angle_limits["max"]
        violation = np.where((angles_deg < lo) | (angles_deg > hi))[0]
        if len(violation) > 0:
            detail = ", ".join(
                [
                    f"j{int(i)}={angles_deg[i]:.2f} not in [{lo[i]:.2f},{hi[i]:.2f}]"
                    for i in violation[:6]
                ]
            )
            raise RuntimeError(f"[Safety] 训练关节限位校验失败，拒绝执行动作: {detail}")

    def _rate_limit_action(self, action_now: np.ndarray, action_prev: np.ndarray) -> np.ndarray:
        if self.cfg.inference_compatible_mode:
            return action_now
        max_delta = float(self.cfg.max_action_delta_rad)
        delta = np.clip(action_now - action_prev, -max_delta, max_delta)
        return action_prev + delta

    def _sample_action(self, obs_prop_t: torch.Tensor, history_t: torch.Tensor, wm_feature_t: torch.Tensor):
        feat = self._build_policy_features(obs_prop_t, history_t, wm_feature_t)
        self.actor_critic.update_distribution(feat)
        action = self.actor_critic.distribution.sample()
        log_prob = self.actor_critic.get_actions_log_prob(action)
        entropy = self.actor_critic.entropy.mean()
        return action, log_prob, entropy

    def _predict_linear_velocity_proxy(self, obs_prop_t: torch.Tensor, history_t: torch.Tensor) -> np.ndarray:
        """使用 sim 训练得到的 vel_predict 分支作为实机速度代理（非 IMU 积分）。"""
        with torch.no_grad():
            pred = self.actor_critic.get_linear_vel(obs_prop_t, history_t)
        return pred.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _build_velocity_proxy(self, obs_t: torch.Tensor, hist_t: torch.Tensor, imu_data: np.ndarray, action_limited: np.ndarray) -> np.ndarray:
        """Build configurable velocity proxy and apply optional axis sign calibration."""
        model_v = self._predict_linear_velocity_proxy(obs_t, hist_t).astype(np.float32)
        source = self.cfg.vel_proxy_source
        if source == "model":
            v = model_v
        else:
            ang_vel_norm = float(np.linalg.norm(obs_t[0, 0:3].detach().cpu().numpy() / 0.015))
            action_norm = float(np.linalg.norm(action_limited))
            imu_v = self.vel_estimator.update(
                imu_data,
                action_norm=action_norm,
                ang_vel_norm=ang_vel_norm,
            ).astype(np.float32)
            if source == "imu":
                v = imu_v
            else:
                a = float(np.clip(self.cfg.vel_proxy_blend_alpha, 0.0, 1.0))
                v = (a * model_v + (1.0 - a) * imu_v).astype(np.float32)
        signs = np.array(
            [self.cfg.vel_proxy_sign_x, self.cfg.vel_proxy_sign_y, self.cfg.vel_proxy_sign_z],
            dtype=np.float32,
        )
        return (v * signs).astype(np.float32)

    def _get_reward_velocity_proxy(self, vel_proxy_raw: np.ndarray) -> np.ndarray:
        """
        Use moving-average velocity for reward calculation to reduce high-frequency noise.
        This follows the same intent as averaged velocity rewards in simulation code.
        """
        self._vel_proxy_ma_buf.append(np.asarray(vel_proxy_raw, dtype=np.float32).copy())
        if len(self._vel_proxy_ma_buf) <= 1:
            return np.asarray(vel_proxy_raw, dtype=np.float32)
        return np.mean(np.stack(list(self._vel_proxy_ma_buf), axis=0), axis=0).astype(np.float32)

    def _estimate_value(self, wm_feature_t: torch.Tensor):
        critic_obs = torch.zeros((1, self.critic_obs_dim), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            value = self.actor_critic.evaluate(critic_obs, wm_feature_t)
        return value.squeeze(-1)

    def _append_rollout(self, transition: Dict[str, np.ndarray]):
        self.rollout.append(transition)
        if len(self.rollout) > self.cfg.update_horizon:
            self.rollout.pop(0)

    def _append_wm(self, wm_item: Dict[str, np.ndarray]):
        self.wm_buffer.append(wm_item)
        max_keep = max(4000, self.cfg.wm_batch_size * self.cfg.wm_batch_length * 10)
        if len(self.wm_buffer) > max_keep:
            self.wm_buffer = self.wm_buffer[-max_keep:]

    def _train_world_model(self) -> float:
        if self.rwm.world_model is None:
            return 0.0
        if len(self.wm_buffer) < self.cfg.wm_batch_size * self.cfg.wm_batch_length:
            return 0.0

        wm = self.rwm.world_model
        wm.train()
        losses = []

        def wm_policy_action(prop, history, wm_feature, **kwargs):
            """
            Homomorphic actor path for WM training.
            Mirror inference logic (history_encoder + command + wm_feature_encoder + actor),
            and always output [B, T, 18] without relying on rollout-time act().
            """
            # prop/history/wm_feature are expected [B, T, D] from WorldModelRWM._train
            b, t, _ = prop.shape
            prop2 = prop.reshape(b * t, -1)
            hist2 = history.reshape(b * t, -1)
            wm2 = wm_feature.reshape(b * t, -1)

            with torch.no_grad():
                latent = self.actor_critic.history_encoder(hist2)
                cmd = prop2[:, 6:9]
                wm_lat = self.actor_critic.wm_feature_encoder(wm2)
                actor_in = torch.cat((latent, cmd, wm_lat), dim=-1)
                act = self.actor_critic.actor(actor_in)
            return act.reshape(b, t, -1)

        for _ in range(self.cfg.wm_grad_steps):
            starts = np.random.randint(0, len(self.wm_buffer) - self.cfg.wm_batch_length, size=self.cfg.wm_batch_size)
            batch_prop = []
            batch_hist = []
            batch_action = []
            batch_reward = []
            batch_first = []
            for s in starts:
                seg = self.wm_buffer[s : s + self.cfg.wm_batch_length]
                batch_prop.append(np.stack([x["prop"] for x in seg], axis=0).astype(np.float32))
                batch_hist.append(np.stack([x["history"] for x in seg], axis=0).astype(np.float32))
                batch_action.append(np.stack([x["wm_action"] for x in seg], axis=0).astype(np.float32))
                # 同构到 world_model/rwm_runner.py 的形状契约：
                # is_first/reward 在 encoder/reward head 中应为 [B, T]（不要引入多余末维 1）
                r = np.stack([x["reward"] for x in seg], axis=0).astype(np.float32)  # [T]
                f = np.stack([x["is_first"] for x in seg], axis=0).astype(np.float32)  # [T]
                batch_reward.append(r)
                batch_first.append(f)

            data = {
                "prop": np.stack(batch_prop, axis=0),
                "history": np.stack(batch_hist, axis=0),
                "action": np.stack(batch_action, axis=0),
                "reward": np.stack(batch_reward, axis=0),
                "is_first": np.stack(batch_first, axis=0),
            }
            if not self._wm_shape_debug_printed:
                print(
                    "[WM] batch shapes: "
                    f"prop={data['prop'].shape}, history={data['history'].shape}, "
                    f"action={data['action'].shape}, reward={data['reward'].shape}, "
                    f"is_first={data['is_first'].shape}"
                )
                self._wm_shape_debug_printed = True
            try:
                # 实机 33 维本体观测下，优先使用 replay action 训练 WM，避免 prop_dim 断言不匹配。
                act_func = None if self.cfg.wm_use_replay_action else wm_policy_action
                _, _, metrics = wm._train(data, act_func=act_func)
                # tools.Optimizer(name="model") returns key "model_loss"
                # Keep robust fallback for future refactors.
                wm_loss_scalar = float(
                    np.mean(metrics.get("model_loss", metrics.get("loss", 0.0)))
                )
                losses.append(wm_loss_scalar)
            except Exception as e:
                print(f"[WM] 训练失败，跳过本次: {e}")
                print(traceback.format_exc())
                return 0.0
        self.actor_critic.train()
        return float(np.mean(losses)) if losses else 0.0

    def _train_policy(self):
        if len(self.rollout) < self.cfg.update_horizon:
            return 0.0, 0.0, 0.0

        obs = torch.tensor(np.stack([x["obs"] for x in self.rollout]), dtype=torch.float32, device=self.device)
        hist = torch.tensor(np.stack([x["history"] for x in self.rollout]), dtype=torch.float32, device=self.device)
        wm_feat = torch.tensor(np.stack([x["wm_feature"] for x in self.rollout]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack([x["action"] for x in self.rollout]), dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(np.stack([x["logp"] for x in self.rollout]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.stack([x["reward"] for x in self.rollout]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.stack([x["done"] for x in self.rollout]), dtype=torch.float32, device=self.device)
        values = torch.tensor(np.stack([x["value"] for x in self.rollout]), dtype=torch.float32, device=self.device)
        vel_target = torch.tensor(np.stack([x["vel_proxy"] for x in self.rollout]), dtype=torch.float32, device=self.device)

        returns = torch.zeros_like(rewards)
        adv = torch.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.cfg.gamma * next_value * non_terminal - values[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * non_terminal * gae
            adv[t] = gae
            returns[t] = adv[t] + values[t]
            next_value = values[t]

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        pol_losses, val_losses, ents = [], [], []

        for _ in range(self.cfg.num_epochs):
            np.random.shuffle(idx)
            for st in range(0, n, self.cfg.minibatch_size):
                mb = idx[st : st + self.cfg.minibatch_size]
                obs_b, hist_b, wm_b = obs[mb], hist[mb], wm_feat[mb]
                act_b, old_logp_b = actions[mb], old_logp[mb]
                ret_b, adv_b = returns[mb], adv[mb]

                feat = self._build_policy_features(obs_b, hist_b, wm_b)
                self.actor_critic.update_distribution(feat)
                new_logp = self.actor_critic.get_actions_log_prob(act_b)
                entropy = self.actor_critic.entropy.mean()

                ratio = torch.exp(new_logp - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                critic_obs = torch.zeros((len(mb), self.critic_obs_dim), dtype=torch.float32, device=self.device)
                value_pred = self.actor_critic.evaluate(critic_obs, wm_b).squeeze(-1)
                value_loss = F.mse_loss(value_pred, ret_b)
                pred_vel = self.actor_critic.get_linear_vel(obs_b, hist_b)
                vel_loss = F.mse_loss(pred_vel, vel_target[mb])

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    + self.cfg.vel_predict_coef * vel_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                pol_losses.append(float(policy_loss.item()))
                val_losses.append(float(value_loss.item()))
                ents.append(float(entropy.item()))

        self.rollout.clear()
        return float(np.mean(pol_losses)), float(np.mean(val_losses)), float(np.mean(ents))

    def _save_checkpoint(self, step: int):
        # Keep a checkpoint format compatible with `RealRobotRWMInference`:
        # - model_state_dict: ActorCriticRWM.state_dict()
        # - world_model_dict: WorldModelRWM.state_dict()
        ckpt = {
            "iter": int(step),
            "step": int(step),  # backward compat
            "model_state_dict": self.actor_critic.state_dict(),
            "actor_critic": self.actor_critic.state_dict(),  # backward compat
            "optimizer": self.optimizer.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rwm.world_model is not None:
            wm_sd = self.rwm.world_model.state_dict()
            ckpt["world_model_dict"] = wm_sd
            ckpt["world_model"] = wm_sd  # backward compat
        path = os.path.join(self.cfg.output_dir, f"sim2real_step_{step}.pt")
        torch.save(ckpt, path)
        print(f"[Sim2Real] 保存检查点: {path}")

    def run(self):
        self._setup_hardware()
        prev_action_for_obs = np.zeros(18, dtype=np.float32)
        executed_prev_action = np.zeros(18, dtype=np.float32)
        self.vel_estimator.reset()

        print(f"[Sim2Real] 开始训练: stage={self.cfg.stage}, total_steps={self.cfg.total_steps}")
        print(
            f"[Sim2Real] 使用命令覆盖: cmd=[{self.cfg.cmd_x:.3f}, {self.cfg.cmd_y:.3f}, {self.cfg.cmd_yaw:.3f}]"
        )
        if self.cfg.inference_compatible_mode:
            print("[Sim2Real] inference-compatible-mode=ON: 关闭训练侧额外安全限制（用于和 test_rwm_real_robot.py 对照）")
        # 关键性能点：循环内不要反复 open/close 日志文件
        log_f = open(self.log_path, "a", buffering=1)
        for step in range(self.cfg.total_steps):
            # For interrupt-resume: record last reached step index.
            self._last_step = int(step)
            t0 = time.perf_counter()
            try:
                self._check_voltage_or_stop(step)
                obs_prop, obs_wo_cmd, position_read, imu_data = create_observation_from_real_robot(
                    self.servos,
                    self.q_imu,
                    step,
                    self.history_length,
                    cpg_reward,
                    prev_action_for_obs,
                    # Online training may stall the main loop; use short IMU timeout.
                    imu_timeout_sec=0.05,
                    # Limit draining to avoid latency spikes when the IMU queue
                    # accumulates during WM/policy compute.
                    imu_drain_max=2,
                    # Disable 10s IMU re-init during online training to avoid
                    # observation jumps that can cause physical pauses.
                    imu_reinit_period_sec=None,
                )
                self._check_imu_or_stop(imu_data)
                self._consecutive_failures = 0
            except Exception as e:
                self._consecutive_failures += 1
                print(f"[Safety] 传感或安全检查失败({self._consecutive_failures}): {e}")
                # 失败时发送最近安全动作，避免突然失控
                try:
                    safe_angles = action_to_servo_angles(self._last_safe_action)
                    safe_angles = self._safe_joint_limits_with_margin(safe_angles)
                    self.servos.write_all_positions(angles_to_ticks(safe_angles))
                except Exception:
                    pass
                if self._consecutive_failures >= self.cfg.max_consecutive_failures:
                    raise RuntimeError("[Safety] 连续失败过多，执行急停。")
                time.sleep(0.05)
                continue

            self.trajectory_history.append(obs_wo_cmd.astype(np.float32))
            history_flat = np.concatenate(list(self.trajectory_history), axis=0)

            obs_t = torch.tensor(obs_prop, dtype=torch.float32, device=self.device).unsqueeze(0)
            # 覆盖命令，避免 create_observation_from_real_robot 内固定小命令导致几乎不动
            obs_t[0, 6] = float(self.cfg.cmd_x)
            obs_t[0, 7] = float(self.cfg.cmd_y)
            obs_t[0, 8] = float(self.cfg.cmd_yaw)
            obs_prop = obs_t[0].detach().cpu().numpy().astype(np.float32)
            hist_t = torch.tensor(history_flat, dtype=torch.float32, device=self.device).unsqueeze(0)

            self.rwm._prop_buffer[0].copy_(obs_t[0])
            wm_feat_t = self.rwm.update_world_model(
                {"prop": self.rwm._prop_buffer, "is_first": self.rwm.wm_is_first},
                # 注意：WM/action history 的 prev_action 必须与 test_rwm_real_robot.py 保持同构。
                # test_rwm_real_robot.py 传入的是 clip-only（scale前，且非对称映射前）的 action_for_obs。
                prev_action=prev_action_for_obs,
            )

            if self.cfg.stage == "wm_only" or step < self.cfg.warmup_steps:
                with torch.no_grad():
                    action_t = self.rwm.get_inference_policy()(obs_t, hist_t, wm_feat_t)
                    logp_t = torch.zeros(1, device=self.device)
                    entropy_t = torch.zeros(1, device=self.device)
            else:
                action_t, logp_t, entropy_t = self._sample_action(obs_t, hist_t, wm_feat_t)

            # 与 test_rwm_real_robot.py 保持一致：两套动作空间
            # 1) action_for_obs：仅做 clip（scale前，且非对称映射前），用于 history/WM prev_action
            # 2) action_limited：scale + 非对称映射 + clip（最终用于舵机执行/平滑约束等）
            action_raw_np = action_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
            action_for_obs = np.clip(
                action_raw_np,
                self.action_limits["min"],
                self.action_limits["max"],
            ).astype(np.float32)

            # scale（逐维）之后用于执行
            action_np = action_raw_np * ACTION_SCALE_PER_DIM * float(self.cfg.action_scale_multiplier)

            if USE_ASYMMETRIC_ANKLE_MAPPING:
                action_np = apply_asymmetric_ankle_mapping_rad(
                    action_np,
                    lift_range_rad=ASYM_ANKLE_LIFT_RANGE_RAD,
                    sink_range_rad=ASYM_ANKLE_SINK_RANGE_RAD,
                )

            # 启动阶段渐进放大，降低初始突发动作（兼容模式下关闭）
            ramp = 1.0 if self.cfg.inference_compatible_mode else min(
                1.0, float(step + 1) / max(1.0, float(self.cfg.startup_ramp_steps))
            )
            action_np *= ramp
            action_limited = np.clip(action_np, self.action_limits["min"], self.action_limits["max"])
            # 动作变化率限制，防止单步冲击
            action_limited = self._rate_limit_action(action_limited, executed_prev_action)
            real_angles = action_to_servo_angles(action_limited)
            # 用软限位缓冲区而非硬贴边
            real_angles = self._safe_joint_limits_with_margin(real_angles)
            # 强校验：不在训练限位内则拒绝执行
            self._strict_validate_joint_limits(real_angles)
            ticks = angles_to_ticks(real_angles)
            try:
                self.servos.write_all_positions(ticks)
                # 回写为“真正执行的动作”（由 clip 后角度反算），保证下一步仍在安全域内
                executed_action = servo_angles_to_sim_angles(real_angles).astype(np.float32)
                action_limited = executed_action
                self._last_safe_action = action_limited.copy()
            except Exception as e:
                self._consecutive_failures += 1
                print(f"[Safety] 舵机写入失败({self._consecutive_failures}): {e}")
                if self._consecutive_failures >= self.cfg.max_consecutive_failures:
                    raise RuntimeError("[Safety] 舵机连续写入失败，执行急停。")
                continue

            # 奖励与监督使用可配置速度代理（model/imu/blend + 轴向校准）。
            vel_proxy = self._build_velocity_proxy(obs_t, hist_t, imu_data, action_limited)
            vel_proxy_reward = self._get_reward_velocity_proxy(vel_proxy)
            reward_terms = self._compute_reward(
                obs_prop, executed_prev_action, action_limited, vel_proxy_reward, step
            )
            reward = reward_terms["reward"]

            value_t = self._estimate_value(wm_feat_t)
            done = 0.0

            self._append_rollout(
                {
                    "obs": obs_prop.astype(np.float32),
                    "history": history_flat.astype(np.float32),
                    "wm_feature": wm_feat_t.detach().cpu().numpy().reshape(-1).astype(np.float32),
                    "action": action_limited.astype(np.float32),
                    "logp": float(logp_t.item()),
                    "value": float(value_t.item()),
                    "reward": float(reward),
                    "done": done,
                    "vel_proxy": vel_proxy.astype(np.float32),
                }
            )
            self._append_wm(
                {
                    "prop": obs_prop.astype(np.float32),
                    "history": history_flat.astype(np.float32),
                    "wm_action": self.rwm.wm_action.detach().cpu().numpy().reshape(-1).astype(np.float32),
                    "reward": np.array(float(reward), dtype=np.float32),
                    "is_first": np.array(1.0 if step == 0 else 0.0, dtype=np.float32),
                }
            )
            self._dataset_cache.append(
                {
                    "obs": obs_prop.astype(np.float32),
                    "history": history_flat.astype(np.float32),
                    "wm_feature": wm_feat_t.detach().cpu().numpy().reshape(-1).astype(np.float32),
                    "action": action_limited.astype(np.float32),
                    "reward": np.array(float(reward), dtype=np.float32),
                    "imu": imu_data.astype(np.float32),
                    "vel_est": self.vel_estimator.v.copy().astype(np.float32),
                    "vel_proxy": vel_proxy.astype(np.float32),
                    "wm_action": self.rwm.wm_action.detach().cpu().numpy().reshape(-1).astype(np.float32),
                }
            )
            self._flush_dataset(force=False)

            wm_loss = 0.0
            if step > 0 and (step % self.cfg.wm_train_every == 0):
                wm_loss = self._train_world_model()
                # 直接在 WM 训练触发点打印，保证每次训练都能看到真实 wm_loss
                print(f"[WM] step={step} wm_loss={wm_loss:.6f} (wm_train_every={self.cfg.wm_train_every})")
            self._maybe_switch_stage(step, wm_loss)

            pol_loss, val_loss, ent = self._last_pol_loss, self._last_val_loss, self._last_entropy
            if self.cfg.stage == "policy_finetune" and step > self.cfg.warmup_steps and (
                (step + 1) % self.cfg.update_horizon == 0
            ):
                pol_loss, val_loss, ent = self._train_policy()
                self._last_pol_loss, self._last_val_loss, self._last_entropy = pol_loss, val_loss, ent
                print(
                    f"[POL] step={step} policy_loss={pol_loss:.6f} "
                    f"value_loss={val_loss:.6f} entropy={ent:.6f}"
                )

            log_f.write(
                f"{step}\t{reward:.6f}\t{reward_terms['r_upright']:.6f}\t{reward_terms['r_cmd']:.6f}\t"
                f"{reward_terms['r_smooth']:.6f}\t{reward_terms['r_cpg']:.6f}\t"
                f"{vel_proxy[0]:.6f}\t{vel_proxy[1]:.6f}\t{vel_proxy[2]:.6f}\t"
                f"{wm_loss:.6f}\t{pol_loss:.6f}\t{val_loss:.6f}\t{ent:.6f}\n"
            )
            if step % self.cfg.log_every == 0:
                log_f.flush()

            if step % self.cfg.log_every == 0:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                print(
                    f"[Step {step:05d}] R={reward:.3f} upright={reward_terms['r_upright']:.3f} "
                    f"cpg={reward_terms['r_cpg']:.3f} "
                    f"v_proxy_raw=({vel_proxy[0]:.2f},{vel_proxy[1]:.2f},{vel_proxy[2]:.2f}) "
                    f"v_proxy_ma=({vel_proxy_reward[0]:.2f},{vel_proxy_reward[1]:.2f},{vel_proxy_reward[2]:.2f}) "
                    f"wm_loss={wm_loss:.3f} pol={pol_loss:.3f} val={val_loss:.3f} loop={dt_ms:.1f}ms"
                )

            if step > 0 and (step % self.cfg.save_every == 0):
                self._save_checkpoint(step)

            # history/WM prev_action：用 action_for_obs
            prev_action_for_obs = action_for_obs.copy()
            # 舵机执行与平滑约束：用 action_limited
            executed_prev_action = action_limited.copy()
            self._apply_battery_mode(step)

            dt = time.perf_counter() - t0
            if dt < TARGET_DT:
                time.sleep(TARGET_DT - dt)
        log_f.close()

        self._save_checkpoint(self.cfg.total_steps)
        self._safe_shutdown()


def parse_args():
    p = argparse.ArgumentParser("Real robot sim2real training")
    p.add_argument("--model-path", type=str, required=True, help="已训练模型checkpoint路径")
    p.add_argument("--stage", type=str, default="wm_only", choices=["wm_only", "policy_finetune"])
    p.add_argument("--total-steps", type=int, default=1536)#2048
    p.add_argument("--update-horizon", type=int, default=256)
    p.add_argument("--policy-lr", type=float, default=3e-5)
    p.add_argument("--wm-train-every", type=int, default=256)#512
    p.add_argument("--wm-grad-steps", type=int, default=2)
    p.add_argument("--wm-use-replay-action", dest="wm_use_replay_action", action="store_true")
    p.add_argument("--wm-use-policy-action", dest="wm_use_replay_action", action="store_false")
    p.set_defaults(wm_use_replay_action=False)
    p.add_argument("--vel-predict-coef", type=float, default=0.5)
    p.add_argument("--action-scale-multiplier", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=1024)
    p.add_argument("--data-flush-every", type=int, default=200)
    p.add_argument("--auto-stage-switch", action="store_true")
    p.add_argument("--switch-min-steps", type=int, default=1200)
    p.add_argument("--switch-wm-loss-threshold", type=float, default=0.25)
    p.add_argument("--battery-mode", action="store_true")
    p.add_argument("--battery-run-steps", type=int, default=1200)
    p.add_argument("--battery-rest-seconds", type=int, default=180)
    p.add_argument("--battery-max-cycles", type=int, default=4)
    p.add_argument("--soft-limit-margin-deg", type=float, default=4.0)
    p.add_argument("--train-limit-extra-margin-deg", type=float, default=8.0)
    p.add_argument("--train-joint-limits-file", type=str, default="real_robot_sim2real/train_joint_limits.json")
    p.add_argument("--max-action-delta-rad", type=float, default=0.15)#0.06
    p.add_argument("--startup-ramp-steps", type=int, default=2)#200
    p.add_argument("--imu-roll-pitch-stop-deg", type=float, default=38.0)
    p.add_argument("--imu-gyro-stop-dps", type=float, default=550.0)
    p.add_argument("--voltage-check-every", type=int, default=100)
    p.add_argument("--min-safe-voltage", type=float, default=10.8)
    p.add_argument("--max-consecutive-failures", type=int, default=5)
    p.add_argument("--use-cpg-rewards", action="store_true")
    p.add_argument("--disable-cpg-rewards", dest="use_cpg_rewards", action="store_false")
    p.set_defaults(use_cpg_rewards=True)
    p.add_argument("--cpg-ref-joint-file", type=str, default="world_model/joint_angles_tetrapod.csv")
    p.add_argument("--cpg-track-sigma-rad", type=float, default=0.35)
    p.add_argument("--reward-w-upright", type=float, default=0.26)
    p.add_argument("--reward-w-vel-track", type=float, default=0.24)
    p.add_argument("--reward-w-yaw-track", type=float, default=0.18)
    p.add_argument("--reward-w-smooth", type=float, default=0.12)
    p.add_argument("--reward-w-ang-stability", type=float, default=0.10)
    p.add_argument("--reward-w-cpg-track", type=float, default=0.10)
    p.add_argument("--cmd-x", type=float, default=0.0)
    p.add_argument("--cmd-y", type=float, default=0.1)
    p.add_argument("--cmd-yaw", type=float, default=1.57)
    p.add_argument("--inference-compatible-mode", action="store_true")
    p.add_argument("--enable-torch-compile", action="store_true", help="启用torch.compile（CPU上可能更卡）")
    p.add_argument("--vel-proxy-source", type=str, default="model", choices=["model", "imu", "blend"])
    p.add_argument("--vel-proxy-blend-alpha", type=float, default=0.8)
    p.add_argument("--vel-proxy-sign-x", type=float, default=1.0)
    p.add_argument("--vel-proxy-sign-y", type=float, default=1.0)
    p.add_argument("--vel-proxy-sign-z", type=float, default=1.0)
    p.add_argument("--vel-proxy-reward-ma-window", type=int, default=1, help="奖励计算使用vel_proxy滑动平均窗口(1表示不平均)")
    p.add_argument("--output-dir", type=str, default="real_robot_sim2real/outputs")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        stage=args.stage,
        total_steps=args.total_steps,
        update_horizon=args.update_horizon,
        policy_lr=args.policy_lr,
        action_scale_multiplier=args.action_scale_multiplier,
        wm_train_every=args.wm_train_every,
        wm_grad_steps=args.wm_grad_steps,
        wm_use_replay_action=args.wm_use_replay_action,
        vel_predict_coef=args.vel_predict_coef,
        save_every=args.save_every,
        data_flush_every=args.data_flush_every,
        auto_stage_switch=args.auto_stage_switch,
        switch_min_steps=args.switch_min_steps,
        switch_wm_loss_threshold=args.switch_wm_loss_threshold,
        battery_mode=args.battery_mode,
        battery_run_steps=args.battery_run_steps,
        battery_rest_seconds=args.battery_rest_seconds,
        battery_max_cycles=args.battery_max_cycles,
        soft_limit_margin_deg=args.soft_limit_margin_deg,
        train_limit_extra_margin_deg=args.train_limit_extra_margin_deg,
        train_joint_limits_file=args.train_joint_limits_file,
        max_action_delta_rad=args.max_action_delta_rad,
        startup_ramp_steps=args.startup_ramp_steps,
        imu_roll_pitch_stop_deg=args.imu_roll_pitch_stop_deg,
        imu_gyro_stop_dps=args.imu_gyro_stop_dps,
        voltage_check_every=args.voltage_check_every,
        min_safe_voltage=args.min_safe_voltage,
        max_consecutive_failures=args.max_consecutive_failures,
        use_cpg_rewards=args.use_cpg_rewards,
        cpg_ref_joint_file=args.cpg_ref_joint_file,
        cpg_track_sigma_rad=args.cpg_track_sigma_rad,
        reward_w_upright=args.reward_w_upright,
        reward_w_vel_track=args.reward_w_vel_track,
        reward_w_yaw_track=args.reward_w_yaw_track,
        reward_w_smooth=args.reward_w_smooth,
        reward_w_ang_stability=args.reward_w_ang_stability,
        reward_w_cpg_track=args.reward_w_cpg_track,
        cmd_x=args.cmd_x,
        cmd_y=args.cmd_y,
        cmd_yaw=args.cmd_yaw,
        inference_compatible_mode=args.inference_compatible_mode,
        enable_torch_compile=args.enable_torch_compile,
        vel_proxy_source=args.vel_proxy_source,
        vel_proxy_blend_alpha=args.vel_proxy_blend_alpha,
        vel_proxy_sign_x=args.vel_proxy_sign_x,
        vel_proxy_sign_y=args.vel_proxy_sign_y,
        vel_proxy_sign_z=args.vel_proxy_sign_z,
        vel_proxy_reward_ma_window=args.vel_proxy_reward_ma_window,
        output_dir=args.output_dir,
    )
    trainer = RealRobotTrainer(model_path=args.model_path, cfg=cfg)
    try:
        trainer.run()
    except KeyboardInterrupt:
        # `SIGINT` may already call safe_shutdown via the signal handler,
        # but we still want to persist a checkpoint for interrupt-resume.
        try:
            trainer._save_checkpoint(getattr(trainer, "_last_step", 0))
        except Exception as e:
            print(f"[Sim2Real] KeyboardInterrupt checkpoint保存失败: {e}")
        trainer._safe_shutdown()
    except StopIteration:
        trainer._safe_shutdown()


if __name__ == "__main__":
    main()
