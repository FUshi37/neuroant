import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Dict, List

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import test_rwm_real_robot as rrm
from test_rwm_real_robot import (
    ACTION_SCALE_PER_DIM,
    ASYM_ANKLE_LIFT_RANGE_RAD,
    ASYM_ANKLE_SINK_RANGE_RAD,
    ROBOT_CONFIG,
    TARGET_DT,
    USE_ASYMMETRIC_ANKLE_MAPPING,
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
class CollectConfig:
    total_steps: int = 1000
    output_dir: str = "real_robot_sim2real/outputs/dataset_collect"
    log_every: int = 100
    data_flush_every: int = 200
    save_every: int = 99999999
    cmd_x: float = 0.0
    cmd_y: float = 0.5
    cmd_yaw: float = 1.57
    action_scale_multiplier: float = 1.0
    vel_proxy_source: str = "model"  # model / imu / blend
    vel_proxy_blend_alpha: float = 0.8
    vel_proxy_sign_x: float = 1.0
    vel_proxy_sign_y: float = 1.0
    vel_proxy_sign_z: float = 1.0
    use_cpg_rewards: bool = True
    cpg_ref_joint_file: str = "world_model/joint_angles_tetrapod.csv"
    cpg_track_sigma_rad: float = 0.35
    reward_w_upright: float = 0.26
    reward_w_vel_track: float = 0.24
    reward_w_yaw_track: float = 0.18
    reward_w_smooth: float = 0.12
    reward_w_ang_stability: float = 0.10
    reward_w_cpg_track: float = 0.10
    vel_proxy_reward_ma_window: int = 1
    enable_torch_compile: bool = True
    wm_update_stride: int = 5
    flush_at_end_only: bool = False
    no_storage: bool = False


class RealVelocityEstimator:
    def __init__(self, dt: float = 0.02, alpha: float = 0.85, leak: float = 0.995):
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.leak = float(leak)
        self.v = np.zeros(3, dtype=np.float32)
        self._acc_lp = np.zeros(3, dtype=np.float32)
        self._acc_bias = np.zeros(3, dtype=np.float32)

    @staticmethod
    def _rot_matrix_from_rpy(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
        return rz @ ry @ rx

    def update(self, imu_data: np.ndarray, action_norm: float, ang_vel_norm: float) -> np.ndarray:
        roll, pitch, yaw = imu_data[0], imu_data[1], imu_data[2]
        acc_body = np.array(imu_data[6:9], dtype=np.float32)
        r_bw = self._rot_matrix_from_rpy(roll, pitch, yaw)
        acc_world = r_bw @ acc_body
        acc_world[2] -= 9.81
        static_like = (action_norm < 0.03) and (ang_vel_norm < 0.25)
        if static_like:
            self._acc_bias = 0.995 * self._acc_bias + 0.005 * acc_world
        acc_world = acc_world - self._acc_bias
        self._acc_lp = self.alpha * self._acc_lp + (1.0 - self.alpha) * acc_world
        self.v = self.leak * self.v + self._acc_lp * self.dt
        if static_like:
            self.v *= 0.80
            if np.linalg.norm(self.v) < 0.08:
                self.v[:] = 0.0
        self.v = np.clip(self.v, -1.0, 1.0)
        return self.v.copy()


class Collector:
    def __init__(self, model_path: str, cfg: CollectConfig):
        self.cfg = cfg
        # Default for data collection: disable torch.compile to reduce CPU jitter.
        rrm.WM_OPT_TORCH_COMPILE = bool(self.cfg.enable_torch_compile)
        rrm.POLICY_OPT_TORCH_COMPILE = bool(self.cfg.enable_torch_compile)
        rrm.WM_OPT_COMPILE_OBS_STEP = bool(self.cfg.enable_torch_compile)
        print(
            f"[Collect] torch.compile={'ON' if self.cfg.enable_torch_compile else 'OFF'} "
            f"(wm/policy/obs_step)"
        )
        self.rwm = RealRobotRWMInference(model_path=model_path, device="cpu", remove_dof_vel=remove_dof_vel)
        self.actor_critic = self.rwm.actor_critic
        self.actor_critic.eval()

        self.servos = None
        self.q_imu = None
        self.imu_process = None
        self.running = True

        self.history_length = 5
        self.obs_without_command_dim = (42 if remove_dof_vel else 60) + (6 if cpg_reward else 0)
        self.trajectory_history = [np.zeros(self.obs_without_command_dim, dtype=np.float32) for _ in range(self.history_length)]

        self.action_limits = get_action_limits()
        self.vel_estimator = RealVelocityEstimator(dt=TARGET_DT)
        self._dataset_cache: List[Dict[str, np.ndarray]] = []
        self._cpg_ref = self._load_cpg_reference()
        self._last_wm_feat = None
        self._vel_proxy_ma_buf = []

        self.log_path = None
        self.dataset_dir = None
        self._dataset_chunk_id = 0
        if not self.cfg.no_storage:
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            self.log_path = os.path.join(self.cfg.output_dir, "collect_log.tsv")
            self.dataset_dir = os.path.join(self.cfg.output_dir, "dataset")
            os.makedirs(self.dataset_dir, exist_ok=True)
            self._dataset_chunk_id = self._infer_next_chunk_id()
            if not os.path.exists(self.log_path):
                with open(self.log_path, "w") as f:
                    f.write(
                        "step\treward\tr_upright\tr_cmd\tr_smooth\tr_cpg\t"
                        "v_proxy_x\tv_proxy_y\tv_proxy_z\tloop_ms\n"
                    )
        else:
            print("[Collect] no_storage=ON: 不写入任何日志与dataset文件")

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

    def _infer_next_chunk_id(self) -> int:
        files = [f for f in os.listdir(self.dataset_dir) if f.startswith("train_chunk_") and f.endswith(".npz")]
        max_id = -1
        for f in files:
            try:
                idx = int(os.path.splitext(f)[0].split("_")[-1])
                max_id = max(max_id, idx)
            except Exception:
                continue
        next_id = max_id + 1
        if next_id > 0:
            print(f"[Data] 检测到历史chunk，自动续写: next_chunk_id={next_id:05d}")
        return next_id

    def _flush_dataset(self, force: bool = False):
        if self.cfg.no_storage:
            return
        if self.cfg.flush_at_end_only and (not force):
            return
        if (not force) and len(self._dataset_cache) < self.cfg.data_flush_every:
            return
        if len(self._dataset_cache) == 0:
            return
        chunk_file = os.path.join(self.dataset_dir, f"train_chunk_{self._dataset_chunk_id:05d}.npz")
        np.savez_compressed(
            chunk_file,
            obs=np.stack([x["obs"] for x in self._dataset_cache], axis=0).astype(np.float32),
            history=np.stack([x["history"] for x in self._dataset_cache], axis=0).astype(np.float32),
            wm_feature=np.stack([x["wm_feature"] for x in self._dataset_cache], axis=0).astype(np.float32),
            action=np.stack([x["action"] for x in self._dataset_cache], axis=0).astype(np.float32),
            reward=np.stack([x["reward"] for x in self._dataset_cache], axis=0).astype(np.float32),
            imu=np.stack([x["imu"] for x in self._dataset_cache], axis=0).astype(np.float32),
            vel_est=np.stack([x["vel_est"] for x in self._dataset_cache], axis=0).astype(np.float32),
            vel_proxy=np.stack([x["vel_proxy"] for x in self._dataset_cache], axis=0).astype(np.float32),
            wm_action=np.stack([x["wm_action"] for x in self._dataset_cache], axis=0).astype(np.float32),
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
        print("\n[Collect] 收到中断信号，执行安全停机...")
        self.running = False
        try:
            if self.imu_process is not None:
                self.imu_process.terminate()
        except Exception:
            pass

    def _build_velocity_proxy(self, obs_t: torch.Tensor, hist_t: torch.Tensor, imu_data: np.ndarray, action_limited: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            model_v = self.actor_critic.get_linear_vel(obs_t, hist_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
        source = self.cfg.vel_proxy_source
        if source == "model":
            v = model_v
        else:
            ang_vel_norm = float(np.linalg.norm(obs_t[0, 0:3].detach().cpu().numpy() / 0.015))
            action_norm = float(np.linalg.norm(action_limited))
            imu_v = self.vel_estimator.update(imu_data, action_norm=action_norm, ang_vel_norm=ang_vel_norm).astype(np.float32)
            if source == "imu":
                v = imu_v
            else:
                a = float(np.clip(self.cfg.vel_proxy_blend_alpha, 0.0, 1.0))
                v = (a * model_v + (1.0 - a) * imu_v).astype(np.float32)
        signs = np.array([self.cfg.vel_proxy_sign_x, self.cfg.vel_proxy_sign_y, self.cfg.vel_proxy_sign_z], dtype=np.float32)
        return (v * signs).astype(np.float32)

    def _get_reward_velocity_proxy(self, vel_proxy_raw: np.ndarray) -> np.ndarray:
        win = max(1, int(self.cfg.vel_proxy_reward_ma_window))
        self._vel_proxy_ma_buf.append(np.asarray(vel_proxy_raw, dtype=np.float32).copy())
        if len(self._vel_proxy_ma_buf) > win:
            self._vel_proxy_ma_buf = self._vel_proxy_ma_buf[-win:]
        if len(self._vel_proxy_ma_buf) <= 1:
            return np.asarray(vel_proxy_raw, dtype=np.float32)
        return np.mean(np.stack(self._vel_proxy_ma_buf, axis=0), axis=0).astype(np.float32)

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

    def run(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.servos = Servos()
        self.servos.set_position_control()
        self.servos.enable_torque(range(18))
        self.q_imu = Queue()
        self.imu_process = Process(target=read_imu, args=(self.q_imu,))
        self.imu_process.daemon = True
        self.imu_process.start()
        time.sleep(0.5)
        self.servos.Robot_initialize(ROBOT_CONFIG["neutral_angles"])
        time.sleep(0.8)

        prev_action_for_obs = np.zeros(18, dtype=np.float32)
        executed_prev_action = np.zeros(18, dtype=np.float32)
        log_f = None
        if not self.cfg.no_storage:
            log_f = open(self.log_path, "a", buffering=1)
        print(f"[Collect] 开始采集: total_steps={self.cfg.total_steps}, output={self.cfg.output_dir}")
        try:
            for step in range(self.cfg.total_steps):
                if not self.running:
                    break
                t0 = time.perf_counter()
                obs_prop, obs_wo_cmd, _, imu_data = create_observation_from_real_robot(
                    self.servos,
                    self.q_imu,
                    step,
                    self.history_length,
                    cpg_reward=cpg_reward,
                    previous_actions=prev_action_for_obs,
                    imu_timeout_sec=0.05,
                    imu_drain_max=2,
                    imu_reinit_period_sec=None,
                )

                self.trajectory_history.pop(0)
                self.trajectory_history.append(obs_wo_cmd.astype(np.float32))
                history_flat = np.concatenate(self.trajectory_history, axis=0)

                obs_t = torch.tensor(obs_prop, dtype=torch.float32).unsqueeze(0)
                obs_t[0, 6] = float(self.cfg.cmd_x)
                obs_t[0, 7] = float(self.cfg.cmd_y)
                obs_t[0, 8] = float(self.cfg.cmd_yaw)
                obs_prop = obs_t[0].detach().cpu().numpy().astype(np.float32)
                hist_t = torch.tensor(history_flat, dtype=torch.float32).unsqueeze(0)

                do_wm_update = (step % max(1, int(self.cfg.wm_update_stride)) == 0) or (self._last_wm_feat is None)
                if do_wm_update:
                    self.rwm._prop_buffer[0].copy_(obs_t[0])
                    wm_feat_t = self.rwm.update_world_model(
                        {"prop": self.rwm._prop_buffer, "is_first": self.rwm.wm_is_first},
                        prev_action=prev_action_for_obs,
                    )
                    self._last_wm_feat = wm_feat_t.detach().clone()
                else:
                    wm_feat_t = self._last_wm_feat
                with torch.no_grad():
                    action_t = self.rwm.get_inference_policy()(obs_t, hist_t, wm_feat_t)

                action_raw = action_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
                action_for_obs = np.clip(action_raw, self.action_limits["min"], self.action_limits["max"]).astype(np.float32)
                action_np = action_raw * ACTION_SCALE_PER_DIM * float(self.cfg.action_scale_multiplier)
                if USE_ASYMMETRIC_ANKLE_MAPPING:
                    action_np = apply_asymmetric_ankle_mapping_rad(
                        action_np,
                        lift_range_rad=ASYM_ANKLE_LIFT_RANGE_RAD,
                        sink_range_rad=ASYM_ANKLE_SINK_RANGE_RAD,
                    )
                action_limited = np.clip(action_np, self.action_limits["min"], self.action_limits["max"])
                real_angles = action_to_servo_angles(action_limited)
                for i in range(18):
                    real_angles[i] = np.clip(
                        real_angles[i],
                        float(ROBOT_CONFIG["angle_limits"]["min"][i]),
                        float(ROBOT_CONFIG["angle_limits"]["max"][i]),
                    )
                self.servos.write_all_positions(angles_to_ticks(real_angles))
                executed_action = servo_angles_to_sim_angles(real_angles).astype(np.float32)
                action_limited = executed_action

                vel_proxy = self._build_velocity_proxy(obs_t, hist_t, imu_data, action_limited)
                vel_proxy_reward = self._get_reward_velocity_proxy(vel_proxy)
                reward_terms = self._compute_reward(
                    obs_prop, executed_prev_action, action_limited, vel_proxy_reward, step
                )
                reward = float(reward_terms["reward"])

                if not self.cfg.no_storage:
                    self._dataset_cache.append(
                        {
                            "obs": obs_prop.astype(np.float32),
                            "history": history_flat.astype(np.float32),
                            "wm_feature": wm_feat_t.detach().cpu().numpy().reshape(-1).astype(np.float32),
                            "action": action_limited.astype(np.float32),
                            "reward": np.array(reward, dtype=np.float32),
                            "imu": imu_data.astype(np.float32),
                            "vel_est": self.vel_estimator.v.copy().astype(np.float32),
                            "vel_proxy": vel_proxy.astype(np.float32),
                            "wm_action": self.rwm.wm_action.detach().cpu().numpy().reshape(-1).astype(np.float32),
                        }
                    )
                    self._flush_dataset(force=False)

                dt_ms = (time.perf_counter() - t0) * 1000.0
                if log_f is not None:
                    log_f.write(
                        f"{step}\t{reward:.6f}\t{reward_terms['r_upright']:.6f}\t{reward_terms['r_cmd']:.6f}\t"
                        f"{reward_terms['r_smooth']:.6f}\t{reward_terms['r_cpg']:.6f}\t"
                        f"{vel_proxy[0]:.6f}\t{vel_proxy[1]:.6f}\t{vel_proxy[2]:.6f}\t{dt_ms:.3f}\n"
                    )
                if step % self.cfg.log_every == 0:
                    print(
                        f"[Step {step:05d}] R={reward:.3f} upright={reward_terms['r_upright']:.3f} "
                        f"cpg={reward_terms['r_cpg']:.3f} "
                        f"v_proxy_raw=({vel_proxy[0]:.2f},{vel_proxy[1]:.2f},{vel_proxy[2]:.2f}) "
                        f"v_proxy_ma=({vel_proxy_reward[0]:.2f},{vel_proxy_reward[1]:.2f},{vel_proxy_reward[2]:.2f}) "
                        f"loop={dt_ms:.1f}ms"
                    )

                prev_action_for_obs = action_for_obs.copy()
                executed_prev_action = action_limited.copy()
                _ = executed_prev_action
                dt = time.perf_counter() - t0
                if dt < TARGET_DT:
                    time.sleep(TARGET_DT - dt)
                if not self.running:
                    break
        finally:
            if log_f is not None:
                log_f.close()
            self._safe_shutdown()


def parse_args():
    p = argparse.ArgumentParser("Collect dataset with test-like control loop")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--total-steps", type=int, default=1000)
    p.add_argument("--output-dir", type=str, default="real_robot_sim2real/outputs/dataset_collect")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--data-flush-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=99999999)  # kept for command compatibility
    p.add_argument("--cmd-x", type=float, default=0.0)
    p.add_argument("--cmd-y", type=float, default=0.5)
    p.add_argument("--cmd-yaw", type=float, default=1.57)
    p.add_argument("--action-scale-multiplier", type=float, default=1.0)
    p.add_argument("--vel-proxy-source", type=str, default="model", choices=["model", "imu", "blend"])
    p.add_argument("--vel-proxy-blend-alpha", type=float, default=0.8)
    p.add_argument("--vel-proxy-sign-x", type=float, default=1.0)
    p.add_argument("--vel-proxy-sign-y", type=float, default=1.0)
    p.add_argument("--vel-proxy-sign-z", type=float, default=1.0)
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
    p.add_argument("--vel-proxy-reward-ma-window", type=int, default=1, help="奖励计算使用vel_proxy滑动平均窗口(1表示不平均)")
    p.add_argument("--enable-torch-compile", action="store_true", help="启用torch.compile")
    p.add_argument("--disable-torch-compile", dest="enable_torch_compile", action="store_false", help="关闭torch.compile")
    p.set_defaults(enable_torch_compile=True)
    p.add_argument("--wm-update-stride", type=int, default=5, help="每N步更新一次WM，其余步复用上次wm_feature")
    p.add_argument("--flush-at-end-only", action="store_true", help="仅结束时一次性写盘（更流畅，但中断可能丢最后未保存数据）")
    p.add_argument("--no-storage", action="store_true", help="仅测试运行，不写入任何日志或dataset文件")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CollectConfig(
        total_steps=args.total_steps,
        output_dir=args.output_dir,
        log_every=args.log_every,
        data_flush_every=args.data_flush_every,
        save_every=args.save_every,
        cmd_x=args.cmd_x,
        cmd_y=args.cmd_y,
        cmd_yaw=args.cmd_yaw,
        action_scale_multiplier=args.action_scale_multiplier,
        vel_proxy_source=args.vel_proxy_source,
        vel_proxy_blend_alpha=args.vel_proxy_blend_alpha,
        vel_proxy_sign_x=args.vel_proxy_sign_x,
        vel_proxy_sign_y=args.vel_proxy_sign_y,
        vel_proxy_sign_z=args.vel_proxy_sign_z,
        use_cpg_rewards=args.use_cpg_rewards,
        cpg_ref_joint_file=args.cpg_ref_joint_file,
        cpg_track_sigma_rad=args.cpg_track_sigma_rad,
        reward_w_upright=args.reward_w_upright,
        reward_w_vel_track=args.reward_w_vel_track,
        reward_w_yaw_track=args.reward_w_yaw_track,
        reward_w_smooth=args.reward_w_smooth,
        reward_w_ang_stability=args.reward_w_ang_stability,
        reward_w_cpg_track=args.reward_w_cpg_track,
        vel_proxy_reward_ma_window=args.vel_proxy_reward_ma_window,
        enable_torch_compile=args.enable_torch_compile,
        wm_update_stride=args.wm_update_stride,
        flush_at_end_only=args.flush_at_end_only,
        no_storage=args.no_storage,
    )
    collector = Collector(args.model_path, cfg)
    collector.run()


if __name__ == "__main__":
    main()
