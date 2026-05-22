# Real robot RWM inference - Simplified for deployment
import os
import time
import math
import platform
import shutil
import torch
import numpy as np
from collections import deque
import sys
import contextlib
import csv

# Real robot hardware imports (optional on PC-side inference service)
HARDWARE_IMPORT_ERROR = None
try:
    import lib.device_model as deviceModel
    from lib.data_processor.roles.jy901s_dataProcessor import JY901SDataProcessor
    from lib.protocol_resolver.roles.wit_protocol_resolver import WitProtocolResolver
    from Servos import *
    from set_imu import *
    from utils import *
    from reflex_related import *
except Exception as _hw_e:
    HARDWARE_IMPORT_ERROR = _hw_e
from robot_config import ROBOT_CONFIG, real_to_sim_angles, sim_to_real_angles, angles_to_ticks, ticks_to_angles
from multiprocessing import Process, Queue

# Import RWM model components
from world_model.actor_cirtic_rwm import ActorCriticRWM
from deployment_safety import (
    RiskLevelEstimator,
    BadLegTracker,
    SafetyActionFilter,
    WorldModelCandidateSelector,
    RIGHT_FRONT_ANKLE_ACTION_OFFSET,
    RIGHT_FRONT_KNEE_ACTION_OFFSET,
    apply_right_front_action_offset,
    default_action_scale_per_dim,
    policy_action_to_exec_rad,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "validation_outputs")


# ==================== PORT ERROR DETECTION ====================
class PortErrorDetector:
    """Detects 'Port is in use' error and raises exception immediately"""
    def __init__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.enabled = True  # Flag to enable/disable detection
        
    def write(self, text):
        """Write to original output and check for port error"""
        self.original_stdout.write(text)
        # Only check if enabled and not during cleanup
        if self.enabled and "Port is in use" in text and "[TxRxResult]" in text:
            self.original_stdout.write("\n" + "="*60 + "\n")
            self.original_stdout.write("❌ CRITICAL: Port is in use error detected!\n")
            self.original_stdout.write("="*60 + "\n")
            self.original_stdout.flush()
            # Disable further detection to avoid cascade
            self.enabled = False
            raise RuntimeError("PORT_IN_USE_ERROR: Port is in use!")
            
    def flush(self):
        self.original_stdout.flush()
        
    def disable(self):
        """Disable error detection (for cleanup phase)"""
        self.enabled = False
        
    def enable(self):
        """Re-enable error detection"""
        self.enabled = True

# Install port error detector
port_detector = PortErrorDetector()
sys.stdout = port_detector
# ============================================================


INTERPOLATION_STEPS = 1 # Simplified: no interpolation like CPGs
TARGET_DT = 0.02  # 20ms = 50Hz like CPGs for higher control frequency
MAX_STEPS = 1000
# 策略输出逐维缩放（与 action 索引一致）：
# [0:3] l1_bc,l1_cf,l1_ft  [3:6] l2  [6:9] l3  [9:12] r1  [12:15] r2  [15:18] r3
ACTION_SCALE_PER_DIM = np.array(
    default_action_scale_per_dim(), dtype=np.float32
)
cpg_reward = True
remove_dof_vel = True  # when True: remove dof_vel(18) from observation/history
USE_ADMITTANCE = True  # Enable/disable admittance filter

# ==================== Performance / Logging Switches ====================
# 控制台打印尽量关掉，减少 I/O 对控制周期的影响；
# 时间拆段计时保留，用于定位瓶颈。
ENABLE_CONTROL_PRINT = False  # 关掉大多数运行时打印（仅保留关键错误/计时报告）
ENABLE_TIMING_REPORT = True
TIMING_REPORT_EVERY_N_STEPS = 50
LOG_FLUSH_EVERY_N_STEPS = 20  # 降低 flush 频率，减少文件 I/O 抖动
# 下面这些“每步大段打印”非常耗时，默认关闭
ENABLE_ACTION_CLIP_PRINT = False
ENABLE_JOINT_LIMIT_PRINT = False
# ======================================================================

# ==================== Contact anomaly detection switches ====================
# Toggle world-model-prior based contact anomaly detection/action correction.
ENABLE_CONTACT_ANOMALY_DETECTOR = True
ENABLE_WM_CANDIDATE_SELECTOR = True
CONTACT_ANOMALY_THRESHOLD = 0.15
CONTACT_ANOMALY_EMA_ALPHA = 0.90
CONTACT_ANOMALY_TRIGGER_COUNT = 3
CONTACT_ANOMALY_ACTION_SCALE = 0.70
CONTACT_ANOMALY_LIFT_GAIN = 0.20
CONTACT_ANOMALY_MAX_LIFT_GAIN = 0.60
# ============================================================================

# ------------------- Inference CPU tweaks (no model change) -------------------
# 树莓派上可试 1~4；None 表示不调用 set_num_threads（沿用 PyTorch 默认）
OPTIM_TORCH_NUM_THREADS = 2
# 单线程推理时常设 1，避免多线程调度开销（与 OMP 配合试）
OPTIM_TORCH_NUM_INTEROP_THREADS = 1
# 同步设置常见 BLAS/OpenMP 线程（对 numpy/部分算子有效；最好在进程早期设置）
OPTIM_ENV_OMP_THREADS = None  # 设为与 OPTIM_TORCH_NUM_THREADS 相同整数可试，例如 2

# torch.compile：WM 编码器 + RSSM obs_step + 策略子模块（需 PyTorch 2.0+）
WM_OPT_TORCH_COMPILE = True
POLICY_OPT_TORCH_COMPILE = True
WM_COMPILE_MODE = "default"  # CPU 可试 "reduce-overhead"；失败会自动回退

# Auto-disable torch.compile when no native compiler is available.
# This avoids runtime fallback-to-safe-actions on environments lacking toolchains.
if os.environ.get("DISABLE_TORCH_COMPILE", "0") == "1":
    WM_OPT_TORCH_COMPILE = False
    POLICY_OPT_TORCH_COMPILE = False
else:
    if platform.system().lower().startswith("win"):
        # On Windows, torch.compile/inductor relies on MSVC cl.exe.
        has_native_compiler = shutil.which("cl") is not None or shutil.which("cl.exe") is not None
    else:
        has_native_compiler = any(
            shutil.which(x) is not None for x in ("gcc", "g++", "cc", "clang", "clang++")
        )
    if not has_native_compiler:
        WM_OPT_TORCH_COMPILE = False
        POLICY_OPT_TORCH_COMPILE = False
        print("Warning: no native compiler detected, torch.compile disabled.")

# 仅编译 RSSM.obs_step（若 dynamo 对 dict 状态报错，可设 False 只保留 encoder 编译）
WM_OPT_COMPILE_OBS_STEP = True

# ONNX Runtime：仅替换 WM 的 MultiEncoder 前向（需: pip install onnx onnxruntime）
# 为 True 时优先用 ORT，跳过对 encoder 的 torch.compile
WM_OPT_ONNX_ENCODER = False
# None = 自动保存到 checkpoint 同目录 wm_encoder.onnx
WM_ONNX_EXPORT_PATH = None
# ==============================================================================


def apply_cpu_performance_settings(
    num_threads=None,
    num_interop_threads=None,
    env_omp_threads=None,
):
    """
    在加载大模型之前调用效果最佳。
    env_omp_threads：若给定，同步设置 OMP/MKL/OpenBLAS/NUMEXPR（当前进程内）。
    """
    if num_threads is not None:
        try:
            torch.set_num_threads(int(num_threads))
        except Exception:
            pass
    if num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(num_interop_threads))
        except Exception:
            pass
    if env_omp_threads is not None:
        v = str(int(env_omp_threads))
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ[k] = v


class EncoderInferenceWrapper(torch.nn.Module):
    """Tensor 输入，便于 torch.compile / ONNX 导出（与 MultiEncoder({'prop','is_first'}) 一致）。"""

    def __init__(self, multi_encoder):
        super().__init__()
        self.encoder = multi_encoder

    def forward(self, prop, is_first):
        return self.encoder({"prop": prop, "is_first": is_first})

# ---------------------------
# Ankle asymmetric mapping (sim-to-real consistency)
# If your policy was trained in simulation with use_asymmetric_ankle_mapping=True
# (see world_model/hexapodMBRL.py), enable this to match the same semantics on real robot.
# ---------------------------
USE_ASYMMETRIC_ANKLE_MAPPING = True
# Ranges are in radians around the neutral (0) pose in sim joint space.
# Positive direction means "lift" after alignment (left ankles: +, right ankles: -).
ASYM_ANKLE_LIFT_RANGE_RAD = 0.50#1.0
ASYM_ANKLE_SINK_RANGE_RAD = 0.05#0.10


def apply_asymmetric_ankle_mapping_rad(actions_rad, lift_range_rad=1.0, sink_range_rad=0.10):
    """
    Apply asymmetric ankle mapping like hexapodMBRL._apply_asymmetric_ankle_mapping, but in *radians*.

    We interpret `actions_rad` as desired joint offsets in sim joint space (rad) around the neutral pose.
    For the 6 ankle joints, the "lift" direction is:
      - left ankles (2,5,8): +rad
      - right ankles (11,14,17): -rad

    Mapping: large range for lift, small range for sink.
    """
    a = np.array(actions_rad, dtype=np.float32, copy=True)
    ankle_idx = np.array([2, 5, 8, 11, 14, 17], dtype=np.int64)
    # lift_dir aligns "lift" to positive
    lift_dir = np.array([1, 1, 1, -1, -1, -1], dtype=np.float32)
    lift_range_rad = float(max(lift_range_rad, 0.0))
    sink_range_rad = float(max(sink_range_rad, 0.0))
    if lift_range_rad == 0.0 and sink_range_rad == 0.0:
        return a

    ankle = a[ankle_idx]
    aligned = ankle * lift_dir
    # Normalize by lift range so +side can reach lift_range; clamp to [-1, 1]
    denom = lift_range_rad if lift_range_rad > 1e-6 else 1e-6
    aligned_n = np.clip(aligned / denom, -1.0, 1.0)
    mapped_aligned = np.where(aligned_n >= 0.0, aligned_n * lift_range_rad, aligned_n * sink_range_rad)
    mapped = mapped_aligned * lift_dir
    a[ankle_idx] = mapped
    return a


# 定义弧度转角度函数
def radians_to_degrees(radians):
    return radians * 180.0 / math.pi

def interpolate_actions(prev_actions, current_actions, alpha):
    """
    Linear interpolation between previous and current actions for smooth transitions.

    Args:
        prev_actions: Previous action array (18,)
        current_actions: Current action array (18,)
        alpha: Interpolation factor (0.0 = prev_actions, 1.0 = current_actions)

    Returns:
        interpolated_actions: Smoothed action array
    """
    if prev_actions is None:
        return current_actions
    return prev_actions + alpha * (current_actions - prev_actions)

def servo_angles_to_sim_angles(servo_angles_deg):
    """
    Convert servo angles (degrees) to simulation joint angles (radians) that match action ranges.

    Args:
        servo_angles_deg: 18-dim array of servo angles in degrees
                         Order: [l1_hip, l1_knee, l1_ankle, r1_hip, r1_knee, r1_ankle,
                                l2_hip, l2_knee, l2_ankle, r2_hip, r2_knee, r2_ankle,
                                l3_hip, l3_knee, l3_ankle, r3_hip, r3_knee, r3_ankle]

    Returns:
        sim_angles_rad: 18-dim array of simulation joint angles in radians
                        Order: [l1_hip, l1_knee, l1_ankle, l2_hip, l2_knee, l2_ankle,
                               l3_hip, l3_knee, l3_ankle, r1_hip, r1_knee, r1_ankle,
                               r2_hip, r2_knee, r2_ankle, r3_hip, r3_knee, r3_ankle]
    """
    # Initialize simulation angles array
    sim_angles_deg = np.zeros(18)

    # Reorder from servo order to simulation order and apply inverse mappings
    # Servo indices:  0-2:l1, 3-5:r1, 6-8:l2, 9-11:r2, 12-14:l3, 15-17:r3
    # Sim indices:    0-2:l1, 3-5:l2, 6-8:l3, 9-11:r1, 12-14:r2, 15-17:r3

    # l1 leg (servo 0-2 -> sim 0-2)
    # Hip joint: servo = 180 - action_deg => action_deg = 180 - servo
    sim_angles_deg[0] = 180.0 - servo_angles_deg[0]
    # Knee joint: servo = 180 - action_deg => action_deg = 180 - servo (left leg)
    sim_angles_deg[1] = 180.0 - servo_angles_deg[1]
    # Ankle joint: servo = 60 + action_deg => action_deg = servo - 60 (left leg)
    sim_angles_deg[2] = servo_angles_deg[2] - 60.0

    # l2 leg (servo 6-8 -> sim 3-5)
    # Hip joint: servo = 180 - action_deg => action_deg = 180 - servo
    sim_angles_deg[3] = 180.0 - servo_angles_deg[6]
    # Knee joint: servo = 180 - action_deg => action_deg = 180 - servo (left leg)
    sim_angles_deg[4] = 180.0 - servo_angles_deg[7]
    # Ankle joint: servo = 60 + action_deg => action_deg = servo - 60 (left leg)
    sim_angles_deg[5] = servo_angles_deg[8] - 60.0

    # l3 leg (servo 12-14 -> sim 6-8)
    # Hip joint: servo = 180 - action_deg => action_deg = 180 - servo
    sim_angles_deg[6] = 180.0 - servo_angles_deg[12]
    # Knee joint: servo = 180 - action_deg => action_deg = 180 - servo (left leg)
    sim_angles_deg[7] = 180.0 - servo_angles_deg[13]
    # Ankle joint: servo = 60 + action_deg => action_deg = servo - 60 (left leg)
    sim_angles_deg[8] = servo_angles_deg[14] - 60.0

    # r1 leg (servo 3-5 -> sim 9-11)
    # Hip:   servo = 180 - sim → sim = 180 - servo
    sim_angles_deg[9] = 180.0 - servo_angles_deg[3]
    # Knee:  servo = 180 + sim → sim = servo - 180
    sim_angles_deg[10] = servo_angles_deg[4] - 180.0
    # Ankle: servo = 60  - sim → sim = 60  - servo
    sim_angles_deg[11] = 60.0 - servo_angles_deg[5]

    # r2 leg (servo 9-11 -> sim 12-14)
    sim_angles_deg[12] = 180.0 - servo_angles_deg[9]
    sim_angles_deg[13] = servo_angles_deg[10] - 180.0
    sim_angles_deg[14] = 60.0 - servo_angles_deg[11]

    # r3 leg (servo 15-17 -> sim 15-17)
    sim_angles_deg[15] = 180.0 - servo_angles_deg[15]
    sim_angles_deg[16] = servo_angles_deg[16] - 180.0
    sim_angles_deg[17] = 60.0 - servo_angles_deg[17]

    # Convert to radians
    sim_angles_rad = np.radians(sim_angles_deg)

    return sim_angles_rad


def sim_angles_rad_to_servo_angles_deg(sim_angles_rad):
    """
    Convert simulation joint angles (radians, sim order) to servo angles (degrees, servo order).
    Inverse of servo_angles_to_sim_angles. Used when replaying sim dof_pos.

    Args:
        sim_angles_rad: 18-dim array in sim order [l1_bc,l1_cf,l1_ft, l2,l2,l2, l3,l3,l3, r1,r1,r1, r2,r2,r2, r3,r3,r3] (radians)

    Returns:
        servo_angles_deg: 18-dim array in servo order [l1,l1,l1, r1,r1,r1, l2,l2,l2, r2,r2,r2, l3,l3,l3, r3,r3,r3] (degrees)
    """
    sim_deg = np.degrees(sim_angles_rad)
    servo = np.zeros(18)
    # l1 (sim 0-2 -> servo 0-2)
    servo[0] = 180.0 - sim_deg[0]
    servo[1] = 180.0 - sim_deg[1]
    servo[2] = sim_deg[2] + 60.0
    # l2 (sim 3-5 -> servo 6-8)
    servo[6] = 180.0 - sim_deg[3]
    servo[7] = 180.0 - sim_deg[4]
    servo[8] = sim_deg[5] + 60.0
    # l3 (sim 6-8 -> servo 12-14)
    servo[12] = 180.0 - sim_deg[6]
    servo[13] = 180.0 - sim_deg[7]
    servo[14] = sim_deg[8] + 60.0
    # r1 (sim 9-11 -> servo 3-5)
    servo[3] = 180.0 - sim_deg[9]
    servo[4] = 180.0 + sim_deg[10]   # right knee: servo = 180 + sim
    servo[5] = 60.0 - sim_deg[11]
    # r2 (sim 12-14 -> servo 9-11)
    servo[9] = 180.0 - sim_deg[12]
    servo[10] = 180.0 + sim_deg[13]  # right knee: servo = 180 + sim
    servo[11] = 60.0 - sim_deg[14]
    # r3 (sim 15-17 -> servo 15-17)
    servo[15] = 180.0 - sim_deg[15]
    servo[16] = 180.0 + sim_deg[16]  # right knee: servo = 180 + sim
    servo[17] = 60.0 - sim_deg[17]
    return servo


def load_sim_dof_pos(filepath):
    """
    Load dof_pos (18-dim, sim order, radians) from each line of wm_obs_prop-style file.
    Each line is one observation block [[ ... ]]; layout: [0:3] base_ang_vel, [3:6] projected_gravity,
    [6:9] commands, [9:27] dof_pos (18 values).

    Returns:
        list of np.ndarray of shape (18,) in radians, sim order.
    """
    import re
    out = []
    with open(filepath, 'r') as f:
        content = f.read()
    # Split by "]]" to get each observation block (then take content before last "]" of block)
    blocks = re.split(r'\]\s*\]', content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Remove leading "[["
        block = re.sub(r'^\s*\[\s*\[\s*', '', block)
        if not block:
            continue
        # Extract all floats
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', block)
        nums = [float(x) for x in nums]
        if len(nums) < 27:
            continue
        dof_pos = np.array(nums[9:27], dtype=np.float64)
        out.append(dof_pos)
    return out


def get_action_limits():
    """
    Calculate the safe action limits in radians to prevent servo angle clipping.
    Based on the action_to_servo_angles mapping and servo angle limits.

    Returns:
        action_limits: dict with 'min' and 'max' arrays for each action dimension (in radians)
    """
    # Get servo angle limits from config (degrees)
    servo_min = ROBOT_CONFIG['angle_limits']['min']
    servo_max = ROBOT_CONFIG['angle_limits']['max']

    # Initialize action limits arrays (in degrees first, then convert to radians)
    action_min_deg = np.zeros(18)
    action_max_deg = np.zeros(18)

    # Based on action_to_servo_angles mapping:
    # Hip joints: servo = 180 - action_deg => action_deg = 180 - servo
    # Left knee: servo = 180 - action_deg => action_deg = 180 - servo
    # Right knee: servo = 180 + action_deg => action_deg = servo - 180
    # Left ankle: servo = 60 + action_deg => action_deg = servo - 60
    # Right ankle: servo = 60 - action_deg => action_deg = 60 - servo

    # Action indices: 0-2:l1, 3-5:l2, 6-8:l3, 9-11:r1, 12-14:r2, 15-17:r3
    # Servo indices:  0-2:l1, 3-5:r1, 6-8:l2, 9-11:r2, 12-14:l3, 15-17:r3

    # Initialize with raw limits (will be adjusted with safety margins below)
    # l1 leg actions (0-2) -> servo indices (0-2)
    action_min_deg[0] = 180.0 - servo_max[0]  # Hip
    action_max_deg[0] = 180.0 - servo_min[0]
    action_min_deg[1] = 180.0 - servo_max[1]  # Left knee
    action_max_deg[1] = 180.0 - servo_min[1]
    action_min_deg[2] = servo_min[2] - 60.0   # Left ankle
    action_max_deg[2] = servo_max[2] - 60.0

    # l2 leg actions (3-5) -> servo indices (6-8)
    action_min_deg[3] = 180.0 - servo_max[6]  # Hip
    action_max_deg[3] = 180.0 - servo_min[6]
    action_min_deg[4] = 180.0 - servo_max[7]  # Left knee
    action_max_deg[4] = 180.0 - servo_min[7]
    action_min_deg[5] = servo_min[8] - 60.0   # Left ankle
    action_max_deg[5] = servo_max[8] - 60.0

    # l3 leg actions (6-8) -> servo indices (12-14)
    action_min_deg[6] = 180.0 - servo_max[12]  # Hip
    action_max_deg[6] = 180.0 - servo_min[12]
    action_min_deg[7] = 180.0 - servo_max[13]  # Left knee
    action_max_deg[7] = 180.0 - servo_min[13]
    action_min_deg[8] = servo_min[14] - 60.0   # Left ankle
    action_max_deg[8] = servo_max[14] - 60.0

    # r1 leg actions (9-11) -> servo indices (3-5)
    action_min_deg[9] = 180.0 - servo_max[3]   # Hip: servo=180-sim → sim=180-servo
    action_max_deg[9] = 180.0 - servo_min[3]
    action_min_deg[10] = servo_min[4] - 180.0   # Right knee: servo=180+sim → sim=servo-180
    action_max_deg[10] = servo_max[4] - 180.0
    action_min_deg[11] = 60.0 - servo_max[5]   # Right ankle
    action_max_deg[11] = 60.0 - servo_min[5]

    # r2 leg actions (12-14) -> servo indices (9-11)
    action_min_deg[12] = 180.0 - servo_max[9]  # Hip
    action_max_deg[12] = 180.0 - servo_min[9]
    action_min_deg[13] = servo_min[10] - 180.0  # Right knee: servo=180+sim → sim=servo-180
    action_max_deg[13] = servo_max[10] - 180.0
    action_min_deg[14] = 60.0 - servo_max[11]  # Right ankle
    action_max_deg[14] = 60.0 - servo_min[11]

    # r3 leg actions (15-17) -> servo indices (15-17)
    action_min_deg[15] = 180.0 - servo_max[15] # Hip
    action_max_deg[15] = 180.0 - servo_min[15]
    action_min_deg[16] = servo_min[16] - 180.0  # Right knee: servo=180+sim → sim=servo-180
    action_max_deg[16] = servo_max[16] - 180.0
    action_min_deg[17] = 60.0 - servo_max[17]  # Right ankle
    action_max_deg[17] = 60.0 - servo_min[17]

    # Convert to radians
    action_min_rad = np.radians(action_min_deg)
    action_max_rad = np.radians(action_max_deg)

    # Add safety margin to ensure servo angles stay strictly within limits
    safety_margin_deg = 1.0  # 2.0 degree safety margin for reliability
    safety_margin_rad = np.radians(safety_margin_deg)

    # Apply safety margins to all action limits
    # For hip joints (servo = 180 - action): smaller action -> larger servo
    # Map action indices to servo indices for hip joints
    hip_action_to_servo = {0: 0, 3: 6, 6: 12, 9: 3, 12: 9, 15: 15}
    for action_idx, servo_idx in hip_action_to_servo.items():
        # To ensure servo <= servo_max - margin, action >= 180 - (servo_max - margin)
        action_min_deg[action_idx] = 180.0 - (servo_max[servo_idx] - safety_margin_deg)
        # To ensure servo >= servo_min + margin, action <= 180 - (servo_min + margin)
        action_max_deg[action_idx] = 180.0 - (servo_min[servo_idx] + safety_margin_deg)

    # For left knee joints (same as hip: servo = 180 - action)
    # Map action indices to servo indices for left knee joints
    left_knee_action_to_servo = {1: 1, 4: 7, 7: 13}  # l1, l2, l3 knees
    for action_idx, servo_idx in left_knee_action_to_servo.items():
        action_min_deg[action_idx] = 180.0 - (servo_max[servo_idx] - safety_margin_deg)
        action_max_deg[action_idx] = 180.0 - (servo_min[servo_idx] + safety_margin_deg)

    # For right knee joints (servo = 180 + action): larger action -> larger servo
    right_knee_action_to_servo = {10: 4, 13: 10, 16: 16}  # r1, r2, r3 knees
    for action_idx, servo_idx in right_knee_action_to_servo.items():
        # To ensure servo >= servo_min + margin, action >= (servo_min + margin) - 180
        action_min_deg[action_idx] = (servo_min[servo_idx] + safety_margin_deg) - 180.0
        # To ensure servo <= servo_max - margin, action <= (servo_max - margin) - 180
        action_max_deg[action_idx] = (servo_max[servo_idx] - safety_margin_deg) - 180.0

    # For left ankle joints (servo = 60 + action): larger action -> larger servo
    # Map action indices to servo indices for left ankle joints
    left_ankle_action_to_servo = {2: 2, 5: 8, 8: 14}  # l1, l2, l3 ankles
    for action_idx, servo_idx in left_ankle_action_to_servo.items():
        # To ensure servo >= servo_min + margin, action >= (servo_min + margin) - 60
        action_min_deg[action_idx] = (servo_min[servo_idx] + safety_margin_deg) - 60.0
        # To ensure servo <= servo_max - margin, action <= (servo_max - margin) - 60
        action_max_deg[action_idx] = (servo_max[servo_idx] - safety_margin_deg) - 60.0

    # For right ankle joints (servo = 60 - action): smaller action -> larger servo
    # Map action indices to servo indices for right ankle joints
    right_ankle_action_to_servo = {11: 5, 14: 11, 17: 17}  # r1, r2, r3 ankles
    for action_idx, servo_idx in right_ankle_action_to_servo.items():
        # To ensure servo <= servo_max - margin, action >= 60 - (servo_max - margin)
        action_min_deg[action_idx] = 60.0 - (servo_max[servo_idx] - safety_margin_deg)
        # To ensure servo >= servo_min + margin, action <= 60 - (servo_min + margin)
        action_max_deg[action_idx] = 60.0 - (servo_min[servo_idx] + safety_margin_deg)

    # Convert to radians
    action_min_rad = np.radians(action_min_deg)
    action_max_rad = np.radians(action_max_deg)

    return {
        'min': action_min_rad,
        'max': action_max_rad,
        'min_deg': action_min_deg,
        'max_deg': action_max_deg
    }

def action_to_servo_angles(actions_radians):
    """
    Convert policy actions (radians) to servo angles (degrees) with proper mapping.

    Args:
        actions_radians: 18-dim array of actions in radians
                        Order: [l1_hip, l1_knee, l1_ankle, l2_hip, l2_knee, l2_ankle,
                               l3_hip, l3_knee, l3_ankle, r1_hip, r1_knee, r1_ankle,
                               r2_hip, r2_knee, r2_ankle, r3_hip, r3_knee, r3_ankle]

    Returns:
        servo_angles: 18-dim array of servo angles in degrees
                      Order: [l1_hip, l1_knee, l1_ankle, r1_hip, r1_knee, r1_ankle,
                             l2_hip, l2_knee, l2_ankle, r2_hip, r2_knee, r2_ankle,
                             l3_hip, l3_knee, l3_ankle, r3_hip, r3_knee, r3_ankle]
    """
    # Convert actions to degrees
    actions_deg = radians_to_degrees(actions_radians)

    # Initialize servo angles array
    servo_angles = np.zeros(18)

    # Reorder from action order to servo order and apply joint-specific mappings
    # Action indices: 0-2:l1, 3-5:l2, 6-8:l3, 9-11:r1, 12-14:r2, 15-17:r3
    # Servo indices:  0-2:l1, 3-5:r1, 6-8:l2, 9-11:r2, 12-14:l3, 15-17:r3

    # l1 leg (action 0-2 -> servo 0-2)
    # Hip joint (idx 0): action=0 -> 180°, positive action -> smaller angle
    servo_angles[0] = 180.0 - actions_deg[0]
    # Knee joint (idx 1): left leg, action=-90° -> 270°, action increase -> angle decrease
    servo_angles[1] = 270.0 - (actions_deg[1] + 90.0)
    # Ankle joint (idx 2): left leg, action=90° -> 150°, action increase -> angle increase
    servo_angles[2] = 150.0 + (actions_deg[2] - 90.0)

    # r1 leg (action 9-11 -> servo 3-5)
    servo_angles[3] = 180.0 - actions_deg[9]
    servo_angles[4] = 180.0 + actions_deg[10]   # right knee: servo = 180 + sim
    servo_angles[5] = 150.0 - (actions_deg[11] + 90.0)

    # l2 leg (action 3-5 -> servo 6-8)
    # Hip joint (idx 6): action=0 -> 180°, positive action -> smaller angle
    servo_angles[6] = 180.0 - actions_deg[3]
    # Knee joint (idx 7): left leg, action=-90° -> 270°, action increase -> angle decrease
    servo_angles[7] = 270.0 - (actions_deg[4] + 90.0)
    # Ankle joint (idx 8): left leg, action=90° -> 150°, action increase -> angle increase
    servo_angles[8] = 150.0 + (actions_deg[5] - 90.0)

    # r2 leg (action 12-14 -> servo 9-11)
    servo_angles[9] = 180.0 - actions_deg[12]
    servo_angles[10] = 180.0 + actions_deg[13]  # right knee: servo = 180 + sim
    servo_angles[11] = 150.0 - (actions_deg[14] + 90.0)

    # l3 leg (action 6-8 -> servo 12-14)
    # Hip joint (idx 12): action=0 -> 180°, positive action -> smaller angle
    servo_angles[12] = 180.0 - actions_deg[6]
    # Knee joint (idx 13): left leg, action=-90° -> 270°, action increase -> angle decrease
    servo_angles[13] = 270.0 - (actions_deg[7] + 90.0)
    # Ankle joint (idx 14): left leg, action=90° -> 150°, action increase -> angle increase
    servo_angles[14] = 150.0 + (actions_deg[8] - 90.0)

    # r3 leg (action 15-17 -> servo 15-17)
    servo_angles[15] = 180.0 - actions_deg[15]
    servo_angles[16] = 180.0 + actions_deg[16]  # right knee: servo = 180 + sim
    servo_angles[17] = 150.0 - (actions_deg[17] + 90.0)

    return servo_angles


class AdmittanceFilter:
    """Admittance controller to smooth desired joint actions."""
    def __init__(self, m, d, k, dt, num_joints):
        self.m = m
        self.d = d
        self.k = k
        self.dt = dt
        self.num_joints = num_joints
        self.q = np.zeros(num_joints, dtype=np.float32)
        self.qd = np.zeros(num_joints, dtype=np.float32)
        self.initialized = False

    def reset(self, q_desired):
        self.q = q_desired.astype(np.float32).copy()
        self.qd = np.zeros(self.num_joints, dtype=np.float32)
        self.initialized = True

    def update(self, q_desired):
        q_desired = q_desired.astype(np.float32)
        if not self.initialized:
            self.reset(q_desired)
            return self.q.copy()

        # m*qdd + d*qd + k*(q - q_des) = 0
        qdd = (self.k * (q_desired - self.q) - self.d * self.qd) / self.m
        self.qd = self.qd + qdd * self.dt
        self.q = self.q + self.qd * self.dt
        return self.q.copy()


class _LegacyImaginationStabilityFilter:
    """Sampling-based safety filter using Dreamer RSSM imagination."""

    def __init__(
        self,
        world_model,
        action_dim=18,
        horizon=5,
        num_samples=8,
        noise_scale=0.05,
        device="cuda",
    ):
        """
        world_model: contains encoder and dynamics.obs_step.
        """
        self.world_model = world_model
        self.action_dim = action_dim
        self.horizon = min(int(horizon), 5)
        self.num_samples = min(int(num_samples), 8)
        self.noise_scale = float(noise_scale)
        self.device = torch.device(device)
        self.gravity_target = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)

        # Stability weights: orientation is most important, then angular velocity.
        self.w_gravity = 1.0
        self.w_ang_vel = 0.35
        self.w_smooth = 0.08
        self.min_improvement = 0.02
        self.max_perturb = 0.20
        self.last_debug = {}

    def _ensure_2d(self, x):
        if x is None:
            return None
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    def _repeat_latent(self, latent, n):
        if isinstance(latent, dict):
            out = {}
            for k, v in latent.items():
                if torch.is_tensor(v):
                    out[k] = v.repeat_interleave(n, dim=0)
                else:
                    out[k] = v
            return out
        if torch.is_tensor(latent):
            return latent.repeat_interleave(n, dim=0)
        raise TypeError(f"Unsupported latent type: {type(latent)}")

    def compute_stability(self, obs_prop, prev_action=None):
        """
        obs_prop: tensor containing projected_gravity and base_ang_vel.
        Return scalar instability (higher = worse).
        """
        obs_prop = self._ensure_2d(obs_prop).to(self.device, dtype=torch.float32)
        base_ang_vel = obs_prop[:, 0:3]
        projected_gravity = obs_prop[:, 3:6]

        gravity_dev = torch.norm(projected_gravity - self.gravity_target.unsqueeze(0), dim=-1)
        ang_vel_mag = torch.norm(base_ang_vel, dim=-1)
        instability = self.w_gravity * gravity_dev + self.w_ang_vel * ang_vel_mag

        if prev_action is not None:
            prev_action = self._ensure_2d(prev_action).to(self.device, dtype=torch.float32)
            cur_action = obs_prop[:, -self.action_dim:]
            smooth_penalty = torch.norm(cur_action - prev_action, dim=-1)
            instability = instability + self.w_smooth * smooth_penalty

        return instability

    def rollout_imagination(self, prev_latent, action_candidates, is_first):
        """
        Perform parallel rollout for all action candidates.
        """
        dynamics = self.world_model.dynamics
        n = action_candidates.shape[0]
        latent = self._repeat_latent(prev_latent, n)
        is_first = self._ensure_2d(is_first).to(self.device, dtype=torch.float32)
        is_first = is_first.repeat_interleave(n, dim=0)
        action_candidates = action_candidates.to(self.device, dtype=torch.float32)

        latent_seq = []
        for _ in range(self.horizon):
            if hasattr(dynamics, "imagine_with_action"):
                latent = dynamics.imagine_with_action(latent, action_candidates)
            elif hasattr(dynamics, "img_step"):
                latent = dynamics.img_step(latent, action_candidates, sample=True)
            else:
                # Fallback: obs_step with zero embed (no new real observation in imagination).
                if isinstance(latent, dict):
                    any_latent = next(v for v in latent.values() if torch.is_tensor(v))
                    batch = any_latent.shape[0]
                else:
                    batch = latent.shape[0]
                embed_dim = getattr(dynamics, "_embed", 1024)
                zero_embed = torch.zeros(batch, embed_dim, device=self.device, dtype=torch.float32)
                latent, _ = dynamics.obs_step(latent, action_candidates, zero_embed, is_first, True)
            latent_seq.append(latent)
            is_first = torch.zeros_like(is_first)
        return latent_seq

    def evaluate_actions(self, prev_latent, action_candidates, is_first, obs_decoder=None):
        """
        For each candidate:
        - rollout imagination
        - compute instability over horizon
        - use MAX instability (not sum)
        """
        with torch.no_grad():
            latent_seq = self.rollout_imagination(prev_latent, action_candidates, is_first)
            scores = []
            for latent in latent_seq:
                if obs_decoder is not None:
                    pred = obs_decoder(latent)
                    if isinstance(pred, dict):
                        pred_prop = pred.get("prop", None)
                        if pred_prop is None:
                            raise KeyError("obs_decoder output has no 'prop' key.")
                    else:
                        pred_prop = pred
                else:
                    feat = self.world_model.dynamics.get_deter_feat(latent)
                    # If no decoder is given, approximate using feature prefix.
                    pred_prop = feat[:, :6]
                scores.append(self.compute_stability(pred_prop))
            stacked = torch.stack(scores, dim=0)  # [H, N]
            return torch.max(stacked, dim=0).values

    def select_action(self, obs_prop, prev_latent, action_nominal, is_first, prev_action=None):
        """
        Perturb nominal action, evaluate imagined instability, select safest.
        """
        with torch.no_grad():
            action_nominal = self._ensure_2d(action_nominal).to(self.device, dtype=torch.float32)
            obs_prop = self._ensure_2d(obs_prop).to(self.device, dtype=torch.float32)
            n = self.num_samples

            noise = torch.randn(n, self.action_dim, device=self.device) * self.noise_scale
            noise = torch.clamp(noise, -self.max_perturb, self.max_perturb)
            candidates = action_nominal.repeat(n, 1) + noise
            candidates = torch.clamp(candidates, -1.0, 1.0)

            base_score = self.compute_stability(obs_prop).mean()
            cand_scores = self.evaluate_actions(
                prev_latent=prev_latent,
                action_candidates=candidates,
                is_first=is_first,
                obs_decoder=getattr(self.world_model, "decoder", None),
            )
            best_idx = torch.argmin(cand_scores)
            best_score = cand_scores[best_idx]
            improved = (base_score - best_score) > self.min_improvement

            if not bool(improved):
                return torch.clamp(action_nominal, -1.0, 1.0)
            return candidates[best_idx:best_idx + 1]


class ImaginationStabilityFilter:
    """Sampling-based stability filter using Dreamer/RSSM imagination."""

    def __init__(
        self,
        world_model,
        action_dim=18,
        horizon=5,
        num_samples=8,
        noise_scale=0.05,
        device="cuda",
    ):
        """
        world_model: contains encoder and dynamics.obs_step.
        """
        self.world_model = world_model
        self.action_dim = int(action_dim)
        self.horizon = min(int(horizon), 5)
        self.num_samples = min(int(num_samples), 8)
        self.noise_scale = float(noise_scale)
        self.device = torch.device(device)
        self.gravity_target = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)

        dynamics = getattr(world_model, "dynamics", None)
        self.rssm_action_dim = int(getattr(dynamics, "_num_actions", self.action_dim))

        # Orientation dominates; angular velocity and smoothness are secondary.
        self.w_gravity = 1.0
        self.w_ang_vel = 0.35
        self.w_smooth = 0.08
        self.min_improvement = 0.02
        self.max_perturb = 0.20

    def _ensure_2d(self, x):
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    def _repeat_latent(self, latent, n):
        if isinstance(latent, dict):
            return {
                k: (v.repeat_interleave(n, dim=0) if torch.is_tensor(v) else v)
                for k, v in latent.items()
            }
        if torch.is_tensor(latent):
            return latent.repeat_interleave(n, dim=0)
        raise TypeError(f"Unsupported latent type: {type(latent)}")

    def _expand_action_for_rssm(self, action):
        """Map 18-dim joint targets to the RSSM action input used by this checkpoint."""
        action = self._ensure_2d(action).to(self.device, dtype=torch.float32)
        if action.shape[-1] == self.rssm_action_dim:
            return action
        if self.rssm_action_dim == self.action_dim:
            return action
        if self.rssm_action_dim % self.action_dim == 0:
            # Some checkpoints train RSSM on a short action-history vector.
            return action.repeat(1, self.rssm_action_dim // self.action_dim)
        raise ValueError(f"Cannot map action_dim={action.shape[-1]} to rssm_action_dim={self.rssm_action_dim}")

    def _decoder_from_world_model(self):
        if hasattr(self.world_model, "decoder"):
            return self.world_model.decoder
        heads = getattr(self.world_model, "heads", None)
        if heads is not None and "decoder" in heads:
            return heads["decoder"]
        return None

    def _dist_to_tensor(self, value):
        if torch.is_tensor(value):
            return value
        mean = getattr(value, "mean", None)
        if callable(mean):
            return mean()
        if torch.is_tensor(mean):
            return mean
        mode = getattr(value, "mode", None)
        if callable(mode):
            return mode()
        raise TypeError(f"Cannot convert decoder output of type {type(value)} to tensor")

    def _decode_prop(self, latent, obs_decoder=None):
        """Decode imagined RSSM state to proprioceptive observation when a decoder is available."""
        dynamics = self.world_model.dynamics
        decoder = obs_decoder or self._decoder_from_world_model()
        if decoder is None:
            # Conservative fallback if no decoder is exposed.
            feat = dynamics.get_deter_feat(latent)
            return feat[:, :6]

        feat = dynamics.get_feat(latent) if hasattr(dynamics, "get_feat") else dynamics.get_deter_feat(latent)
        pred = decoder(feat)
        if isinstance(pred, dict):
            if "prop" not in pred:
                raise KeyError("World-model decoder output has no 'prop' key.")
            pred = pred["prop"]
        return self._dist_to_tensor(pred)

    def _infer_embed_dim(self):
        dynamics = self.world_model.dynamics
        deter_dim = int(getattr(dynamics, "_deter", 0))
        obs_layers = getattr(dynamics, "_obs_out_layers", None)
        if obs_layers is not None and len(obs_layers) > 0 and hasattr(obs_layers[0], "in_features"):
            return int(obs_layers[0].in_features) - deter_dim
        return 1024

    def compute_stability(self, obs_prop, prev_action=None):
        """
        obs_prop: tensor containing base_ang_vel at [0:3] and projected_gravity at [3:6].

        Instability score:
        - projected_gravity deviation from [0, 0, -1]
        - angular velocity magnitude
        - optional action smoothness if obs_prop carries an action tail

        Return: (B,) instability, higher is worse.
        """
        obs_prop = self._ensure_2d(obs_prop).to(self.device, dtype=torch.float32)
        base_ang_vel = obs_prop[:, 0:3]
        projected_gravity = obs_prop[:, 3:6]

        gravity_dev = torch.norm(projected_gravity - self.gravity_target.unsqueeze(0), dim=-1)
        ang_vel_mag = torch.norm(base_ang_vel, dim=-1)
        instability = self.w_gravity * gravity_dev + self.w_ang_vel * ang_vel_mag

        if prev_action is not None and obs_prop.shape[-1] >= self.action_dim:
            prev_action = self._ensure_2d(prev_action).to(self.device, dtype=torch.float32)
            cur_action = obs_prop[:, -self.action_dim:]
            if prev_action.shape[0] == 1 and cur_action.shape[0] > 1:
                prev_action = prev_action.repeat(cur_action.shape[0], 1)
            instability = instability + self.w_smooth * torch.norm(cur_action - prev_action, dim=-1)

        return instability

    def rollout_imagination(self, prev_latent, action_candidates, is_first):
        """
        Perform parallel rollout for all action candidates.

        No encoder is called here; this is pure Dreamer imagination.
        """
        dynamics = self.world_model.dynamics
        n = action_candidates.shape[0]
        latent = self._repeat_latent(prev_latent, n)
        is_first = is_first.reshape(-1).to(self.device, dtype=torch.float32).repeat_interleave(n)
        rssm_action = self._expand_action_for_rssm(action_candidates)

        # Prefer the RSSM's vectorized imagination API if exposed by the world model.
        if hasattr(dynamics, "imagine_with_action"):
            try:
                action_seq = rssm_action.unsqueeze(1).repeat(1, self.horizon, 1)
                imagined = dynamics.imagine_with_action(action_seq, latent)
                return [{k: v[:, t] for k, v in imagined.items()} for t in range(self.horizon)]
            except Exception:
                # Fall through to explicit img_step for slight interface/checkpoint differences.
                pass

        latent_seq = []
        for _ in range(self.horizon):
            if hasattr(dynamics, "img_step"):
                latent = dynamics.img_step(latent, rssm_action, sample=True)
            else:
                # Fallback: obs_step with zero embed (no new real observation in imagination).
                if isinstance(latent, dict):
                    any_latent = next(v for v in latent.values() if torch.is_tensor(v))
                    batch = any_latent.shape[0]
                else:
                    batch = latent.shape[0]
                zero_embed = torch.zeros(batch, self._infer_embed_dim(), device=self.device, dtype=torch.float32)
                latent, _ = dynamics.obs_step(latent, rssm_action, zero_embed, is_first, True)
            latent_seq.append(latent)
            is_first = torch.zeros_like(is_first)
        return latent_seq

    def evaluate_actions(self, prev_latent, action_candidates, is_first, obs_decoder=None):
        """
        For each candidate:
        - rollout imagination
        - compute instability over horizon
        - use MAX instability (not sum)

        Return:
        - (N,) instability scores
        """
        with torch.no_grad():
            latent_seq = self.rollout_imagination(prev_latent, action_candidates, is_first)
            scores = []
            for latent in latent_seq:
                pred_prop = self._decode_prop(latent, obs_decoder=obs_decoder)
                scores.append(self.compute_stability(pred_prop))
            return torch.max(torch.stack(scores, dim=0), dim=0).values

    def select_action(self, obs_prop, prev_latent, action_nominal, is_first, prev_action=None):
        """
        Perturb nominal action, evaluate imagined instability, select safest.
        """
        with torch.no_grad():
            action_nominal = self._ensure_2d(action_nominal).to(self.device, dtype=torch.float32)
            self._ensure_2d(obs_prop).to(self.device, dtype=torch.float32)
            prev_action_t = None if prev_action is None else self._ensure_2d(prev_action).to(self.device, dtype=torch.float32)
            self.last_debug = {"used": False, "error": ""}

            # Include nominal action as candidate 0, then add bounded perturbations.
            n = self.num_samples
            noise = torch.randn(max(0, n - 1), self.action_dim, device=self.device) * self.noise_scale
            noise = torch.clamp(noise, -self.max_perturb, self.max_perturb)
            candidates = torch.cat([action_nominal, action_nominal.repeat(max(0, n - 1), 1) + noise], dim=0)
            candidates = torch.clamp(candidates, -1.0, 1.0)

            cand_scores = self.evaluate_actions(
                prev_latent=prev_latent,
                action_candidates=candidates,
                is_first=is_first,
                obs_decoder=self._decoder_from_world_model(),
            )
            if prev_action_t is not None:
                cand_scores = cand_scores + self.w_smooth * torch.norm(
                    candidates - prev_action_t.repeat(candidates.shape[0], 1), dim=-1
                )

            best_idx = torch.argmin(cand_scores)
            best_score = cand_scores[best_idx]
            nominal_score = cand_scores[0]
            selected = candidates[best_idx:best_idx + 1]
            delta = torch.norm(selected - action_nominal, dim=-1)[0]
            improvement = nominal_score - best_score
            used = bool(improvement > self.min_improvement)
            self.last_debug = {
                "used": used,
                "best_idx": int(best_idx.detach().cpu().item()),
                "nominal_score": float(nominal_score.detach().cpu().item()),
                "best_score": float(best_score.detach().cpu().item()),
                "improvement": float(improvement.detach().cpu().item()),
                "delta": float(delta.detach().cpu().item()),
                "num_samples": int(candidates.shape[0]),
                "horizon": int(self.horizon),
                "noise_scale": float(self.noise_scale),
                "error": "",
            }
            if not used:
                return torch.clamp(action_nominal, -1.0, 1.0)
            return selected


class ContactAnomalyDetector:
    """World-model PRIOR based contact anomaly detector for real-time control."""

    def __init__(
        self,
        world_model,
        threshold=0.15,
        ema_alpha=0.9,
        trigger_count=3,
        action_dim=18,
        device="cuda",
    ):
        self.world_model = world_model
        self.threshold = float(threshold)
        self.ema_alpha = float(ema_alpha)
        self.trigger_count = int(trigger_count)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)

        self.ema_error = None
        self.anomaly_counter = 0
        self._rssm_action_dim = self._infer_rssm_action_dim()
        self.last_obs_real = None
        self.last_obs_pred = None

    def _infer_rssm_action_dim(self):
        dynamics = getattr(self.world_model, "dynamics", None)
        return int(getattr(dynamics, "_num_actions", self.action_dim))

    def _ensure_2d(self, x):
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    def _expand_action_for_rssm(self, action):
        action = self._ensure_2d(action).to(self.device, dtype=torch.float32)
        if action.shape[-1] == self._rssm_action_dim:
            return action
        if self._rssm_action_dim == self.action_dim:
            return action
        if self._rssm_action_dim % self.action_dim == 0:
            repeat = self._rssm_action_dim // self.action_dim
            return action.repeat(1, repeat)
        raise ValueError(
            f"Cannot map action_dim={action.shape[-1]} to rssm_action_dim={self._rssm_action_dim}"
        )

    def _dist_to_tensor(self, value):
        if torch.is_tensor(value):
            return value
        mean = getattr(value, "mean", None)
        if callable(mean):
            return mean()
        if torch.is_tensor(mean):
            return mean
        mode = getattr(value, "mode", None)
        if callable(mode):
            return mode()
        raise TypeError(f"Cannot convert decoder output of type {type(value)} to tensor")

    @torch.no_grad()
    def predict_next_obs(self, prev_latent, action, obs_dict):
        """
        Use PRIOR from obs_step to predict next observation:
        1) encoder(obs_dict) -> embed
        2) obs_step(prev_latent, action, embed) -> (posterior, prior)
        3) decode prior branch (not posterior) for anomaly prediction

        PRIOR is used because it is the model's prediction before conditioning on
        the current real observation. A large prior-vs-real mismatch indicates
        unexpected contact/slip/disturbance.
        """
        prop = obs_dict["prop"].to(self.device, dtype=torch.float32)
        is_first = obs_dict["is_first"].to(self.device, dtype=torch.float32)
        obs_for_wm = {"prop": prop, "is_first": is_first}

        embed = self.world_model.encoder(obs_for_wm)
        rssm_action = self._expand_action_for_rssm(action)
        latent_post, latent_prior = self.world_model.dynamics.obs_step(
            prev_latent,
            rssm_action,
            embed,
            is_first,
            sample=True,
        )

        feat = self.world_model.dynamics.get_feat(latent_prior)
        pred = self.world_model.heads["decoder"](feat)
        if isinstance(pred, dict):
            if "prop" not in pred:
                raise KeyError("Decoder prediction has no 'prop' key.")
            obs_pred = pred["prop"]
        else:
            obs_pred = pred
        obs_pred = self._dist_to_tensor(obs_pred)
        return obs_pred, latent_post

    def compute_error(self, obs_real, obs_pred):
        """
        Compute weighted L2 error on partial observation only:
        - base_ang_vel [0:3]
        - projected_gravity [3:6]

        Commands and joint states are intentionally ignored. Contact anomalies
        should show up first as unexpected body angular velocity/orientation
        change, while commanded motion and leg phase can vary normally.
        """
        obs_real = self._ensure_2d(obs_real).to(self.device, dtype=torch.float32)
        obs_pred = self._ensure_2d(obs_pred).to(self.device, dtype=torch.float32)

        ang_real = obs_real[..., 0:3]
        ang_pred = obs_pred[..., 0:3]
        grav_real = obs_real[..., 3:6]
        grav_pred = obs_pred[..., 3:6]

        ang_err = torch.norm(ang_real - ang_pred, dim=-1)
        grav_err = torch.norm(grav_real - grav_pred, dim=-1)
        return grav_err + 0.3 * ang_err

    @torch.no_grad()
    def detect(self, prev_latent, action, obs_dict):
        """
        Return:
        - is_anomaly (bool)
        - ema_error (tensor)
        - latent_post (for next-step state update)
        - anomaly_counter (consecutive trigger count)

        EMA suppresses one-frame sensor noise. The consecutive trigger prevents
        single bad packets or transient impacts from immediately changing gait.
        """
        obs_real = obs_dict["prop"].to(self.device, dtype=torch.float32)
        obs_pred, latent_post = self.predict_next_obs(prev_latent, action, obs_dict)
        # Cache current real/predicted observations so runtime correction can
        # localize which leg contributed most to the prediction mismatch.
        self.last_obs_real = obs_real.detach()
        self.last_obs_pred = obs_pred.detach()
        error = self.compute_error(obs_real, obs_pred)

        if self.ema_error is None:
            self.ema_error = error.detach()
        else:
            self.ema_error = (
                self.ema_alpha * self.ema_error
                + (1.0 - self.ema_alpha) * error.detach()
            )

        trigger = bool(torch.any(self.ema_error > self.threshold).item())
        if trigger:
            self.anomaly_counter += 1
        else:
            self.anomaly_counter = 0
        is_anomaly = self.anomaly_counter >= self.trigger_count
        return is_anomaly, self.ema_error, latent_post, self.anomaly_counter


def compute_leg_errors(obs_real, obs_pred):
    """
    Compute per-leg joint prediction errors from observation joint slice [9:27].

    Why this works:
    - In normal contact, WM prior and real joint states stay close.
    - A stuck leg typically shows a larger mismatch in its 3 joints because
      commanded motion is blocked by unexpected contact.
    """
    joint_real = obs_real[..., 9:27]
    joint_pred = obs_pred[..., 9:27]
    leg_errors = []
    for i in range(6):
        idx = slice(i * 3, (i + 1) * 3)
        err = torch.norm(joint_real[..., idx] - joint_pred[..., idx], dim=-1)
        leg_errors.append(err)
    return torch.stack(leg_errors, dim=0)


def detect_stuck_leg(leg_errors, threshold=0.15):
    """
    Select the leg with maximum prediction mismatch if it exceeds threshold.
    """
    max_err, leg_id = torch.max(leg_errors, dim=0)
    if bool(max_err > threshold):
        return int(leg_id.item())
    return None


class LegLiftController:
    """
    Runtime one-leg lift override.

    Why short lift is enough here:
    - Obstacles are low-height and usually require only brief clearance.
    - Keeping lift short (3~5 control steps) avoids oscillation and quickly
      hands control back to the policy.
    """

    def __init__(self, lift_steps=4):
        self.active = False
        self.leg_id = None
        self.counter = 0
        self.lift_steps = int(lift_steps)

    def trigger(self, leg_id):
        self.active = True
        self.leg_id = int(leg_id)
        self.counter = 0

    def apply(self, action):
        if not self.active:
            return action

        corrected = action.clone()
        base = self.leg_id * 3
        knee = base + 1
        ankle = base + 2
        ankle_dir = torch.tensor([1, 1, 1, -1, -1, -1], device=corrected.device, dtype=corrected.dtype)
        knee_dir = torch.tensor([-1, -1, -1, 1, 1, 1], device=corrected.device, dtype=corrected.dtype)
        corrected[..., knee] = corrected[..., knee] + 0.25 * knee_dir[self.leg_id]
        corrected[..., ankle] = corrected[..., ankle] + 0.25 * ankle_dir[self.leg_id]

        self.counter += 1
        if self.counter >= self.lift_steps:
            self.active = False
            self.leg_id = None
            self.counter = 0
        return corrected


def adjust_action(
    action,
    is_anomaly,
    anomaly_steps,
    base_scale=0.7,
    lift_gain=0.2,
    max_lift_gain=0.6,
):
    """
    Joint-space recovery strategy.

    Strategy:
    1. Stabilize by scaling down the policy action.
    2. Lift legs to escape unexpected ground contact / obstacle contact.
    3. Increase lift progressively if the anomaly persists.

    Coordinate convention in this codebase:
    - ankle lift direction: left ankles [2,5,8] are +, right ankles [11,14,17] are -
    - knee assist direction: left knees [1,4,7] bend/lift with -, right knees [10,13,16] with +

    Ankles are the primary lifting joints; knees are added with a smaller gain
    to help leg clearance without over-folding the leg.
    """
    if not is_anomaly:
        return action

    recovered = action * float(base_scale)
    gain = float(lift_gain) * (1.0 + 0.5 * float(anomaly_steps))
    gain = min(gain, float(max_lift_gain))

    ankle_idx = torch.tensor([2, 5, 8, 11, 14, 17], device=recovered.device, dtype=torch.long)
    ankle_dir = torch.tensor([1, 1, 1, -1, -1, -1], device=recovered.device, dtype=recovered.dtype)
    knee_idx = torch.tensor([1, 4, 7, 10, 13, 16], device=recovered.device, dtype=torch.long)
    knee_dir = torch.tensor([-1, -1, -1, 1, 1, 1], device=recovered.device, dtype=recovered.dtype)

    recovered[..., ankle_idx] = recovered[..., ankle_idx] + gain * ankle_dir
    recovered[..., knee_idx] = recovered[..., knee_idx] + (0.35 * gain) * knee_dir
    return torch.clamp(recovered, -1.0, 1.0)


class RealRobotRWMInference:
    """Simplified RWM inference for real robot deployment"""

    def __init__(self, model_path=None, device='cpu', remove_dof_vel=False):
        """
        RWM inference for real robot deployment with full model architecture.

        Args:
            model_path: Path to trained model
            device: Device to run inference on
        """
        self.device = device
        self.remove_dof_vel = remove_dof_vel
        self._model_path = model_path
        # WM 推理优化（torch.compile / ONNX）运行时状态
        self._wm_prop_dim = None
        self._wm_encoder_wrapper = None
        self._ort_encoder_session = None
        self._compiled_obs_step = None
        print(f"Initializing RWM inference on device: {device}")

        # Model configuration (matching the actual checkpoint parameters)
        base_obs_dim = 27 if self.remove_dof_vel else 45  # without dof_vel(18)
        self.obs_dim = base_obs_dim + (6 if cpg_reward else 0)  # proprioceptive obs dimension
        self.num_actions = 18

        # Load the trained model
        if model_path and os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device)

                # Load model state dict
                state_dict = checkpoint['model_state_dict']

                # Handle DDP model state dict (remove 'module.' prefix if present)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                    else:
                        # torch.compile may save parameters under `*_orig_mod.*` namespaces.
                        # Normalize keys so that we can load weights into the uncompiled module.
                        kk = k.replace("._orig_mod.", ".")
                        kk = kk.replace("_orig_mod.", "")
                        new_state_dict[kk] = v

                # Infer key dims from checkpoint to support different obs layouts (e.g. remove_dof_vel).
                # If the checkpoint is missing some history-encoder keys, we must fall back to the
                # expected history_dim from current runtime flags, otherwise inference will crash
                # due to input feature mismatch.
                inferred_history_dim = new_state_dict.get('history_encoder.0.weight', torch.empty(0, 0)).shape[1]
                # training-side: history_length=5, obs_without_command_dim=(42 or 60) + (6 if cpg_reward)
                expected_history_dim = ((42 if self.remove_dof_vel else 60) + (6 if cpg_reward else 0)) * 5
                if not inferred_history_dim:
                    inferred_history_dim = expected_history_dim

                inferred_latent_dim = new_state_dict.get('history_encoder.4.weight', torch.empty(0, 0)).shape[0] or 35
                inferred_wm_feature_dim = new_state_dict.get('wm_feature_encoder.0.weight', torch.empty(0, 0)).shape[1] or 512
                inferred_wm_latent_dim = new_state_dict.get('wm_feature_encoder.4.weight', torch.empty(0, 0)).shape[0] or 32
                critic_in = new_state_dict.get('critic.0.weight', torch.empty(0, 0)).shape[1]
                inferred_num_critic_obs = (critic_in - inferred_wm_latent_dim) if critic_in else 246

                self.history_dim = int(inferred_history_dim)
                self.wm_feature_dim = int(inferred_wm_feature_dim)

                # Build model with inferred dims
                self.actor_critic = ActorCriticRWM(
                    num_actor_obs=self.obs_dim,
                    num_critic_obs=int(inferred_num_critic_obs),
                    num_actions=self.num_actions,
                    encoder_hidden_dims=[256, 128],
                    wm_encoder_hidden_dims=[64, 64],
                    actor_hidden_dims=[256, 128, 64],
                    critic_hidden_dims=[512, 256, 128],
                    activation='elu',
                    init_noise_std=1.0,
                    latent_dim=int(inferred_latent_dim),
                    wm_latent_dim=int(inferred_wm_latent_dim),
                    cpg_reward_enabled=cpg_reward,
                    history_dim=self.history_dim,
                    wm_feature_dim=self.wm_feature_dim,
                    prop_dim=self.obs_dim,
                ).to(device)

                # Load the cleaned state dict with strict=False to handle dimension mismatches
                missing_keys, unexpected_keys = self.actor_critic.load_state_dict(new_state_dict, strict=False)

                if missing_keys:
                    print(f"Warning: Missing keys in state dict: {len(missing_keys)} keys")
                    print(f"First few missing: {missing_keys[:3]}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in state dict: {len(unexpected_keys)} keys")

                # Check if critical components were loaded
                critical_components = ['actor.0.weight', 'history_encoder.0.weight', 'wm_feature_encoder.0.weight']
                loaded_critical = all(comp in new_state_dict for comp in critical_components)
                if loaded_critical:
                    print("Critical model components loaded successfully")
                else:
                    print("Warning: Some critical components may not be loaded properly")

                self.model_loaded = True
                print("Model loaded successfully!")
                print(f"Training iteration: {checkpoint.get('iter', 'Unknown')}")

                # Store checkpoint for std parameter access
                self.checkpoint = checkpoint
                self._init_world_model_from_checkpoint(checkpoint)

            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using placeholder implementation")
                self.model_loaded = False
                # IMPORTANT: also build placeholder actor_critic so callers can still run.
                # Previously, AttributeError could happen if torch.load failed and this branch didn't
                # create `self.actor_critic` (because the placeholder construction lived only in the `else:`).
                self.history_dim = (48 if self.remove_dof_vel else 66) * 5 if cpg_reward else (42 if self.remove_dof_vel else 60) * 5
                self.wm_feature_dim = 512
                self.actor_critic = ActorCriticRWM(
                    num_actor_obs=self.obs_dim,
                    num_critic_obs=246,
                    num_actions=self.num_actions,
                    encoder_hidden_dims=[256, 128],
                    wm_encoder_hidden_dims=[64, 64],
                    actor_hidden_dims=[256, 128, 64],
                    critic_hidden_dims=[512, 256, 128],
                    activation='elu',
                    init_noise_std=1.0,
                    latent_dim=35,
                    wm_latent_dim=32,
                    cpg_reward_enabled=cpg_reward,
                    history_dim=self.history_dim,
                    wm_feature_dim=self.wm_feature_dim,
                    prop_dim=self.obs_dim,
                ).to(device)
        else:
            print("No valid model path provided - using placeholder implementation")
            self.model_loaded = False
            # Fallback dims
            self.history_dim = (48 if self.remove_dof_vel else 66) * 5 if cpg_reward else (42 if self.remove_dof_vel else 60) * 5
            self.wm_feature_dim = 512
            self.actor_critic = ActorCriticRWM(
                num_actor_obs=self.obs_dim,
                num_critic_obs=246,
                num_actions=self.num_actions,
                encoder_hidden_dims=[256, 128],
                wm_encoder_hidden_dims=[64, 64],
                actor_hidden_dims=[256, 128, 64],
                critic_hidden_dims=[512, 256, 128],
                activation='elu',
                init_noise_std=1.0,
                latent_dim=35,
                wm_latent_dim=32,
                cpg_reward_enabled=cpg_reward,
                history_dim=self.history_dim,
                wm_feature_dim=self.wm_feature_dim,
                prop_dim=self.obs_dim,
            ).to(device)

        # Set model to evaluation mode
        if self.model_loaded:
            self.actor_critic.eval()

        # Preallocated buffers (avoid per-step torch.tensor / torch.cat)
        self._prop_buffer = torch.zeros(1, self.obs_dim, device=self.device, dtype=torch.float32)
        self._prev_action_buf = torch.zeros(1, 18, device=self.device, dtype=torch.float32)
        # wm_feature_encoder 输出缓存：仅在 WM 做 obs_step 更新 wm_feature 后失效
        self._cached_wm_latent = None
        self._wm_latent_cache_valid = False

        # Initialize basic state
        self.step_count = 0
        self.wm_is_first = torch.ones(1, device=self.device)

        # World model runtime state (only set defaults if WM not loaded)
        if getattr(self, "world_model", None) is None:
            self.world_model = None
            self.wm_latent = None
            self.wm_feature = torch.zeros((1, 512), device=self.device)
            self.wm_update_interval = 5
            self.wm_action_history = torch.zeros((1, self.wm_update_interval, 18), device=self.device)
            self.wm_action = None

        # torch.compile / ONNX / warmup（依赖 wm_is_first、wm_action_history）
        if self.model_loaded:
            self._setup_runtime_optimizations()

    def _init_world_model_from_checkpoint(self, checkpoint):
        """Load WorldModelRWM weights and infer action history length."""
        try:
            from world_model.models import WorldModelRWM
            import yaml
            import pathlib
            import argparse

            wm_sd = checkpoint.get("world_model_dict", None)
            if wm_sd is None:
                print("Warning: checkpoint has no world_model_dict; wm_feature will stay zeros.")
                return

            # Infer prop_dim from encoder MLP weight (trained input dim)
            prop_dim = int(wm_sd["encoder._mlp.layers.Encoder_linear0.weight"].shape[1])
            self._wm_prop_dim = prop_dim

            # Infer wm action history dim from RSSM img_in weight
            # inp_dim = stoch*discrete + num_actions
            img_in_inp_dim = int(wm_sd["dynamics._img_in_layers.0.weight"].shape[1])
            stoch = 32
            discrete = 32
            num_actions = img_in_inp_dim - (stoch * discrete)
            update_interval = (num_actions // 18) if (num_actions % 18 == 0) else 5
            if num_actions % 18 != 0:
                print(
                    f"Warning: inferred wm num_actions={num_actions} not multiple of 18; "
                    "defaulting update_interval=5"
                )

            # Load default config and override key fields
            cfg_path = pathlib.Path(__file__).parent / "world_model" / "configs.yaml"
            configs = yaml.safe_load(cfg_path.read_text())
            defaults = dict(configs.get("defaults", {}))
            defaults["device"] = str(self.device)
            defaults["num_actions"] = int(num_actions)
            # Real robot has no camera; keep model but disable camera usage at runtime.
            # Avoid pri_obs decoding to keep shapes simple.
            defaults["decode_pri_obs"] = False
            # Ensure encoder/decoder MLPs are built on CPU when CUDA is unavailable.
            defaults["encoder"] = dict(defaults.get("encoder", {}))
            defaults["decoder"] = dict(defaults.get("decoder", {}))
            defaults["encoder"]["device"] = str(self.device)
            defaults["decoder"]["device"] = str(self.device)
            # Some YAML scalars like 1e-4 may be parsed as strings; cast for optimizer init
            for k in ("model_lr", "opt_eps", "grad_clip", "weight_decay"):
                if k in defaults and isinstance(defaults[k], str):
                    try:
                        defaults[k] = float(defaults[k])
                    except Exception:
                        pass

            wm_config = argparse.Namespace(**defaults)
            obs_shape = {"prop": (prop_dim,), "image": (64, 64, 1)}

            wm = WorldModelRWM(
                wm_config,
                obs_shape,
                use_camera=False,
                device=torch.device(self.device) if isinstance(self.device, str) else self.device,
            ).to(self.device)
            wm.eval()

            # Strip possible DDP prefix
            wm_sd_stripped = {}
            for k, v in wm_sd.items():
                key = k[7:] if k.startswith("module.") else k
                wm_sd_stripped[key] = v

            missing, unexpected = wm.load_state_dict(wm_sd_stripped, strict=False)
            if missing:
                print(f"World model loaded with missing keys: {len(missing)} (strict=False)")
            if unexpected:
                print(f"World model loaded with unexpected keys: {len(unexpected)} (strict=False)")

            self.world_model = wm
            self.wm_update_interval = int(update_interval)
            self.wm_action_history = torch.zeros((1, self.wm_update_interval, 18), device=self.device)
            self.wm_action = None
            self.wm_latent = None
            self.wm_feature = torch.zeros((1, int(getattr(wm_config, "dyn_deter", 512))), device=self.device)

            print(
                f"World model initialized: prop_dim={prop_dim}, "
                f"update_interval={self.wm_update_interval}, wm_feature_dim={self.wm_feature.shape[1]}"
            )

        except Exception as e:
            print(f"Warning: failed to init world model from checkpoint: {e}")
            self.world_model = None

    def _compile_module_safe(self, module, name):
        if not hasattr(torch, "compile"):
            return module
        try:
            return torch.compile(module, mode=WM_COMPILE_MODE)
        except Exception as e:
            print(f"  {name}: torch.compile 跳过 ({e})")
            return module

    def _export_wm_encoder_onnx(self, path, prop_dim):
        """导出 WM MultiEncoder 为 ONNX（固定 batch=1）。"""
        wm = self.world_model
        wrap = EncoderInferenceWrapper(wm.encoder).eval().to(self.device)
        dummy_prop = torch.zeros(1, prop_dim, device=self.device, dtype=torch.float32)
        dummy_first = torch.ones(1, device=self.device, dtype=torch.float32)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.onnx.export(
            wrap,
            (dummy_prop, dummy_first),
            path,
            input_names=["prop", "is_first"],
            output_names=["embed"],
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  WM encoder ONNX 已导出: {path}")

    def _wm_encoder_forward(self, prop_tensor):
        """prop_tensor: [1, prop_dim]，与原先 encoder(wm_obs) 等价。"""
        if self._ort_encoder_session is not None:
            p = prop_tensor.detach().cpu().numpy().astype(np.float32)
            f = self.wm_is_first.detach().cpu().numpy().astype(np.float32)
            out = self._ort_encoder_session.run(None, {"prop": p, "is_first": f})[0]
            return torch.from_numpy(np.asarray(out, dtype=np.float32)).to(self.device)
        if self._wm_encoder_wrapper is not None:
            return self._wm_encoder_wrapper(prop_tensor, self.wm_is_first)
        return self.world_model.encoder({"prop": prop_tensor, "is_first": self.wm_is_first})

    def _warmup_wm_inference(self, prop_dim):
        """触发 torch.compile / ORT 首次构图，避免第一步实机卡顿。"""
        wm = self.world_model
        if wm is None or prop_dim is None:
            return
        saved_lat = self.wm_latent
        saved_feat = self.wm_feature
        saved_first = self.wm_is_first.clone()
        if self.wm_action is None:
            self.wm_action = self.wm_action_history.reshape(1, -1).clone()
        try:
            # 与 torch.compile/inductor 兼容：inference_mode 在部分 CPU 版 PyTorch 会触发
            # "Inference tensors do not track version counter"，部署推理用 no_grad 即可。
            with torch.no_grad():
                dummy = torch.zeros(1, prop_dim, device=self.device, dtype=torch.float32)
                obs_fn = self._compiled_obs_step or wm.dynamics.obs_step
                for _ in range(3):
                    emb = self._wm_encoder_forward(dummy)
                    self.wm_latent, _ = obs_fn(
                        self.wm_latent,
                        self.wm_action,
                        emb,
                        self.wm_is_first,
                        True,
                    )
        except Exception as e:
            print(f"  WM warmup 跳过: {e}")
        finally:
            self.wm_latent = saved_lat
            self.wm_feature = saved_feat
            self.wm_is_first.copy_(saved_first)

    def _setup_runtime_optimizations(self):
        """torch.compile（策略+WM）、可选 ONNX encoder、warmup。"""
        if not getattr(self, "model_loaded", False):
            return

        if POLICY_OPT_TORCH_COMPILE and hasattr(torch, "compile"):
            try:
                self.actor_critic.history_encoder = self._compile_module_safe(
                    self.actor_critic.history_encoder, "history_encoder"
                )
                self.actor_critic.wm_feature_encoder = self._compile_module_safe(
                    self.actor_critic.wm_feature_encoder, "wm_feature_encoder"
                )
                self.actor_critic.actor = self._compile_module_safe(
                    self.actor_critic.actor, "actor"
                )
                print("  Policy: history_encoder / wm_feature_encoder / actor 已尝试 torch.compile")
            except Exception as e:
                print(f"  Policy torch.compile 失败: {e}")

        wm = getattr(self, "world_model", None)
        if wm is None:
            return

        prop_dim = getattr(self, "_wm_prop_dim", None)
        if prop_dim is None:
            print("  WM 优化跳过: 无 _wm_prop_dim")
            return

        # --- ONNX encoder（优先）---
        if WM_OPT_ONNX_ENCODER:
            try:
                import onnxruntime as ort
            except ImportError:
                print("  WM ONNX: 未安装 onnxruntime，使用 PyTorch encoder")
                ort = None
            if ort is not None:
                onnx_path = WM_ONNX_EXPORT_PATH
                if onnx_path is None and self._model_path:
                    onnx_path = os.path.join(
                        os.path.dirname(os.path.abspath(self._model_path)),
                        "wm_encoder.onnx",
                    )
                elif onnx_path is None:
                    onnx_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "wm_encoder.onnx",
                    )
                try:
                    if not os.path.isfile(onnx_path):
                        self._export_wm_encoder_onnx(onnx_path, prop_dim)
                    so = ort.SessionOptions()
                    try:
                        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    except Exception:
                        pass
                    nt = int(OPTIM_TORCH_NUM_THREADS) if OPTIM_TORCH_NUM_THREADS is not None else 2
                    so.intra_op_num_threads = max(1, nt)
                    so.inter_op_num_threads = 1
                    self._ort_encoder_session = ort.InferenceSession(
                        onnx_path, so, providers=["CPUExecutionProvider"]
                    )
                    print(f"  WM encoder: ONNX Runtime ({onnx_path})")
                except Exception as e:
                    print(f"  WM ONNX 失败，回退 PyTorch: {e}")
                    self._ort_encoder_session = None

        # --- torch.compile：encoder 包装 + obs_step（无 ONNX 时）---
        if self._ort_encoder_session is None and WM_OPT_TORCH_COMPILE and hasattr(torch, "compile"):
            try:
                wrap = EncoderInferenceWrapper(wm.encoder).eval().to(self.device)
                self._wm_encoder_wrapper = torch.compile(wrap, mode=WM_COMPILE_MODE)
                print("  WM encoder: torch.compile(EncoderInferenceWrapper)")
            except Exception as e:
                print(f"  WM encoder compile 失败，使用 eager: {e}")
                self._wm_encoder_wrapper = EncoderInferenceWrapper(wm.encoder).eval().to(self.device)
            if WM_OPT_COMPILE_OBS_STEP:
                try:
                    self._compiled_obs_step = torch.compile(
                        wm.dynamics.obs_step, mode=WM_COMPILE_MODE
                    )
                    print("  WM dynamics.obs_step: torch.compile")
                except Exception as e:
                    print(f"  WM obs_step compile 失败，使用 eager: {e}")
                    self._compiled_obs_step = None
            else:
                self._compiled_obs_step = None
                print("  WM dynamics.obs_step: 未编译 (WM_OPT_COMPILE_OBS_STEP=False)")
        elif self._ort_encoder_session is None:
            # 未启用 compile：仍用 wrapper 统一接口（与 ONNX 路径行为一致）
            self._wm_encoder_wrapper = EncoderInferenceWrapper(wm.encoder).eval().to(self.device)

        self._warmup_wm_inference(prop_dim)

    def get_inference_policy(self):
        """Get policy for inference"""
        return self._policy_inference

    def _policy_inference(self, obs, history, wm_feature):
        """
        Policy inference using loaded RWM model.

        Args:
            obs: Current observation tensor [batch_size, obs_dim]
            history: Trajectory history tensor [batch_size, history_dim]
            wm_feature: World model features [batch_size, feature_dim]

        Returns:
            actions: Action tensor [batch_size, action_dim]
        """
        if not self.model_loaded:
            # Fallback to safe placeholder actions
            batch_size = obs.shape[0]
            action_dim = 18
            return torch.randn(batch_size, action_dim, device=self.device) * 0.01

        try:
            # Use the full ActorCriticRWM model for inference（no_grad 与 torch.compile 更兼容）
            with torch.no_grad():
                # Get deterministic actions by manually constructing the actor input
                proprioceptive_obs = obs  # obs is already proprioceptive in our case

                # Ensure inputs are batched
                if proprioceptive_obs.dim() == 1:
                    proprioceptive_obs = proprioceptive_obs.unsqueeze(0)
                if history.dim() == 1:
                    history = history.unsqueeze(0)
                if wm_feature.dim() == 1:
                    wm_feature = wm_feature.unsqueeze(0)

                # Manually construct actor input like in act() method with proprioception_only=True
                latent_vector = self.actor_critic.history_encoder(history)
                command = proprioceptive_obs[:, 6:9]  # Extract commands from proprioceptive obs (positions 6:9)
                # wm_feature 在非 WM 更新步不变，可复用 wm_feature_encoder 输出
                if self._wm_latent_cache_valid and self._cached_wm_latent is not None:
                    wm_latent_vector = self._cached_wm_latent
                else:
                    wm_latent_vector = self.actor_critic.wm_feature_encoder(wm_feature)
                    self._cached_wm_latent = wm_latent_vector
                    self._wm_latent_cache_valid = True

                # Concatenate: latent_vector + command + wm_latent_vector
                concat_observations = torch.concat((latent_vector, command, wm_latent_vector), dim=-1)

                # Get deterministic actions (mean of policy)
                actions_mean = self.actor_critic.actor(concat_observations)

                # Remove batch dimension if input was not batched
                if obs.dim() == 1:
                    actions_mean = actions_mean.squeeze(0)

                return actions_mean

        except Exception as e:
            print(f"Policy inference error: {e}")
            print("Falling back to safe actions")
            batch_size = obs.shape[0] if obs.dim() > 1 else 1
            return torch.zeros(batch_size, 18, device=self.device)

    def update_world_model(self, obs_dict, prev_action=None):
        """
        Update world model state using loaded model.

        Args:
            obs_dict: Observation dictionary with 'prop', 'is_first', etc.
            prev_action: previous action in radians, shape (18,) or tensor (1,18). Used to build 5-step action history.

        Returns:
            wm_feature: World model features
        """
        # If world model not available, keep zeros (better than random noise)
        if self.world_model is None:
            if 'is_first' in obs_dict:
                obs_dict['is_first'][:] = 0
            self.step_count += 1
            return self.wm_feature

        try:
            with torch.no_grad():
                # Update action history with prev_action (like playMBRL, wm_action is history of last K actions)
                if prev_action is None:
                    self._prev_action_buf.zero_()
                elif isinstance(prev_action, torch.Tensor):
                    pa = prev_action.to(self.device, dtype=torch.float32, non_blocking=False)
                    if pa.dim() == 1:
                        self._prev_action_buf[0].copy_(pa)
                    else:
                        self._prev_action_buf.copy_(pa)
                else:
                    self._prev_action_buf[0].copy_(
                        torch.from_numpy(np.asarray(prev_action, dtype=np.float32))
                    )
                prev_action_t = self._prev_action_buf

                # 移位写入新动作（避免 torch.cat；用 clone 避免重叠内存 undefined）
                ha = self.wm_action_history
                ha[:, :-1] = ha[:, 1:].clone()
                ha[:, -1, :].copy_(prev_action_t)
                self.wm_action = ha.reshape(1, -1)

                # Update world model latent every K steps (matches training update_interval)
                if self.step_count % self.wm_update_interval == 0:
                    prop_t = obs_dict["prop"].to(self.device)
                    wm_embed = self._wm_encoder_forward(prop_t)
                    obs_fn = self._compiled_obs_step or self.world_model.dynamics.obs_step
                    self.wm_latent, _ = obs_fn(
                        self.wm_latent,
                        self.wm_action,
                        wm_embed,
                        self.wm_is_first,
                        True,
                    )
                    self.wm_feature = self.world_model.dynamics.get_deter_feat(self.wm_latent)
                    self.wm_is_first[:] = 0
                    # wm_feature 更新后需重算 policy 侧的 wm_latent_vector
                    self._wm_latent_cache_valid = False

        except Exception as e:
            print(f"World model update error: {e}")
            # keep last wm_feature

        self.step_count += 1
        return self.wm_feature

    def reset(self):
        """Reset inference state"""
        self.step_count = 0
        self._wm_latent_cache_valid = False
        self._cached_wm_latent = None
        print("RWM inference state reset")


def read_imu(q_imu):
    """IMU reading process - Simplified like CPGs (no while loop, callback-driven)"""
    # Re-import modules for subprocess
    import lib.device_model as deviceModel
    from lib.data_processor.roles.jy901s_dataProcessor import JY901SDataProcessor
    from lib.protocol_resolver.roles.wit_protocol_resolver import WitProtocolResolver
    
    # Define callback function for subprocess
    def onUpdate(deviceModel):
        """IMU data update callback"""
        try:
            IMU_data = np.array([deviceModel.getDeviceData("angleX"),
                                deviceModel.getDeviceData("angleY"),
                                deviceModel.getDeviceData("angleZ"),
                                deviceModel.getDeviceData("gyroX"),
                                deviceModel.getDeviceData("gyroY"),
                                deviceModel.getDeviceData("gyroZ"),
                                deviceModel.getDeviceData("accX"),
                                deviceModel.getDeviceData("accY"),
                                deviceModel.getDeviceData("accZ")])
            q_imu.put(IMU_data)
        except Exception as e:
            print(f"IMU onUpdate error: {e}")

    try:
        device = deviceModel.DeviceModel(
            "JY901",
            WitProtocolResolver(),
            JY901SDataProcessor(),
            "51_0"
        )

        if (platform.system().lower() == 'linux'):
            device.serialConfig.portName = '/dev/ttyUSB0'#'/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0'
        else:
            device.serialConfig.portName = "COM39"
        device.serialConfig.baud = 230400
        device.ADDR = 0x50
        device.openDevice()

        readConfig(device)
        device.dataProcessor.onVarChanged.append(onUpdate)
        
        print("IMU process: Device opened successfully, callback registered")
        # No while loop - rely on callback mechanism like CPGs
        
    except Exception as e:
        print(f"IMU process error: {e}")


def onUpdate(deviceModel):
    """IMU data update callback"""
    global IMU_data
    IMU_data = np.array([deviceModel.getDeviceData("angleX"),
                        deviceModel.getDeviceData("angleY"),
                        deviceModel.getDeviceData("angleZ"),
                        deviceModel.getDeviceData("gyroX"),
                        deviceModel.getDeviceData("gyroY"),
                        deviceModel.getDeviceData("gyroZ"),
                        deviceModel.getDeviceData("accX"),
                        deviceModel.getDeviceData("accY"),
                        deviceModel.getDeviceData("accZ")])
    q_imu_1.put(IMU_data)


def create_observation_from_real_robot(
    servos,
    q_imu,
    step,
    history_length=5,
    cpg_reward=False,
    previous_actions=None,
    imu_timeout_sec: float = 10.0,
    imu_drain_max: int = 3,
    imu_reinit_period_sec=None,
):
    """
    Create observation from real robot sensors, matching the structure used in simulation.
    Based on hexapodMBRL.compute_observations()
    """
    # Read servo positions and velocities (joints are in degrees, convert to simulation units)
    position_Read = servos.read_all_positions()  # degrees
    theta_rad = position_Read * math.pi / 180.0  # convert to radians

    velocity_rads = np.zeros(18)
    if not remove_dof_vel:
        try:
            velocity_Read = servos.read_all_velocity()
            velocity_rads = velocity_Read * 0.229 * 2 * math.pi / 60.0
        except Exception as e:
            print(f"Warning: Failed to read joint velocities: {e}, using zeros")
            velocity_rads = np.zeros(18)

    # Read IMU data like CPGs: blocking get with timeout, then drain queue for latest.
    # Online training may stall the main loop (WM training/torch compile). In that case
    # the queue can be temporarily empty; to avoid long blocking we support a short timeout
    # and fallback to the last received IMU measurement.
    #
    # First-call robustness: if we haven't got any IMU initialization/cache yet, we
    # temporarily allow a longer timeout so drift-correction init isn't based on zeros.
    effective_timeout_sec = float(imu_timeout_sec)
    if (not hasattr(create_observation_from_real_robot, "imu_init")) and (not hasattr(create_observation_from_real_robot, "last_imu_data")):
        effective_timeout_sec = max(effective_timeout_sec, 1.0)
    try:
        IMU_data = q_imu.get(True, effective_timeout_sec)
        # Drain any additional queued measurements without blocking.
        # This matters online training: when the main loop stalls, a lot of IMU
        # messages may accumulate and draining them with timeouts can introduce
        # noticeable latency/jitter.
        drain_cnt = 0
        while drain_cnt < int(imu_drain_max):
            try:
                IMU_data = q_imu.get(False)  # non-blocking
                drain_cnt += 1
            except Exception:
                break
        # Cache last successful IMU
        create_observation_from_real_robot.last_imu_data = np.asarray(IMU_data, dtype=np.float32).copy()
    except:
        # If no IMU data available, use zeros
        if hasattr(create_observation_from_real_robot, "last_imu_data"):
            IMU_data = create_observation_from_real_robot.last_imu_data.copy()
        else:
            print("Warning: No IMU data available, using zeros")
            IMU_data = np.zeros(9, dtype=np.float32)

    # Handle IMU initialization (for drift correction)
    if not hasattr(create_observation_from_real_robot, 'imu_init'):
        create_observation_from_real_robot.imu_init = IMU_data[0:3].copy()
        create_observation_from_real_robot.imu_init_time = time.time()
        print(f"IMU initialized with angles: roll={create_observation_from_real_robot.imu_init[0]:.3f}, pitch={create_observation_from_real_robot.imu_init[1]:.3f}, yaw={create_observation_from_real_robot.imu_init[2]:.3f}")

    # Method 1: Basic drift correction (subtract initial values)
    IMU_data_corrected = IMU_data[0:3] - create_observation_from_real_robot.imu_init

    # Method 2: Periodic re-initialization every 10 seconds to reduce drift
    current_time = time.time()
    if imu_reinit_period_sec is not None and hasattr(create_observation_from_real_robot, 'imu_init_time'):
        time_since_init = current_time - create_observation_from_real_robot.imu_init_time
        if time_since_init > float(imu_reinit_period_sec):
            print(f"Re-initializing IMU after {time_since_init:.1f} seconds to reduce drift")
            create_observation_from_real_robot.imu_init = IMU_data[0:3].copy()
            create_observation_from_real_robot.imu_init_time = current_time
            IMU_data_corrected = np.zeros(3)  # Reset to zero after re-init
        else:
            IMU_data_corrected = IMU_data[0:3] - create_observation_from_real_robot.imu_init

    IMU_data[0:3] = IMU_data_corrected

    # Yaw offset: add pi/2 (90 deg) to match simulation initial yaw
    IMU_data[2] = IMU_data[2] + 90.0

    # Convert IMU angles to radians
    roll, pitch, yaw = IMU_data[0:3] * math.pi / 180.0
    # 将roll pitch yaw保存到observation_output.txt中

    # Create observation vector matching simulation structure
    # Based on hexapodMBRL.compute_observations()

    # Base angular velocity (from IMU gyro, rad/s)
    base_ang_vel = np.array([IMU_data[3], IMU_data[4], IMU_data[5]])  # gyro data

    # Projected gravity (from IMU orientation)
    # 与仿真一致：仿真中 gravity_vec = [0,0,-1]（世界系 z 向下），projected_gravity = quat_rotate_inverse(base_quat, gravity_vec)，
    # 水平时 body 系中重力为 [0,0,-1]，即第三项为负。JY901 的 angleX/Y/Z 为 roll/pitch/yaw(度)，此处用欧拉 ZYX 推导
    # 的 body 系重力方向，第三项取负以匹配仿真（水平时 ≈ [0, 0, -1]，若仿真用 9.81 量纲则再乘 9.81）。
    projected_gravity = np.array([
        -math.sin(pitch),  # x
        math.sin(roll) * math.cos(pitch),  # y
        -math.cos(roll) * math.cos(pitch)  # z: 取负以与仿真一致（仿真水平时为负）
    ])
    
    # print("projected gravity: ", projected_gravity)

    # Commands (fixed for real robot - can be modified for different behaviors)
    commands = np.array([0.0, 0.055, 1.57])  # [lin_vel_x, lin_vel_y, ang_vel_yaw]

    # Joint positions (convert servo angles to simulation joint angles matching action ranges)
    # Map servo angles to simulation joint angles that match the action space
    sim_joint_angles_rad = servo_angles_to_sim_angles(position_Read)

    # Joint positions relative to default pose (neutral pose is zeros in simulation)
    default_dof_pos = np.zeros(18)  # Neutral pose in simulation space
    dof_pos_scaled = (sim_joint_angles_rad - default_dof_pos) * 1.0  # obs_scales.dof_pos = 1.0

    # Joint velocities (map servo velocities to simulation joint velocities)
    # The velocity direction needs to match the joint angle mapping
    # If servo_angle = f(sim_angle), then servo_vel = f'(sim_angle) * sim_vel
    # From the mappings:
    # Hip joints: servo = 180 - sim => sim_vel = -servo_vel
    # Left knee:  servo = 180 - sim => sim_vel = -servo_vel
    # Right knee: servo = 180 - sim => sim_vel = -servo_vel
    # Left ankle:  servo = 60 + sim => sim_vel = +servo_vel
    # Right ankle: servo = 60 - sim => sim_vel = -servo_vel

    # Reorder velocities to match simulation joint order and apply direction corrections
    sim_joint_velocities = np.zeros(18)

    # l1 leg (servo 0-2 -> sim 0-2)
    sim_joint_velocities[0] = -velocity_rads[0]  # Hip: negative
    sim_joint_velocities[1] = -velocity_rads[1]  # Left knee: negative
    sim_joint_velocities[2] = velocity_rads[2]   # Left ankle: positive

    # l2 leg (servo 6-8 -> sim 3-5)
    sim_joint_velocities[3] = -velocity_rads[6]  # Hip: negative
    sim_joint_velocities[4] = -velocity_rads[7]  # Left knee: negative
    sim_joint_velocities[5] = velocity_rads[8]   # Left ankle: positive

    # l3 leg (servo 12-14 -> sim 6-8)
    sim_joint_velocities[6] = -velocity_rads[12] # Hip: negative
    sim_joint_velocities[7] = -velocity_rads[13] # Left knee: negative
    sim_joint_velocities[8] = velocity_rads[14]  # Left ankle: positive

    # r1 leg (servo 3-5 -> sim 9-11)
    sim_joint_velocities[9] = -velocity_rads[3]    # Hip: servo=180-sim → sim_vel=-servo_vel
    sim_joint_velocities[10] = -velocity_rads[4]   # Right knee: servo=180-sim → sim_vel=-servo_vel
    sim_joint_velocities[11] = -velocity_rads[5]   # Right ankle: servo=60-sim → sim_vel=-servo_vel

    # r2 leg (servo 9-11 -> sim 12-14)
    sim_joint_velocities[12] = -velocity_rads[9]   # Hip
    sim_joint_velocities[13] = -velocity_rads[10]  # Right knee
    sim_joint_velocities[14] = -velocity_rads[11]  # Right ankle

    # r3 leg (servo 15-17 -> sim 15-17)
    sim_joint_velocities[15] = -velocity_rads[15]  # Hip
    sim_joint_velocities[16] = -velocity_rads[16]  # Right knee
    sim_joint_velocities[17] = -velocity_rads[17]  # Right ankle

    dof_vel_scaled = sim_joint_velocities * 0.05  # obs_scales.dof_vel = 1.0, but scaled down

    # Actions (current actions, placeholder for now)
    obs_action = np.zeros(18)

    # Create proprioceptive features for RWM world model (order matches hexapodMBRL.compute_observations)
    # Order: base_ang_vel, projected_gravity, commands, dof_pos [, dof_vel], [phase_bool], obs_action
    if remove_dof_vel:
        proprioceptive_obs = np.concatenate([base_ang_vel*0.030, projected_gravity, commands, dof_pos_scaled])
    else:
        proprioceptive_obs = np.concatenate([base_ang_vel*0.030, projected_gravity, commands, dof_pos_scaled, dof_vel_scaled])

    prev_actions = previous_actions if previous_actions is not None else np.zeros(18)

    # Add phase_bool then previous_actions when cpg_reward (same order as hexapodMBRL: phase_bool before obs_action)
    if cpg_reward:
        # Load phase bool data (similar to simulation)
        if not hasattr(create_observation_from_real_robot, 'phase_bool'):
            # Load phase bool data from CSV file
            import csv
            phase_bool_file = './world_model/contact_data.csv'#'./world_model/joint_angles_tetrapod_phase_bool_normal.csv'
            try:
                with open(phase_bool_file, 'r') as f:
                    reader = csv.reader(f)
                    phase_bool_data = []
                    for row in reader:
                        phase_bool_data.append([float(x) for x in row])
                    create_observation_from_real_robot.phase_bool = np.array(phase_bool_data)
                    print(f"Loaded phase_bool data with shape: {create_observation_from_real_robot.phase_bool.shape}")
            except Exception as e:
                print(f"Warning: Failed to load phase_bool data: {e}, using zeros")
                create_observation_from_real_robot.phase_bool = np.zeros((100, 6))

        # Get current phase bool based on step (cycle through the data)
        cpg_idx = step % create_observation_from_real_robot.phase_bool.shape[0]
        phase_bool_current = create_observation_from_real_robot.phase_bool[cpg_idx]

        # Concatenate phase_bool_current first, then previous_actions (matching compute_observations)
        proprioceptive_obs = np.concatenate([proprioceptive_obs, phase_bool_current])#, prev_actions])  # 51 dims
    else:
        proprioceptive_obs = np.concatenate([proprioceptive_obs])#, prev_actions])

    # Create obs_without_command for history building (matching training logic)
    # Order must match hexapodMBRL.compute_observations(): ... dof_pos [, dof_vel], phase_bool, obs_action
    prev_actions = previous_actions if previous_actions is not None else np.zeros(18)
    if remove_dof_vel:
        base_part = np.concatenate([base_ang_vel*0.030, projected_gravity, dof_pos_scaled])
    else:
        base_part = np.concatenate([base_ang_vel*0.030, projected_gravity, dof_pos_scaled, dof_vel_scaled])
    if cpg_reward:
        obs_without_command = np.concatenate([base_part, phase_bool_current, prev_actions])  # ... phase_bool then previous_actions
    else:
        obs_without_command = np.concatenate([base_part, prev_actions])

    # For policy observation, use full proprioceptive_obs (includes commands)
    policy_obs = proprioceptive_obs.copy()

    return policy_obs.astype(np.float32), obs_without_command.astype(np.float32), position_Read, IMU_data


def run_imu_test_only():
    """
    仅测试 IMU 轴范围：关节不运动，只读 IMU。
    机器人会先回到 neutral 姿态，之后不再下发关节指令；你手动移动机器人即可观察
    IMU_data_corrected（roll/pitch/yaw 校正后）与 projected_gravity 的读数变化。
    用于确认 IMU 各轴方向与量纲。
    """
    print("\n" + "=" * 60)
    print("📐 IMU 轴范围测试模式：关节不动，请手动移动机器人观察读数")
    print("=" * 60 + "\n")

    # Step 1–4: Servos（与主流程一致）
    print("📍 Step 1: Initializing Servos...")
    try:
        servos = Servos()
        print("   ✓ Servos object created")
    except Exception as e:
        print(f"   ❌ Failed to create Servos: {e}")
        raise

    print("\n📍 Step 2: Reading voltage...")
    try:
        voltage = servos.read_voltage(1)
        print(f"   ✓ Voltage: {voltage}V")
        if voltage < ROBOT_CONFIG['control']['voltage_threshold']:
            print(f"   ⚠️  WARNING: Voltage too low!")
    except Exception as e:
        print(f"   ❌ Failed to read voltage: {e}")
        raise

    print("\n📍 Step 3: Setting position control mode...")
    try:
        servos.set_position_control()
        print("   ✓ Position control mode set")
    except Exception as e:
        print(f"   ❌ Failed to set position control: {e}")
        raise

    position_all = range(18)
    print("\n📍 Step 4: Enabling torque...")
    try:
        servos.enable_torque(position_all)
        print("   ✓ Torque enabled for all servos")
    except Exception as e:
        print(f"   ❌ Failed to enable torque: {e}")
        raise

    # Step 5: IMU
    print("\n📍 Step 5: Initializing IMU process...")
    q_imu = Queue()
    imu_process = Process(target=read_imu, args=(q_imu,))
    imu_process.daemon = True
    imu_process.start()
    print("   ✓ IMU process started")

    imu_ready = False
    for i in range(20):
        time.sleep(0.1)
        if not q_imu.empty():
            imu_ready = True
            print("   ✓ IMU data detected")
            break
    if not imu_ready:
        print("   ⚠️  Warning: No IMU data received. Check USB and permissions.")

    # Step 6: 回到 neutral，之后不再写关节
    print("\n📍 Step 6: Moving to neutral position (no further joint commands will be sent)...")
    neutral_angles = ROBOT_CONFIG['neutral_angles']
    real_angles = radians_to_degrees(neutral_angles * np.pi / 180.0)
    ticks = angles_to_ticks(real_angles)
    try:
        servos.Robot_initialize(real_angles)
        print("   ✓ Robot at neutral position")
    except Exception as e:
        print(f"   ❌ Failed to initialize robot position: {e}")
        raise

    # 可选：写 IMU 测试日志
    output_dir = VALIDATION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    imu_test_file = os.path.join(output_dir, "imu_test_axes.txt")
    f_log = open(imu_test_file, 'w')
    f_log.write("step\troll_corrected\tpitch_corrected\tyaw_corrected\tproj_grav_x\tproj_grav_y\tproj_grav_z\n")

    print("\n" + "=" * 60)
    print("开始 IMU 读数循环：请手动改变机器人朝向，观察下方 IMU_data_corrected 与 projected_gravity。Ctrl+C 退出。")
    print("  IMU_data_corrected = (roll, pitch, yaw) 单位度；projected_gravity 与仿真一致，水平时 z≈-1。")
    print("=" * 60 + "\n")

    step = 0
    print_interval = 0.15
    last_print = time.time()

    try:
        while step < MAX_STEPS:
            start_time = time.time()
            try:
                real_obs, obs_without_command, position_Read, IMU_data = create_observation_from_real_robot(
                    servos, q_imu, step, history_length=5, cpg_reward=cpg_reward, previous_actions=None
                )
            except Exception as e:
                print(f"   ⚠️  Read sensors error: {e}")
                time.sleep(0.2)
                step += 1
                continue

            # IMU_data_corrected：校正后的 roll/pitch/yaw（yaw 未加 90 的版本）
            roll_corr = float(IMU_data[0])
            pitch_corr = float(IMU_data[1])
            yaw_corr = float(IMU_data[2]) - 90.0  # 去掉观测里加的 90°

            # projected_gravity 在 real_obs 中位于 [3:6]
            proj_grav = real_obs[3:6]

            now = time.time()
            if now - last_print >= print_interval:
                last_print = now
                print("---")
                print("  IMU_data_corrected (°)   roll    pitch   yaw(未+90)")
                print("  values                  {:7.2f}  {:7.2f}  {:7.2f}".format(roll_corr, pitch_corr, yaw_corr))
                print("  projected_gravity       x       y       z")
                print("  values                  {:7.3f}  {:7.3f}  {:7.3f}".format(proj_grav[0], proj_grav[1], proj_grav[2]))
                print("  (* 水平时 z 应为负，与仿真一致)")

            f_log.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(
                step, roll_corr, pitch_corr, yaw_corr, proj_grav[0], proj_grav[1], proj_grav[2]))
            f_log.flush()

            elapsed = time.time() - start_time
            while (time.time() - start_time) < TARGET_DT:
                pass
            step += 1

    except KeyboardInterrupt:
        print("\n已退出 IMU 测试")
    finally:
        f_log.close()
        print("  IMU 测试日志已保存: {}".format(imu_test_file))
        try:
            imu_process.terminate()
            imu_process.join(timeout=2.0)
            if imu_process.is_alive():
                imu_process.kill()
        except Exception:
            pass


def run_replay_sim_dof(sim_data_path=None):
    """
    使用仿真观测文件 sim_data/wm_obs_prop.txt 中的 dof_pos 作为关节指令回放。
    不跑策略，按步数依次取文件中的 dof_pos，转换为舵机角度后下发。
    若步数超过文件行数则循环使用（step % 帧数）。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if sim_data_path is None:
        sim_data_path = os.path.join(current_dir, 'sim_data', 'wm_obs_prop.txt')
    if not os.path.isfile(sim_data_path):
        print(f"Error: Sim data file not found: {sim_data_path}")
        return

    print("\n" + "=" * 60)
    print("Replay sim dof_pos: 使用仿真观测中的 dof_pos 作为关节指令")
    print("=" * 60 + "\n")

    dof_pos_list = load_sim_dof_pos(sim_data_path)
    if not dof_pos_list:
        print("Error: No valid dof_pos blocks in file.")
        return
    print(f"Loaded {len(dof_pos_list)} frames from {sim_data_path}")

    # 与主流程一致的初始化
    print("📍 Step 1: Initializing Servos...")
    try:
        servos = Servos()
        print("   ✓ Servos object created")
    except Exception as e:
        print(f"   ❌ Failed to create Servos: {e}")
        raise

    print("\n📍 Step 2: Reading voltage...")
    try:
        voltage = servos.read_voltage(1)
        print(f"   ✓ Voltage: {voltage}V")
        if voltage < ROBOT_CONFIG['control']['voltage_threshold']:
            print(f"   ⚠️  WARNING: Voltage too low!")
    except Exception as e:
        print(f"   ❌ Failed to read voltage: {e}")
        raise

    print("\n📍 Step 3: Setting position control mode...")
    try:
        servos.set_position_control()
        print("   ✓ Position control mode set")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        raise

    position_all = range(18)
    print("\n📍 Step 4: Enabling torque...")
    try:
        servos.enable_torque(position_all)
        print("   ✓ Torque enabled")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        raise

    print("\n📍 Step 5: Initializing IMU process...")
    q_imu = Queue()
    imu_process = Process(target=read_imu, args=(q_imu,))
    imu_process.daemon = True
    imu_process.start()
    print("   ✓ IMU process started")
    imu_ready = False
    for _ in range(20):
        time.sleep(0.1)
        if not q_imu.empty():
            imu_ready = True
            print("   ✓ IMU data detected")
            break
    if not imu_ready:
        print("   ⚠️  Warning: No IMU data received, observations will use zeros for IMU.")

    print("\n📍 Step 6: Moving to neutral position...")
    neutral_angles = ROBOT_CONFIG['neutral_angles']
    real_angles = radians_to_degrees(neutral_angles * np.pi / 180.0)
    ticks = angles_to_ticks(real_angles)
    try:
        servos.Robot_initialize(real_angles)
        print("   ✓ Robot at neutral")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        raise

    angle_limits = ROBOT_CONFIG['angle_limits']
    output_dir = VALIDATION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    actions_file = os.path.join(output_dir, "actions_output.txt")
    observations_file = os.path.join(output_dir, "observations_output.txt")
    print(f"\nLogging to {actions_file} and {observations_file}")
    print("开始按 sim dof_pos 回放，Ctrl+C 退出...")

    step = 0
    try:
        with open(actions_file, 'w') as f_actions, open(observations_file, 'w') as f_obs:
            f_actions.write("step\tactions_18\tangles_18\tticks_18\n")
            f_obs.write("step\tbase_ang_vel_3\tprojected_gravity_3\tcommands_3\tdof_pos_18\tdof_vel_18\tobs_action_18\troll\tpitch\tyaw\n")

            while step < MAX_STEPS:
                start_time = time.time()
                frame_idx = step % len(dof_pos_list)
                dof_pos_rad = dof_pos_list[frame_idx].copy()

                # 读实物观测（与主流程一致）
                try:
                    real_obs, obs_without_command, position_Read, IMU_data = create_observation_from_real_robot(
                        servos, q_imu, step, history_length=5, cpg_reward=cpg_reward, previous_actions=None
                    )
                except Exception as e:
                    print(f"  ⚠️  Read sensors error at step {step}: {e}")
                    time.sleep(0.2)
                    step += 1
                    continue

                obs_str = '\t'.join([f'{x:.6f}' for x in real_obs])
                roll, pitch, yaw = IMU_data[0], IMU_data[1], IMU_data[2]
                f_obs.write(f"{step}\t{obs_str}\t{roll:.6f}\t{pitch:.6f}\t{yaw:.6f}\n")
                f_obs.flush()

                # 本步下发的关节指令：sim dof_pos -> 舵机角度 -> ticks
                servo_angles = sim_angles_rad_to_servo_angles_deg(dof_pos_rad)
                for i in range(18):
                    servo_angles[i] = np.clip(servo_angles[i], angle_limits['min'][i], angle_limits['max'][i])
                ticks = angles_to_ticks(servo_angles)

                actions_str = '\t'.join([f'{x:.6f}' for x in dof_pos_rad])
                real_angles_str = '\t'.join([f'{x:.6f}' for x in servo_angles])
                ticks_str = '\t'.join([f'{int(x)}' for x in ticks])
                f_actions.write(f"{step}\t{actions_str}\t{real_angles_str}\t{ticks_str}\n")
                f_actions.flush()

                try:
                    servos.write_all_positions(ticks)
                except Exception as e:
                    print(f"  ⚠️  Write error at step {step}: {e}")

                while (time.time() - start_time) < TARGET_DT:
                    pass
                if step % 100 == 0:
                    print(f"  Step {step} (frame {frame_idx}/{len(dof_pos_list)})")
                step += 1
    except KeyboardInterrupt:
        print("\n已停止回放")
    finally:
        try:
            imu_process.terminate()
            imu_process.join(timeout=2.0)
            if imu_process.is_alive():
                imu_process.kill()
        except Exception:
            pass
    print(f"Replay finished. Logs saved to {output_dir}/")


def test_rwm_real_robot_wm(model_path, enable_rate_limiter=True):
    """Test RWM on real robot - simplified for deployment"""
    
    print("\n" + "="*60)
    print("🚀 Starting Robot Control - Port Error Detection ENABLED")
    print("="*60 + "\n")

    # 在加载权重前设置线程与 OMP，利于 MatMul 与 ORT 一致
    apply_cpu_performance_settings(
        OPTIM_TORCH_NUM_THREADS,
        OPTIM_TORCH_NUM_INTEROP_THREADS,
        OPTIM_ENV_OMP_THREADS,
    )

    # Initialize RWM inference
    rwm_inference = RealRobotRWMInference(model_path, device='cpu', remove_dof_vel=remove_dof_vel)
    policy = rwm_inference.get_inference_policy()
    candidate_selector = WorldModelCandidateSelector(
        world_model=rwm_inference.world_model,
        action_dim=18,
        horizon=4,
        max_lift_candidates=6,
        device='cpu',
    ) if (ENABLE_WM_CANDIDATE_SELECTOR and rwm_inference.world_model is not None) else None
    contact_detector = ContactAnomalyDetector(
        world_model=rwm_inference.world_model,
        threshold=CONTACT_ANOMALY_THRESHOLD,
        ema_alpha=CONTACT_ANOMALY_EMA_ALPHA,
        trigger_count=CONTACT_ANOMALY_TRIGGER_COUNT,
        action_dim=18,
        device='cpu',
    ) if (ENABLE_CONTACT_ANOMALY_DETECTOR and rwm_inference.world_model is not None) else None
    detector_latent = rwm_inference.wm_latent
    detector_last_error = None
    detector_last_anomaly = False
    detector_last_steps = 0
    lift_controller = LegLiftController(lift_steps=4)
    risk_estimator = RiskLevelEstimator()
    bad_leg_tracker = BadLegTracker()
    print(
        "Right-front action offset: "
        f"knee[10]+={RIGHT_FRONT_KNEE_ACTION_OFFSET:.2f}, "
        f"ankle[11]+={RIGHT_FRONT_ANKLE_ACTION_OFFSET:.2f}"
    )
    print(f"Rate limiter: {'ENABLED' if enable_rate_limiter else 'DISABLED'}")

    # Initialize real robot components
    print("📍 Step 1: Initializing Servos...")
    try:
        servos = Servos()
        print("   ✓ Servos object created")
    except Exception as e:
        print(f"   ❌ Failed to create Servos: {e}")
        raise
    
    print("\n📍 Step 2: Reading voltage...")
    try:
        voltage = servos.read_voltage(1)
        print(f"   ✓ Voltage: {voltage}V")
        if voltage < ROBOT_CONFIG['control']['voltage_threshold']:
            print(f"   ⚠️  WARNING: Voltage too low! Required: {ROBOT_CONFIG['control']['voltage_threshold']}V")
    except Exception as e:
        print(f"   ❌ Failed to read voltage: {e}")
        raise

    print("\n📍 Step 3: Setting position control mode...")
    try:
        servos.set_position_control()
        print("   ✓ Position control mode set")
    except Exception as e:
        print(f"   ❌ Failed to set position control: {e}")
        raise
    
    position_all = range(18)
    print("\n📍 Step 4: Enabling torque...")
    print("   Press any key to enable servos! (or press ESC to quit!)")
    try:
        servos.enable_torque(position_all)
        print("   ✓ Torque enabled for all servos")
    except Exception as e:
        print(f"   ❌ Failed to enable torque: {e}")
        raise

    print("\n📍 Step 5: Initializing IMU process...")
    # Initialize IMU
    q_imu = Queue()
    imu_process = Process(target=read_imu, args=(q_imu,))
    imu_process.daemon = True  # Set as daemon so it dies with parent
    imu_process.start()
    print("   ✓ IMU process started")

    # Wait for IMU to initialize, but don't block forever
    imu_ready = False
    for i in range(20):  # Wait up to 2 seconds
        time.sleep(0.1)
        if not q_imu.empty():
            imu_ready = True
            print("   ✓ IMU data detected")
            break

    if not imu_ready:
        print("   ⚠️  Warning: No IMU data received, using simulation mode")
        # Start a mock IMU process for testing
        def mock_imu_process(q):
            import time
            import numpy as np
            import signal
            import sys
            
            def signal_handler(sig, frame):
                sys.exit(0)
            
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            
            try:
                while True:
                    # Generate mock IMU data
                    mock_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.8, 0.0])  # Basic gravity
                    q.put(mock_data)
                    time.sleep(0.05)  # 20Hz
            except KeyboardInterrupt:
                pass

        mock_process = Process(target=mock_imu_process, args=(q_imu,))
        mock_process.daemon = True  # Set as daemon
        mock_process.start()
        print("   ✓ Mock IMU process started")
        imu_process = mock_process  # Replace imu_process reference

    print("\n📍 Step 6: Moving to neutral position...")
    # Initialize robot to neutral position
    neutral_angles = ROBOT_CONFIG['neutral_angles']
    real_angles = radians_to_degrees(neutral_angles * np.pi / 180.0)
    ticks = angles_to_ticks(real_angles)
    try:
        servos.Robot_initialize(real_angles)
        print("   ✓ Robot moved to neutral position")
    except Exception as e:
        print(f"   ❌ Failed to initialize robot position: {e}")
        raise
    
    # 保存initialize角度到文件
    with open(os.path.join(VALIDATION_OUTPUT_DIR, 'initialize_angles_test_rwm_real_robot.txt'), 'a') as f:
        f.write(str(real_angles) + '\n')

    # Initialize trajectory history for policy input
    # Use obs_without_command dimension (matching playMBRL.py)
    history_length = 5
    obs_without_command_dim = (42 if remove_dof_vel else 60) + (6 if cpg_reward else 0)
    history_dim = obs_without_command_dim * history_length
    # 复用 history 张量，避免每步 torch.tensor 分配
    history_tensor_buf = torch.zeros(1, history_dim, dtype=torch.float32, device='cpu')
    action_scale_np = np.asarray(ACTION_SCALE_PER_DIM, dtype=np.float32)
    trajectory_history = deque(maxlen=history_length)

    # Initialize with zeros
    for _ in range(history_length):
        trajectory_history.append(np.zeros(obs_without_command_dim))

    # Get action limits to prevent servo angle clipping
    action_limits = get_action_limits()
    print("Action limits calculated to prevent servo angle clipping:")
    print(f"  Min action limits (degrees): {np.degrees(action_limits['min'])}")
    print(f"  Max action limits (degrees): {np.degrees(action_limits['max'])}")
    safety_filter = SafetyActionFilter(action_limits=action_limits)

    # Admittance controller to smooth actions
    admittance_filter = AdmittanceFilter(
        m=0.5,#0.1
        d=15.0,#10
        k=80.0,#80
        dt=TARGET_DT,
        num_joints=18
    )
    # Track whether we need to initialize the admittance filter
    admittance_needs_init = True

    # Initialize previous actions
    previous_actions = None

    # Initialize output files for validation
    output_dir = VALIDATION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    actions_file = os.path.join(output_dir, "actions_output.txt")
    observations_file = os.path.join(output_dir, "observations_output.txt")
    policy_input_file = os.path.join(output_dir, "policy_input.txt")
    safety_debug_file = os.path.join(output_dir, "safety_debug.csv")

    print(f"Actions will be logged to: {actions_file}")
    print(f"Observations will be logged to: {observations_file}")
    print(f"Policy inputs will be logged to: {policy_input_file}")
    print(f"Safety debug will be logged to: {safety_debug_file}")

    step = 0
    max_steps = MAX_STEPS  # Longer test run for better evaluation
    last_safety_event_key = None

    print("Starting real robot control loop (ACTIVE MODE - robot will move!)")
    print("⚠️  WARNING: Robot control is ENABLED. Ensure robot is in safe position!")
    print("Press Ctrl+C to stop if anything goes wrong.")
    input("Press Enter to continue or Ctrl+C to abort...")

    print("Waiting 3 seconds before starting control...")
    time.sleep(3)
    print("Starting control loop...")

    try:
        with open(actions_file, 'w') as f_actions, open(observations_file, 'w') as f_obs, open(policy_input_file, 'w') as f_pi, open(safety_debug_file, "w", newline="") as f_sd:
            # Set up keyboard interrupt handler for emergency stop
            import signal
            import sys

            def signal_handler(sig, frame):
                print('\n⚠️  EMERGENCY STOP: Keyboard interrupt detected!')
                print('Stopping robot control...')
                try:
                    # Add small delay to ensure any pending operations complete
                    time.sleep(0.2)
                    servos.disable_torque(position_all)
                    print('✓ Robot torque disabled')
                except Exception as e:
                    print(f'✗ Failed to disable torque: {e}')
                try:
                    # Terminate IMU process first to release port
                    imu_process.terminate()
                    imu_process.join(timeout=1.0)
                    print('✓ IMU process terminated')
                except:
                    pass
                try:
                    # Close servo port
                    servos.portHandler.closePort()
                    print('✓ Servo port closed')
                except:
                    pass
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            print('✓ Emergency stop handler installed (Ctrl+C to stop)')
            # Write headers
            f_actions.write("step\tactions_18\tangles_18\tticks_18\n")
            f_obs.write("step\tbase_ang_vel_3\tprojected_gravity_3\tcommands_3\tdof_pos_18\tdof_vel_18\tobs_action_18\troll\tpitch\tyaw\n")
            f_pi.write("step\tcommand\thistory\twm_feature\n")
            safety_writer = csv.DictWriter(
                f_sd,
                fieldnames=[
                    "step", "raw_action_min", "raw_action_max", "raw_action_mean",
                    "exec_action_min", "exec_action_max", "exec_action_mean",
                    "risk_level", "wm_error", "contact_error", "contact_anomaly", "contact_steps",
                    "detected_bad_leg", "bad_leg", "candidate_selected", "candidate_group", "candidate_score",
                    "forced_lift", "max_delta_before_filter",
                    "max_delta_after_filter", "ankle_delta_max",
                ],
            )
            safety_writer.writeheader()

            print("Log files opened successfully")

            while step < max_steps:
                loop_start_pc = time.perf_counter()
                start_time = time.time()
                
                # Checkpoint A: Before reading sensors
                if ENABLE_CONTROL_PRINT and step % 10 == 0:
                    print(f"\n📍 Step {step}: Starting control iteration")

                try:
                    # Checkpoint B: Reading sensors
                    if ENABLE_CONTROL_PRINT and step % 10 == 0:
                        print(f"   📡 Reading sensors from robot...")
                    
                    # Get observation from real robot sensors ONLY
                    sensor_start_pc = time.perf_counter()
                    real_obs, obs_without_command, position_Read, IMU_data = create_observation_from_real_robot(servos, q_imu, step, history_length, cpg_reward, previous_actions)
                    sensor_end_pc = time.perf_counter()
                    
                    if ENABLE_CONTROL_PRINT and step % 10 == 0:
                        print(f"   ✓ Sensors read successfully")
                    
                except Exception as e:
                    print(f"\n❌ ERROR at Step {step} - Reading sensors: {e}")
                    error_count = getattr(servos, 'error_count', 0) + 1
                    servos.error_count = error_count
                    
                    if error_count > 10:
                        print(f"⚠️  Too many consecutive errors ({error_count}), stopping...")
                        break
                    
                    print(f"⚠️  Skipping this step... (error count: {error_count})")
                    time.sleep(0.2)  # Longer delay to let port recover
                    step += 1
                    continue

                # Log complete observations (policy observation vector)
                obs_str = '\t'.join([f'{x:.6f}' for x in real_obs])
                # 获取IMU的角度数据 (roll, pitch, yaw)
                roll = IMU_data[0]   # angleX
                pitch = IMU_data[1]  # angleY
                yaw = IMU_data[2]    # angleZ
                f_obs.write(f"{step}\t{obs_str}\t{roll:.6f}\t{pitch:.6f}\t{yaw:.6f}\n")
                if (step % LOG_FLUSH_EVERY_N_STEPS) == 0:
                    f_obs.flush()  # 降低 flush 频率，减少文件 I/O 抖动

                # For policy input, we use the full observation (proprioceptive features are already included)
                proprioceptive_obs = real_obs
                # 保存observation到文件中
                #with open('./validation_outputs/proprioceptive_obs_test_rwm_real_robot.txt', 'a') as f:
                #    f.write(str(proprioceptive_obs) + '\n')
                with open(os.path.join(VALIDATION_OUTPUT_DIR, 'real_obs_test_rwm_real_robot.txt'), 'a') as f:
                    f.write(str(real_obs) + '\n')

                # Prepare observation dict for world model（复用 prop buffer）
                rwm_inference._prop_buffer[0].copy_(
                    torch.from_numpy(np.asarray(proprioceptive_obs, dtype=np.float32))
                )
                obs_dict = {
                    "prop": rwm_inference._prop_buffer,
                    "is_first": rwm_inference.wm_is_first,
                }

                # Update world model and get features
                wm_start_pc = time.perf_counter()
                wm_feature = rwm_inference.update_world_model(obs_dict, previous_actions)
                wm_end_pc = time.perf_counter()

                # Prepare history
                policy_start_pc = time.perf_counter()
                history_flat = np.concatenate(list(trajectory_history))
                history_tensor_buf[0].copy_(torch.from_numpy(history_flat.astype(np.float32)))
                history_tensor = history_tensor_buf

                # Log policy-related inputs for visualization: command, history (pre-encoder), wm_feature
                prop = obs_dict["prop"].detach().cpu().numpy()
                command_np = prop[0, 6:9]  # command = prop[6:9]
                hist_np = history_flat  # history before encoder
                wm_np = wm_feature.detach().cpu().numpy().flatten()
                command_str = '\t'.join([f'{x:.6f}' for x in command_np])
                hist_str = '\t'.join([f'{x:.6f}' for x in hist_np])
                wm_str = '\t'.join([f'{x:.6f}' for x in wm_np])
                f_pi.write(f"{step}\t{command_str}\t{hist_str}\t{wm_str}\n")
                if (step % LOG_FLUSH_EVERY_N_STEPS) == 0:
                    f_pi.flush()

                # Get action from policy
                #print("obs_dict['prop']: ", obs_dict["prop"])
                #print("history_tensor: ", history_tensor)
                #print("wm_feature: ", wm_feature)
                action_nominal = policy(obs_dict["prop"], history_tensor, wm_feature)
                actions = action_nominal
                detected_bad_leg = None
                if contact_detector is not None:
                    if detector_latent is None:
                        detector_latent = rwm_inference.wm_latent
                    if detector_latent is not None:
                        detector_last_anomaly, detector_last_error, detector_latent, detector_last_steps = contact_detector.detect(
                            prev_latent=detector_latent,
                            action=action_nominal,
                            obs_dict=obs_dict,
                        )
                        # When anomaly is flagged, locate the most likely stuck leg
                        # from per-leg joint prediction errors, then trigger a short
                        # single-leg lift override.
                        if detector_last_steps >= 1:
                            obs_real_for_leg = contact_detector.last_obs_real
                            obs_pred_for_leg = contact_detector.last_obs_pred
                            if (obs_real_for_leg is not None) and (obs_pred_for_leg is not None):
                                leg_errors = compute_leg_errors(obs_real_for_leg, obs_pred_for_leg)
                                detected_bad_leg = detect_stuck_leg(
                                    leg_errors,
                                    threshold=CONTACT_ANOMALY_THRESHOLD,
                                )
                                if detected_bad_leg is not None:
                                    leg_err_vec = leg_errors.detach().cpu().squeeze(-1).numpy()
                                    max_err = float(np.max(leg_err_vec))
                                    print(
                                        f"[LegAnomaly] step={step} detected_bad_leg={detected_bad_leg} "
                                        f"max_err={max_err:.4f} leg_errors={np.array2string(leg_err_vec, precision=4)}"
                                    )
                                if (detected_bad_leg is not None) and (candidate_selector is None) and (not lift_controller.active):
                                    lift_controller.trigger(detected_bad_leg)

                # Risk estimation (world-model anomaly now drives execution safety).
                contact_error_value = 0.0
                if detector_last_error is not None:
                    if torch.is_tensor(detector_last_error):
                        contact_error_value = float(torch.max(detector_last_error).detach().cpu().item())
                    else:
                        contact_error_value = float(detector_last_error)
                stable_bad_leg = bad_leg_tracker.update(
                    detected_bad_leg,
                    active=int(detector_last_steps) >= 1,
                )
                risk_state = risk_estimator.update(
                    wm_error=contact_error_value,
                    contact_anomaly=bool(detector_last_anomaly),
                    contact_steps=int(detector_last_steps),
                    bad_leg=stable_bad_leg if stable_bad_leg >= 0 else (lift_controller.leg_id if lift_controller.active else None),
                )

                if candidate_selector is not None and rwm_inference.wm_latent is not None:
                    prev_action_t = None if previous_actions is None else torch.from_numpy(
                        np.asarray(previous_actions, dtype=np.float32)
                    ).unsqueeze(0)
                    actions = candidate_selector.select(
                        prev_latent=rwm_inference.wm_latent,
                        is_first=rwm_inference.wm_is_first,
                        action_nominal=actions,
                        prev_action=prev_action_t,
                        risk_state=risk_state,
                    )
                # Apply temporary one-leg lift override only when the WM
                # candidate selector is unavailable; otherwise lift recovery is
                # represented inside the risk-conditioned candidate set.
                if candidate_selector is None:
                    actions = lift_controller.apply(actions)
                #print("actions: ", actions)

                # Keep policy action in simulation space for observation/history.
                action_raw_np = actions.detach().cpu().numpy().flatten().astype(np.float32)
                action_for_obs = np.clip(action_raw_np, -1.0, 1.0)
                action_for_exec = apply_right_front_action_offset(action_for_obs)

                # Convert policy-space action to execution rad, then apply hard safety filter.
                action_exec_desired = policy_action_to_exec_rad(
                    action_raw_clipped=action_for_exec,
                    action_scale_per_dim=action_scale_np,
                    use_asymmetric_ankle_mapping=USE_ASYMMETRIC_ANKLE_MAPPING,
                    asym_lift_range_rad=ASYM_ANKLE_LIFT_RANGE_RAD,
                    asym_sink_range_rad=ASYM_ANKLE_SINK_RANGE_RAD,
                )
                action_exec_desired = np.clip(action_exec_desired, action_limits['min'], action_limits['max'])
                if enable_rate_limiter:
                    action_limited, safety_dbg = safety_filter.filter(action_exec_desired, risk_state.level)
                else:
                    action_limited, safety_dbg = safety_filter.bypass(action_exec_desired)

                # Check if any actions were clipped
                action_clipped = np.any(action_exec_desired != action_limited)
                if action_clipped and ENABLE_ACTION_CLIP_PRINT:
                    clipped_count = np.sum(action_exec_desired != action_limited)
                    print(f"  ⚠️  {clipped_count} actions were clipped to stay within safe limits")
                    # Log which actions were clipped (optional)
                    for i in range(18):
                        if action_exec_desired[i] != action_limited[i]:
                            print(f"     Action {i}: {np.degrees(action_exec_desired[i]):.2f}° -> {np.degrees(action_limited[i]):.2f}°")

                    # Extra debug: show raw ankle actions (before clipping)
                    ankle_indices = [2, 5, 8, 11, 14, 17]
                    clipped_ankles = [i for i in ankle_indices if action_exec_desired[i] != action_limited[i]]
                    if clipped_ankles:
                        raw_deg = {i: float(np.degrees(action_exec_desired[i])) for i in ankle_indices}
                        lim_deg = {i: float(np.degrees(action_limited[i])) for i in ankle_indices}
                        lim_min_deg = {i: float(np.degrees(action_limits['min'][i])) for i in ankle_indices}
                        lim_max_deg = {i: float(np.degrees(action_limits['max'][i])) for i in ankle_indices}
                        print("  🦶 Ankle raw actions (deg) and limits:")
                        for i in ankle_indices:
                            flag = "CLIPPED" if i in clipped_ankles else "ok"
                            print(
                                f"     action[{i}]: raw={raw_deg[i]:7.2f}° -> limited={lim_deg[i]:7.2f}° "
                                f"(range [{lim_min_deg[i]:.2f}°, {lim_max_deg[i]:.2f}°]) [{flag}]"
                            )

                # Admittance filter for smooth joint commands (optional)
                if USE_ADMITTANCE:
                    if admittance_needs_init:
                        # Initialize using current real joint angles to avoid sudden jump
                        current_sim_angles = servo_angles_to_sim_angles(position_Read)
                        init_action = np.clip(current_sim_angles, action_limits['min'], action_limits['max'])
                        admittance_filter.reset(init_action)
                        safety_filter.reset(init_action)
                        admittance_needs_init = False
                    action_filtered = admittance_filter.update(action_limited)
                    action_filtered = np.clip(action_filtered, action_limits['min'], action_limits['max'])
                else:
                    action_filtered = action_limited.copy()

                real_angles = action_to_servo_angles(action_filtered)

                # Apply strict angle limits based on real robot measurements
                angle_limits = ROBOT_CONFIG['angle_limits']
                angle_limits_applied = False

                for i in range(18):
                    if real_angles[i] < angle_limits['min'][i]:
                        if ENABLE_JOINT_LIMIT_PRINT:
                            print(f"  ⚠️  Joint {i}: angle {real_angles[i]:.2f}° below limit {angle_limits['min'][i]:.2f}°, clipping")
                        real_angles[i] = angle_limits['min'][i]
                        angle_limits_applied = True
                    elif real_angles[i] > angle_limits['max'][i]:
                        if ENABLE_JOINT_LIMIT_PRINT:
                            print(f"  ⚠️  Joint {i}: angle {real_angles[i]:.2f}° above limit {angle_limits['max'][i]:.2f}°, clipping")
                        real_angles[i] = angle_limits['max'][i]
                        angle_limits_applied = True

                if angle_limits_applied:
                    if ENABLE_JOINT_LIMIT_PRINT:
                        print("  ✓ Angle limits applied to prevent unsafe movement")

                ticks = angles_to_ticks(real_angles)

                # Simplified write like CPGs: single write, no interpolation
                if ENABLE_CONTROL_PRINT and step % 10 == 0:
                    print(f"   📝 Writing positions to servos...")
                
                policy_end_pc = time.perf_counter()
                try:
                    # Single write like CPGs for reliability
                    servo_start_pc = time.perf_counter()
                    servos.write_all_positions(ticks)
                    servo_end_pc = time.perf_counter()
                    
                    if ENABLE_CONTROL_PRINT and step % 10 == 0:
                        print(f"   ✓ Positions written successfully")
                        
                except Exception as e:
                    print(f"\n❌ ERROR at Step {step} - Writing positions: {e}")
                    raise  # Re-raise to stop immediately
                # 将real_angles和ticks写入文件
                with open(os.path.join(VALIDATION_OUTPUT_DIR, 'real_angles_test_rwm_real_robot.txt'), 'a') as f:
                    f.write(str(real_angles) + '\n')
                with open(os.path.join(VALIDATION_OUTPUT_DIR, 'ticks_test_rwm_real_robot.txt'), 'a') as f:
                    f.write(str(ticks) + '\n')
                
                ## Log actions (BEFORE sending to servos) - using limited actions
                #actions_str = '\t'.join([f'{x:.6f}' for x in action_limited])
                # Log actions (BEFORE sending to servos) - using filtered actions
                actions_str = '\t'.join([f'{x:.6f}' for x in action_filtered])
                real_angles_str = '\t'.join([f'{x:.6f}' for x in real_angles])
                ticks_str = '\t'.join([f'{int(x)}' for x in ticks])
                f_actions.write(f"{step}\t{actions_str}\t{real_angles_str}\t{ticks_str}\n")
                if (step % LOG_FLUSH_EVERY_N_STEPS) == 0:
                    f_actions.flush()

                # Final target already sent as the last interpolation point

                # Print validation info (less verbose)
                if ENABLE_CONTROL_PRINT and (step % 10 == 0 or step < 3):
                    print(f"   📊 Step {step} completed:")
                    print(f"      Actions: Range=[{action_exec_desired.min():.3f}, {action_exec_desired.max():.3f}], Mean={action_exec_desired.mean():.3f}")
                    print(f"      Angles: Range=[{real_angles.min():.1f}, {real_angles.max():.1f}]°")
                    print(f"      ✓ All safety checks passed")

                # Check if angles exceed safe limits (detailed per joint)
                angle_limits = ROBOT_CONFIG['angle_limits']
                limit_violations = []

                for i in range(18):
                    if real_angles[i] < angle_limits['min'][i] or real_angles[i] > angle_limits['max'][i]:
                        limit_violations.append(f"Joint {i}: {real_angles[i]:.1f}° (limit: {angle_limits['min'][i]:.1f}°-{angle_limits['max'][i]:.1f}°)")

                if limit_violations:
                    if ENABLE_JOINT_LIMIT_PRINT:
                        print(f"  ❌ CRITICAL: {len(limit_violations)} joints exceed safe limits!")
                        for violation in limit_violations[:3]:  # Show first 3 violations
                            print(f"     {violation}")
                    if len(limit_violations) > 3:
                        if ENABLE_JOINT_LIMIT_PRINT:
                            print(f"     ... and {len(limit_violations) - 3} more violations")
                    if ENABLE_JOINT_LIMIT_PRINT:
                        print("  ⏹️  EMERGENCY STOP: Unsafe angles detected, skipping servo command")
                    # Skip servo command for this step - go to next iteration
                    step += 1
                    continue
                else:
                    if ENABLE_JOINT_LIMIT_PRINT:
                        print(f"  ✓ All angles within safe limits")

                # Additional safety check: ensure no extreme angle changes
                if step > 0 and hasattr(create_observation_from_real_robot, 'last_angles'):
                    angle_changes = np.abs(real_angles - create_observation_from_real_robot.last_angles)
                    max_change = angle_changes.max()
                    if max_change > 60.0:  # Max 30 degrees change per step
                        if ENABLE_JOINT_LIMIT_PRINT:
                            print(f"  ⚠️  WARNING: Large angle change detected: {max_change:.1f}°")
                            print("  ⏹️  EMERGENCY STOP: Excessive movement, skipping servo command")
                        step += 1
                        continue

                create_observation_from_real_robot.last_angles = real_angles.copy()

                # Update trajectory history with obs_without_command (matching training format)
                trajectory_history.append(obs_without_command.copy())

                # Control timing like CPGs: busy-wait until target time
                work_end_pc = time.perf_counter()
                elapsed_time = time.time() - start_time
                target_dt = TARGET_DT  # 20ms = 50Hz like CPGs
                while (time.time() - start_time) < target_dt:
                    pass  # Busy-wait like CPGs for precise timing
                
                final_elapsed = time.time() - start_time
                if final_elapsed > target_dt * 1.5:
                    if ENABLE_CONTROL_PRINT:
                        print(f"Warning: Control loop took {final_elapsed*1000:.1f}ms (target {target_dt*1000:.1f}ms)")

                # Update previous actions for next history input (sim-aligned: clip-only, scale前, 非对称映射前)
                previous_actions = action_for_obs.copy()

                candidate_debug = getattr(candidate_selector, "last_debug", {}) if candidate_selector is not None else {}
                candidate_selected = candidate_debug.get("selected", "nominal")
                candidate_group = candidate_debug.get("selected_group", "")
                safety_writer.writerow(
                    {
                        "step": step,
                        "raw_action_min": float(action_for_obs.min()),
                        "raw_action_max": float(action_for_obs.max()),
                        "raw_action_mean": float(action_for_obs.mean()),
                        "exec_action_min": float(action_filtered.min()),
                        "exec_action_max": float(action_filtered.max()),
                        "exec_action_mean": float(action_filtered.mean()),
                        "risk_level": int(risk_state.level),
                        "wm_error": float(risk_state.ema_error),
                        "contact_error": float(contact_error_value),
                        "contact_anomaly": int(bool(detector_last_anomaly)),
                        "contact_steps": int(detector_last_steps),
                        "detected_bad_leg": int(-1 if detected_bad_leg is None else detected_bad_leg),
                        "bad_leg": int(risk_state.bad_leg),
                        "candidate_selected": candidate_selected,
                        "candidate_group": candidate_group,
                        "candidate_score": candidate_debug.get("score", 0.0),
                        "forced_lift": int(bool(candidate_debug.get("forced_lift", False))),
                        "max_delta_before_filter": safety_dbg.get("max_delta_before_filter", 0.0),
                        "max_delta_after_filter": safety_dbg.get("max_delta_after_filter", 0.0),
                        "ankle_delta_max": safety_dbg.get("ankle_delta_max", 0.0),
                    }
                )
                if (step % LOG_FLUSH_EVERY_N_STEPS) == 0:
                    f_sd.flush()

                safety_event_key = (
                    int(risk_state.level),
                    int(bool(detector_last_anomaly)),
                    int(risk_state.bad_leg),
                    str(candidate_selected),
                )
                safety_event_active = (
                    risk_state.level > 0
                    or bool(detector_last_anomaly)
                    or str(candidate_group) in {"lift", "ankle_protected", "scaled", "blend"}
                )
                if safety_event_active and safety_event_key != last_safety_event_key:
                    last_safety_event_key = safety_event_key
                    print(
                        f"[SafetyEvent] step={step} risk={risk_state.level} "
                        f"contact_anomaly={int(bool(detector_last_anomaly))} "
                        f"contact_steps={int(detector_last_steps)} "
                        f"det_bad_leg={int(-1 if detected_bad_leg is None else detected_bad_leg)} "
                        f"bad_leg={int(risk_state.bad_leg)} "
                        f"cand={candidate_selected} group={candidate_group} "
                        f"forced_lift={int(bool(candidate_debug.get('forced_lift', False)))} "
                        f"score={candidate_debug.get('score', 0.0)} "
                        f"err={contact_error_value:.4f} ema={risk_state.ema_error:.4f} "
                        f"dq_raw={np.degrees(float(safety_dbg.get('max_delta_before_filter', 0.0))):.2f}deg "
                        f"dq_exec={np.degrees(float(safety_dbg.get('max_delta_after_filter', 0.0))):.2f}deg"
                    )

                if ENABLE_TIMING_REPORT and step > 0 and (step % TIMING_REPORT_EVERY_N_STEPS) == 0:
                    sensor_ms = (sensor_end_pc - sensor_start_pc) * 1000.0
                    wm_ms = (wm_end_pc - wm_start_pc) * 1000.0
                    policy_ms = (policy_end_pc - policy_start_pc) * 1000.0
                    servo_ms = (servo_end_pc - servo_start_pc) * 1000.0
                    work_ms = (work_end_pc - loop_start_pc) * 1000.0
                    total_ms = final_elapsed * 1000.0
                    if detector_last_error is None:
                        contact_error_log = "NA"
                    elif torch.is_tensor(detector_last_error):
                        contact_error_log = f"{float(torch.max(detector_last_error).detach().cpu().item()):.4f}"
                    else:
                        contact_error_log = f"{float(detector_last_error):.4f}"
                    print(
                        f"[Timing] step={step} sensor={sensor_ms:.1f}ms wm={wm_ms:.1f}ms "
                        f"policy={policy_ms:.1f}ms servo={servo_ms:.1f}ms work={work_ms:.1f}ms total={total_ms:.1f}ms "
                        f"contact_enabled={int(contact_detector is not None)} "
                        f"contact_anomaly={int(bool(detector_last_anomaly))} "
                        f"contact_error={contact_error_log} "
                        f"contact_steps={int(detector_last_steps)}"
                    )

                step += 1

                # 这里不再额外打印循环时间（交给 [Timing] 报告）

        finished_by_max_steps = step >= max_steps
        if finished_by_max_steps:
            print("\nReached MAX_STEPS. Robot keeps holding the last pose (torque still enabled).")
            input("Press Enter to end run and disable torque...")

        print("Control loop finished")
        print(f"Output files saved in: {output_dir}/")
        print("\nValidation Summary:")
        print(f"  Total steps: {step}")
        print(f"  Actions logged to: {actions_file}")
        print(f"  Observations logged to: {observations_file}")
        print("  Review the logged data to determine if action amplitudes need limiting.")

    except Exception as e:
        print(f"Error during validation: {e}")
        print("Attempting to save partial results...")

    finally:
        # Disable port error detector during cleanup to avoid cascade errors
        print("\n🔧 Starting cleanup process...")
        if 'port_detector' in globals():
            port_detector.disable()
        
        try:
            print("  Terminating IMU process...")
            imu_process.terminate()
            imu_process.join(timeout=2.0)
            if imu_process.is_alive():
                print("  ⚠️  IMU process still alive, forcing kill...")
                imu_process.kill()
            print("  ✓ IMU process terminated")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not terminate IMU process: {e}")
        
        # Small delay to ensure IMU port is released
        time.sleep(0.3)
        
        try:
            print("  Disabling servo torque...")
            servos.disable_torque(position_all)
            print("  ✓ Servos disabled")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not disable servos: {e}")
        
        try:
            print("  Closing servo port...")
            servos.portHandler.closePort()
            print("  ✓ Servo port closed")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not close servo port: {e}")
        
        print("🔧 Cleanup completed")


def verify_action_mapping():
    """Verify the action to servo angle mapping logic"""
    print("Verifying action to servo angle mapping...")

    # Test with zero actions (should give neutral poses)
    zero_actions = np.zeros(18)
    servo_angles = action_to_servo_angles(zero_actions)
    print(f"Zero actions -> Servo angles: {servo_angles}")

    # Expected neutral angles based on mapping:
    # Hip joints: 180° (action=0 -> 180°)
    # Left knee: 270° (action=0 -> 270° - (0 + 90) = 180°, wait this seems wrong)
    # Let me recalculate...

    # Left knee: action=-90° -> 270°, so action=0 -> 270° - (0 + 90°) = 180°
    # Right knee: action=0 -> 180°（同样以 180° 为默认零位）
    # Left ankle: action=90° -> 150°, so action=0 -> 150° + (0 - 90°) = 60°
    # Right ankle: action=-90° -> 150°, so action=0 -> 150° - (0 + 90°) = 60°

    expected_neutral = np.array([
        180, 180, 70,   # l1: hip=180, knee=180, ankle=60
        180, 180, 70,   # r1: hip=180, knee=180, ankle=60
        180, 180, 70,   # l2: hip=180, knee=180, ankle=60
        180, 180, 70,   # r2: hip=180, knee=180, ankle=60
        180, 180, 70,   # l3: hip=180, knee=180, ankle=60
        180, 180, 70    # r3: hip=180, knee=180, ankle=60
    ])

    print(f"Expected neutral angles: {expected_neutral}")
    print(f"Actual neutral angles: {servo_angles}")
    print(f"Difference: {servo_angles - expected_neutral}")

    # Test specific cases
    print("\nTesting specific mapping cases:")

    # Test hip joint: action=0 -> 180°, action=10° -> 170°, action=-10° -> 190°
    test_actions = np.zeros(18)
    test_actions[0] = math.radians(10)   # +10° action on l1 hip
    test_actions[9] = math.radians(-10)  # -10° action on r1 hip
    servo_angles = action_to_servo_angles(test_actions)
    print(f"Hip test - l1 hip (idx 0): {servo_angles[0]:.1f}° (expected: 170.0°)")
    print(f"Hip test - r1 hip (idx 3): {servo_angles[3]:.1f}° (expected: 190.0°)")

    # Test knee joints
    test_actions = np.zeros(18)
    test_actions[1] = math.radians(-90)  # -90° action on l1 knee -> should give 270°
    test_actions[10] = math.radians(-90)  # -90° action on r1 knee -> should give 270°
    servo_angles = action_to_servo_angles(test_actions)
    print(f"Knee test - l1 knee (idx 1): {servo_angles[1]:.1f}° (expected: 270.0°)")
    print(f"Knee test - r1 knee (idx 4): {servo_angles[4]:.1f}° (expected: 270.0°)")

    # Test ankle joints
    test_actions = np.zeros(18)
    test_actions[2] = math.radians(90)   # +90° action on l1 ankle -> should give 150°
    test_actions[11] = math.radians(-90) # -90° action on r1 ankle -> should give 150°
    servo_angles = action_to_servo_angles(test_actions)
    print(f"Ankle test - l1 ankle (idx 2): {servo_angles[2]:.1f}° (expected: 150.0°)")
    print(f"Ankle test - r1 ankle (idx 5): {servo_angles[5]:.1f}° (expected: 150.0°)")

    print("Verification completed.")


def verify_joint_mapping():
    """Verify that joint angle and velocity mapping works correctly"""
    print("\nVerifying joint angle and velocity mapping...")

    # Test with neutral servo angles (around 180 degrees)
    neutral_servo_angles = np.array([
        180, 180, 60,   # l1: hip=180, knee=180, ankle=60
        180, 180, 60,   # r1: hip=180, knee=180, ankle=60
        180, 180, 60,   # l2: hip=180, knee=180, ankle=60
        180, 180, 60,   # r2: hip=180, knee=180, ankle=60
        180, 180, 60,   # l3: hip=180, knee=180, ankle=60
        180, 180, 60    # r3: hip=180, knee=180, ankle=60
    ])

    sim_angles = servo_angles_to_sim_angles(neutral_servo_angles)
    print("Neutral servo angles -> Simulation joint angles:")
    print(f"  Servo angles: {neutral_servo_angles}")
    print(f"  Sim angles (deg): {np.degrees(sim_angles)}")
    print("  Expected: mostly zeros (neutral pose in simulation)")

    # Test action->servo->sim round trip
    test_actions = np.array([0.1, -0.2, 0.3, 0.05, -0.15, 0.25, -0.1, 0.2, -0.3,
                            0.05, 0.15, -0.2, -0.08, 0.18, -0.12, 0.02, -0.25, 0.35])
    test_actions_rad = test_actions

    # Convert actions to servo angles
    servo_angles = action_to_servo_angles(test_actions_rad)

    # Convert servo angles back to sim angles
    sim_angles_back = servo_angles_to_sim_angles(servo_angles)

    print("\nRound trip test (actions -> servo -> sim):")
    print(f"  Original actions (deg): {np.degrees(test_actions)}")
    print(f"  Servo angles: {servo_angles}")
    print(f"  Converted back (deg): {np.degrees(sim_angles_back)}")
    print(f"  Difference: {np.degrees(test_actions - sim_angles_back)}")

    print("Joint mapping verification completed.")


def verify_action_limits():
    """Verify that action limits prevent servo angle clipping"""
    print("\nVerifying action limits...")

    # Get action limits
    action_limits = get_action_limits()

    print("Action limits (degrees):")
    for i in range(18):
        min_deg = np.degrees(action_limits['min'][i])
        max_deg = np.degrees(action_limits['max'][i])
        print(f"  Action {i}: [{min_deg:.2f}, {max_deg:.2f}]°")

    # Test boundary actions to ensure they don't cause servo clipping
    print("\nTesting boundary actions...")

    # Test minimum actions
    min_actions = action_limits['min']
    min_servo_angles = action_to_servo_angles(min_actions)  # Already in radians, function converts internally
    print("Minimum actions -> Servo angles:")
    for i in range(18):
        servo_min = ROBOT_CONFIG['angle_limits']['min'][i]
        servo_max = ROBOT_CONFIG['angle_limits']['max'][i]
        servo_angle = min_servo_angles[i]
        status = "✓" if servo_min <= servo_angle <= servo_max else "✗"
        print(f"  Joint {i}: {servo_angle:.2f}° (limit: {servo_min:.2f}°-{servo_max:.2f}°) {status}")

    # Test maximum actions
    max_actions = action_limits['max']
    max_servo_angles = action_to_servo_angles(max_actions)  # Already in radians, function converts internally
    print("Maximum actions -> Servo angles:")
    for i in range(18):
        servo_min = ROBOT_CONFIG['angle_limits']['min'][i]
        servo_max = ROBOT_CONFIG['angle_limits']['max'][i]
        servo_angle = max_servo_angles[i]
        status = "✓" if servo_min <= servo_angle <= servo_max else "✗"
        print(f"  Joint {i}: {servo_angle:.2f}° (limit: {servo_min:.2f}°-{servo_max:.2f}°) {status}")

    # Check if any servo angles are outside limits
    min_violations = np.sum((min_servo_angles < ROBOT_CONFIG['angle_limits']['min']) |
                           (min_servo_angles > ROBOT_CONFIG['angle_limits']['max']))
    max_violations = np.sum((max_servo_angles < ROBOT_CONFIG['angle_limits']['min']) |
                           (max_servo_angles > ROBOT_CONFIG['angle_limits']['max']))

    if min_violations == 0 and max_violations == 0:
        print("✓ All boundary actions stay within servo limits!")
    else:
        print(f"✗ {min_violations + max_violations} boundary violations detected!")

    print("Action limits verification completed.")


def verify_safety():
    """Synthetic safety checks for deployment-side action manager."""
    print("\nVerifying deployment safety filter...")
    action_limits = get_action_limits()
    sf = SafetyActionFilter(action_limits=action_limits)
    risk = RiskLevelEstimator()
    current = np.zeros(18, dtype=np.float32)
    sf.reset(current)

    seq = [
        np.zeros(18, dtype=np.float32),
        np.ones(18, dtype=np.float32),
        -np.ones(18, dtype=np.float32),
        np.concatenate([np.full(12, 0.8, dtype=np.float32), np.full(6, -1.0, dtype=np.float32)]),
    ]
    ok_rate = True
    ok_prev = True
    ok_ankle_down = True
    ok_risk_tighten = True
    prev_obs_actions = []
    low_risk_after = None
    high_risk_after = None
    for i, raw in enumerate(seq):
        raw_clipped = np.clip(raw, -1.0, 1.0)
        prev_obs_actions.append(raw_clipped.copy())
        exec_des = policy_action_to_exec_rad(
            raw_clipped,
            action_scale_per_dim=np.asarray(ACTION_SCALE_PER_DIM, dtype=np.float32),
            use_asymmetric_ankle_mapping=USE_ASYMMETRIC_ANKLE_MAPPING,
            asym_lift_range_rad=ASYM_ANKLE_LIFT_RANGE_RAD,
            asym_sink_range_rad=ASYM_ANKLE_SINK_RANGE_RAD,
        )
        if i < 2:
            rs = risk.update(wm_error=0.02, contact_anomaly=False, contact_steps=0)
        else:
            rs = risk.update(wm_error=0.30, contact_anomaly=True, contact_steps=3)
        filtered, dbg = sf.filter(exec_des, rs.level)
        if rs.level == 0:
            low_risk_after = dbg["ankle_delta_max"]
        if rs.level == 2:
            high_risk_after = dbg["ankle_delta_max"]
        if dbg["max_delta_after_filter"] > np.radians(3.1):
            ok_rate = False
        if np.any(np.abs(raw_clipped) > 1.0001):
            ok_prev = False
        if rs.level >= 2 and dbg["ankle_delta_max"] > np.radians(1.1):
            ok_ankle_down = False

    if (low_risk_after is not None) and (high_risk_after is not None):
        ok_risk_tighten = bool(high_risk_after < low_risk_after)

    print(f"  raw action kept in [-1,1]: {'PASS' if ok_prev else 'FAIL'}")
    print(f"  per-step delta limited: {'PASS' if ok_rate else 'FAIL'}")
    print(f"  high-risk ankle tightening: {'PASS' if ok_ankle_down else 'FAIL'}")
    print(f"  risk level tightens action: {'PASS' if ok_risk_tighten else 'FAIL'}")
    all_ok = ok_prev and ok_rate and ok_ankle_down and ok_risk_tighten
    print(f"verify-safety result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == '__main__':
    import sys
    disable_rate_limiter = '--disable-rate-limiter' in sys.argv
    if disable_rate_limiter:
        sys.argv = [arg for arg in sys.argv if arg != '--disable-rate-limiter']
    remote_ip = None
    remote_port = 9876
    if '--remote-wm-server-ip' in sys.argv:
        idx = sys.argv.index('--remote-wm-server-ip')
        if idx + 1 < len(sys.argv):
            remote_ip = sys.argv[idx + 1]
        else:
            print("Usage: --remote-wm-server-ip <PC_IP> [--remote-wm-server-port <PORT>]")
            sys.exit(1)
    if '--remote-wm-server-port' in sys.argv:
        idx = sys.argv.index('--remote-wm-server-port')
        if idx + 1 < len(sys.argv):
            remote_port = int(sys.argv[idx + 1])
        else:
            print("Usage: --remote-wm-server-port <PORT>")
            sys.exit(1)

    # Raspberry Pi remote-client mode: keep local one-process mode untouched when absent.
    if remote_ip is not None:
        from rpi_robot_client import run_client
        print(f"Running remote WM client mode -> {remote_ip}:{remote_port}")
        run_client(
            remote_ip,
            remote_port,
            timeout_s=0.05,
            log_every=50,
            enable_rate_limiter=not disable_rate_limiter,
        )
        sys.exit(0)

    # PC server mode for convenience. This reuses the dedicated UDP server implementation.
    if '--pc-wm-server' in sys.argv:
        import argparse
        from pc_wm_server import run_server

        parser = argparse.ArgumentParser(description="PC world-model inference server via test_rwm_real_robot_wm.py")
        parser.add_argument("--pc-wm-server", action="store_true")
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=9876)
        parser.add_argument("--model-path", default=None)
        parser.add_argument("--remove-dof-vel", action="store_true")
        parser.add_argument("--log-every", type=int, default=50)
        parser.add_argument("--use-stability-filter", dest="use_stability_filter", action="store_true", default=True)
        parser.add_argument("--disable-stability-filter", dest="use_stability_filter", action="store_false")
        parser.add_argument("--filter-debug-path", default=None)
        args = parser.parse_args()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = args.model_path or os.path.join(current_dir, 'world_model', 'newact_model_2000.pt')
        run_server(
            args.host,
            args.port,
            model_path,
            remove_dof_vel=args.remove_dof_vel,
            log_every=args.log_every,
            use_stability_filter=args.use_stability_filter,
            filter_debug_path=args.filter_debug_path,
        )
        sys.exit(0)

    if len(sys.argv) > 1:
        if sys.argv[1] == '--verify':
            verify_action_mapping()
        elif sys.argv[1] == '--verify-limits':
            verify_action_limits()
        elif sys.argv[1] == '--verify-mapping':
            verify_joint_mapping()
        elif sys.argv[1] == '--verify-safety':
            verify_safety()
        elif sys.argv[1] == '--imu-test':
            run_imu_test_only()
        elif sys.argv[1] == '--replay-sim-dof':
            sim_path = sys.argv[2] if len(sys.argv) > 2 else None
            run_replay_sim_dof(sim_path)
        else:
            print("Usage: python test_rwm_real_robot_wm.py [--disable-rate-limiter] [--verify|--verify-limits|--verify-mapping|--verify-safety|--imu-test|--replay-sim-dof [sim_data_path]|--remote-wm-server-ip <PC_IP> [--remote-wm-server-port <PORT>]|--pc-wm-server [--model-path PATH] [--host HOST] [--port PORT] [--remove-dof-vel] [--use-stability-filter]]")
    else:
        # Path to your trained RWM model checkpoint
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'world_model', 'newact_model_2000.pt')
        #model_path = os.path.join(current_dir, 'world_model', 'noac_model_3500.pt')
        #model_path = os.path.join(current_dir, 'world_model', 'stage1_wm_only.pt')
        #model_path = os.path.join(current_dir, 'world_model', 'stage2_imag_policy_epochs2000.pt')
        #model_path = os.path.join(current_dir, 'world_model', 'asyresume_model_6000.pt')
        # model_path = os.path.join(current_dir, 'world_model', 'cpgtrack_model_5000.pt')
        # model_path = os.path.join(current_dir, 'world_model', 'model-remove-vel-9_20000.pt')
        # model_path = os.path.join(current_dir, 'world_model', 'model-cpg_10500.pt')
        # model_path = os.path.join(current_dir, 'world_model', 'model_9500.pt')

        print(f"Loading model from: {model_path}")
        test_rwm_real_robot_wm(model_path, enable_rate_limiter=not disable_rate_limiter)
