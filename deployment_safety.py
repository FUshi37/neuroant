import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


ANKLE_INDEX_LIST = [2, 5, 8, 11, 14, 17]
KNEE_INDEX_LIST = [1, 4, 7, 10, 13, 16]
# Policy/action leg order is [l1, l2, l3, r1, r2, r3], not servo order.
# These directions convert the desired physical-positive direction into the
# normalized action sign used by action_to_servo_angles().
LIFT_DIR_LIST = [1, 1, 1, -1, -1, -1]
KNEE_LIFT_DIR_LIST = [1, 1, 1, -1, -1, -1]
HIP_FORWARD_DIR_LIST = [1, 1, 1, -1, -1, -1]

ANKLE_INDICES = np.array(ANKLE_INDEX_LIST, dtype=np.int64)
LIFT_DIR = np.array(LIFT_DIR_LIST, dtype=np.float32)
KNEE_INDICES = np.array(KNEE_INDEX_LIST, dtype=np.int64)
KNEE_LIFT_DIR = np.array(KNEE_LIFT_DIR_LIST, dtype=np.float32)
HIP_FORWARD_DIR = np.array(HIP_FORWARD_DIR_LIST, dtype=np.float32)


# Deployment-side execution scale (rad per normalized action unit).
# Keep defaults aligned with test_rwm_real_robot.py legacy runtime behavior.
DEFAULT_HIP_KNEE_SCALE_RAD = 0.50
DEFAULT_ANKLE_BASE_SCALE_RAD = 0.50

# Safety limits (rad per control step). Recommended ranges in comments.
SAFETY_HIP_KNEE_MAX_DELTA_RAD = math.radians(2.5)  # 2~3 deg/step
SAFETY_ANKLE_MAX_DELTA_RAD = math.radians(1.5)  # 1~2 deg/step
SAFETY_ANKLE_DOWN_MAX_DELTA_RAD = math.radians(1.0)  # sink/downward direction
SAFETY_HIGH_RISK_ANKLE_MAX_DELTA_RAD = math.radians(0.8)  # 0.5~1 deg/step
SAFETY_HIGH_RISK_ANKLE_DOWN_MAX_DELTA_RAD = math.radians(0.5)  # 0.5~1 deg/step
SAFETY_ACCEL_MAX_DELTA_CHANGE_RAD = math.radians(1.5)

RISK_GAIN_LEVEL1 = 0.85
RISK_GAIN_LEVEL2 = 0.65
RISK_RATE_GAIN_LEVEL1 = 0.85
RISK_RATE_GAIN_LEVEL2 = 0.60

RISK_EMA_ALPHA = 0.90
RISK_BASELINE_WARMUP_STEPS = 80
RISK_K_SIGMA = 2.5
RISK_FIXED_LEVEL1_THRESHOLD = 0.12
RISK_FIXED_LEVEL2_THRESHOLD = 0.22
BAD_LEG_TRACK_DECAY = 0.72
BAD_LEG_TRACK_HOLD_STEPS = 8

# Lift recovery candidates are expressed in normalized policy-action units
# before the deployment action scale and servo angle limits are applied.
LIFT_HIP_FORWARD_TARGET_ACTION = 0.75
LIFT_HIP_FORWARD_BLEND = 0.90
LIFT_KNEE_TARGET_ACTION = 1.0
LIFT_KNEE_TARGET_BLEND = 1.0
LIFT_ANKLE_TARGET_ACTION = 0.0
LIFT_ANKLE_TARGET_BLEND = 0.90
LIFT_RECOVERY_HOLD_STEPS = 16
LIFT_RECOVERY_TRIGGER_RISK_LEVEL = 2
LIFT_RECOVERY_TRIGGER_CONTACT_STEPS = 3

# Simple right-front leg action offset.
# Action order: [l1, l2, l3, r1, r2, r3], three joints per leg.
# r1 knee = action[10], r1 ankle = action[11].
# For the right-front leg, larger knee action and smaller ankle action lift the foot.
RIGHT_FRONT_ACTION_OFFSET_ENABLED = True
RIGHT_FRONT_KNEE_ACTION_OFFSET = 0.0
RIGHT_FRONT_ANKLE_ACTION_OFFSET = 0.0


def default_action_scale_per_dim() -> np.ndarray:
    out = np.full(18, DEFAULT_HIP_KNEE_SCALE_RAD, dtype=np.float32)
    out[ANKLE_INDICES] = DEFAULT_ANKLE_BASE_SCALE_RAD
    return out


def apply_right_front_action_offset(action: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    if RIGHT_FRONT_ACTION_OFFSET_ENABLED and out.size >= 12:
        out[10] += RIGHT_FRONT_KNEE_ACTION_OFFSET
        out[11] += RIGHT_FRONT_ANKLE_ACTION_OFFSET
    return np.clip(out, -1.0, 1.0)


def _ankle_down_mask(delta: np.ndarray) -> np.ndarray:
    ankle_delta = delta[ANKLE_INDICES]
    aligned = ankle_delta * LIFT_DIR
    return aligned < 0.0


def apply_asymmetric_ankle_mapping_rad(
    actions_rad: np.ndarray,
    lift_range_rad: float,
    sink_range_rad: float,
) -> np.ndarray:
    """Match test_rwm_real_robot.py asymmetric ankle mapping in radians."""
    out = np.asarray(actions_rad, dtype=np.float32).copy()
    aligned = out[ANKLE_INDICES] * LIFT_DIR
    denom = lift_range_rad if lift_range_rad > 1e-6 else 1e-6
    aligned_n = np.clip(aligned / denom, -1.0, 1.0)
    mapped = np.where(aligned_n >= 0.0, aligned_n * lift_range_rad, aligned_n * sink_range_rad)
    out[ANKLE_INDICES] = mapped * LIFT_DIR
    return out


def policy_action_to_exec_rad(
    action_raw_clipped: np.ndarray,
    action_scale_per_dim: np.ndarray,
    use_asymmetric_ankle_mapping: bool,
    asym_lift_range_rad: float,
    asym_sink_range_rad: float,
) -> np.ndarray:
    """
    Keep execution mapping identical to test_rwm_real_robot.py:
    1) clip raw to [-1, 1]
    2) apply per-dim action scale for all joints
    3) apply optional asymmetric ankle mapping in rad space
    """
    raw = np.asarray(action_raw_clipped, dtype=np.float32).copy()
    raw = np.clip(raw, -1.0, 1.0)
    out = raw * np.asarray(action_scale_per_dim, dtype=np.float32)
    if use_asymmetric_ankle_mapping:
        out = apply_asymmetric_ankle_mapping_rad(out, asym_lift_range_rad, asym_sink_range_rad)
    return out


@dataclass
class RiskState:
    level: int
    ema_error: float
    baseline_mean: float
    baseline_std: float
    contact_anomaly: bool
    contact_steps: int
    bad_leg: int


class RiskLevelEstimator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ema_error = 0.0
        self._m2 = 0.0
        self._mean = 0.0
        self._count = 0

    def update(self, wm_error: float, contact_anomaly: bool, contact_steps: int, bad_leg: Optional[int] = None) -> RiskState:
        wm_error = float(max(0.0, wm_error))
        self.ema_error = RISK_EMA_ALPHA * self.ema_error + (1.0 - RISK_EMA_ALPHA) * wm_error

        if self._count < RISK_BASELINE_WARMUP_STEPS:
            self._count += 1
            delta = self.ema_error - self._mean
            self._mean += delta / self._count
            delta2 = self.ema_error - self._mean
            self._m2 += delta * delta2

        baseline_std = math.sqrt(self._m2 / max(1, self._count - 1)) if self._count > 1 else 0.0
        th1 = self._mean + RISK_K_SIGMA * baseline_std if self._count >= 20 else RISK_FIXED_LEVEL1_THRESHOLD
        th2 = self._mean + (RISK_K_SIGMA + 1.5) * baseline_std if self._count >= 20 else RISK_FIXED_LEVEL2_THRESHOLD

        level = 0
        if self.ema_error > th1 or contact_steps >= 1:
            level = 1
        if self.ema_error > th2 or contact_anomaly or contact_steps >= 3:
            level = 2
        return RiskState(
            level=int(level),
            ema_error=float(self.ema_error),
            baseline_mean=float(self._mean),
            baseline_std=float(baseline_std),
            contact_anomaly=bool(contact_anomaly),
            contact_steps=int(contact_steps),
            bad_leg=int(-1 if bad_leg is None else bad_leg),
        )


class BadLegTracker:
    def __init__(self, decay: float = BAD_LEG_TRACK_DECAY, hold_steps: int = BAD_LEG_TRACK_HOLD_STEPS):
        self.decay = float(decay)
        self.hold_steps = int(max(1, hold_steps))
        self.scores = np.zeros(6, dtype=np.float32)
        self.current = -1
        self.hold = 0

    def reset(self):
        self.scores[:] = 0.0
        self.current = -1
        self.hold = 0

    def update(self, detected_leg: Optional[int], active: bool) -> int:
        self.scores *= self.decay
        leg = -1 if detected_leg is None else int(detected_leg)
        if active and 0 <= leg < 6:
            self.scores[leg] += 1.0
            best = int(np.argmax(self.scores))
            self.current = best
            self.hold = self.hold_steps
        elif self.hold > 0:
            self.hold -= 1
        else:
            self.current = -1
        return int(self.current)


class SafetyActionFilter:
    def __init__(self, action_limits: Dict[str, np.ndarray]):
        self.action_limits = action_limits
        self.prev_exec = None
        self.prev_delta = np.zeros(18, dtype=np.float32)

    def reset(self, current_sim_angles: np.ndarray):
        cur = np.asarray(current_sim_angles, dtype=np.float32)
        self.prev_exec = np.clip(cur, self.action_limits["min"], self.action_limits["max"])
        self.prev_delta = np.zeros_like(self.prev_exec, dtype=np.float32)

    def _make_delta_limits(self, risk_level: int) -> Tuple[np.ndarray, np.ndarray]:
        max_delta = np.full(18, SAFETY_HIP_KNEE_MAX_DELTA_RAD, dtype=np.float32)
        down_limit = max_delta.copy()
        max_delta[ANKLE_INDICES] = SAFETY_ANKLE_MAX_DELTA_RAD
        down_limit[ANKLE_INDICES] = SAFETY_ANKLE_DOWN_MAX_DELTA_RAD
        if risk_level >= 1:
            max_delta *= RISK_RATE_GAIN_LEVEL1
            down_limit *= RISK_RATE_GAIN_LEVEL1
        if risk_level >= 2:
            max_delta *= RISK_RATE_GAIN_LEVEL2
            down_limit *= RISK_RATE_GAIN_LEVEL2
            max_delta[ANKLE_INDICES] = np.minimum(max_delta[ANKLE_INDICES], SAFETY_HIGH_RISK_ANKLE_MAX_DELTA_RAD)
            down_limit[ANKLE_INDICES] = np.minimum(down_limit[ANKLE_INDICES], SAFETY_HIGH_RISK_ANKLE_DOWN_MAX_DELTA_RAD)
        return max_delta, down_limit

    def filter(self, action_exec_desired: np.ndarray, risk_level: int) -> Tuple[np.ndarray, Dict[str, float]]:
        desired = np.asarray(action_exec_desired, dtype=np.float32).copy()
        if self.prev_exec is None:
            self.reset(desired)
        assert self.prev_exec is not None

        amp_gain = 1.0
        if risk_level == 1:
            amp_gain = RISK_GAIN_LEVEL1
        elif risk_level >= 2:
            amp_gain = RISK_GAIN_LEVEL2
        desired *= amp_gain
        desired = np.clip(desired, self.action_limits["min"], self.action_limits["max"])

        raw_delta = desired - self.prev_exec
        max_delta, down_limit = self._make_delta_limits(risk_level)
        limited_delta = np.clip(raw_delta, -max_delta, max_delta)

        down_mask = _ankle_down_mask(limited_delta)
        for idx, is_down in zip(ANKLE_INDICES, down_mask):
            if is_down:
                lim = down_limit[idx]
                limited_delta[idx] = np.clip(limited_delta[idx], -lim, lim)

        delta_change = limited_delta - self.prev_delta
        delta_change = np.clip(delta_change, -SAFETY_ACCEL_MAX_DELTA_CHANGE_RAD, SAFETY_ACCEL_MAX_DELTA_CHANGE_RAD)
        final_delta = self.prev_delta + delta_change

        out = self.prev_exec + final_delta
        out = np.clip(out, self.action_limits["min"], self.action_limits["max"])
        self.prev_delta = out - self.prev_exec
        self.prev_exec = out.copy()

        dbg = {
            "max_delta_before_filter": float(np.max(np.abs(raw_delta))),
            "max_delta_after_filter": float(np.max(np.abs(self.prev_delta))),
            "ankle_delta_max": float(np.max(np.abs(self.prev_delta[ANKLE_INDICES]))),
        }
        return out, dbg

    def bypass(self, action_exec_desired: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        desired = np.asarray(action_exec_desired, dtype=np.float32).copy()
        desired = np.clip(desired, self.action_limits["min"], self.action_limits["max"])
        if self.prev_exec is None:
            self.reset(desired)
        assert self.prev_exec is not None

        raw_delta = desired - self.prev_exec
        self.prev_delta = raw_delta.copy()
        self.prev_exec = desired.copy()

        max_delta = float(np.max(np.abs(raw_delta)))
        return desired, {
            "max_delta_before_filter": max_delta,
            "max_delta_after_filter": max_delta,
            "ankle_delta_max": float(np.max(np.abs(raw_delta[ANKLE_INDICES]))),
        }


class WorldModelCandidateSelector:
    def __init__(
        self,
        world_model,
        action_dim: int = 18,
        horizon: int = 4,
        max_lift_candidates: int = 2,
        device: str = "cpu",
    ):
        self.world_model = world_model
        self.action_dim = int(action_dim)
        self.horizon = int(max(1, min(5, horizon)))
        self.max_lift_candidates = int(max(0, min(6, max_lift_candidates)))
        self.device = torch.device(device)
        self.lift_hip_forward_target = float(LIFT_HIP_FORWARD_TARGET_ACTION)
        self.lift_hip_forward_blend = float(LIFT_HIP_FORWARD_BLEND)
        self.lift_knee_target = float(LIFT_KNEE_TARGET_ACTION)
        self.lift_knee_blend = float(LIFT_KNEE_TARGET_BLEND)
        self.lift_ankle_target = float(LIFT_ANKLE_TARGET_ACTION)
        self.lift_ankle_blend = float(LIFT_ANKLE_TARGET_BLEND)
        self.recovery_hold_steps = int(max(0, LIFT_RECOVERY_HOLD_STEPS))
        self.recovery_leg = -1
        self.recovery_hold = 0
        self.recovery_last_triggered = False
        self.recovery_last_accepted = False
        self.last_debug = {}
        self._select_count = 0

    @staticmethod
    def _risk_level(risk_state: Optional[RiskState]) -> int:
        if risk_state is None:
            return 0
        return int(max(0, min(2, int(getattr(risk_state, "level", 0)))))

    @staticmethod
    def _bad_leg(risk_state: Optional[RiskState]) -> int:
        if risk_state is None:
            return -1
        leg = int(getattr(risk_state, "bad_leg", -1))
        return leg if 0 <= leg < 6 else -1

    @staticmethod
    def _contact_steps(risk_state: Optional[RiskState]) -> int:
        if risk_state is None:
            return 0
        return int(max(0, int(getattr(risk_state, "contact_steps", 0))))

    @staticmethod
    def _candidate_group(name: str) -> str:
        if name.startswith("lift_leg_"):
            return "lift"
        if name.startswith("scale_"):
            return "scaled"
        if name == "blend_prev":
            return "blend"
        if name == "ankle_protected":
            return "ankle_protected"
        return "nominal"

    @staticmethod
    def _risk_scales(risk_level: int) -> Tuple[float, float, float, float]:
        if risk_level >= 2:
            return 0.50, 0.30, 0.85, 0.28
        if risk_level == 1:
            return 0.65, 0.45, 0.70, 0.40
        return 0.75, 0.55, 0.60, 0.55

    @staticmethod
    def _blend01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _fixed_raw_offset(leg_id: int, joint_offset: int) -> float:
        if RIGHT_FRONT_ACTION_OFFSET_ENABLED and int(leg_id) == 3:
            if int(joint_offset) == 1:
                return float(RIGHT_FRONT_KNEE_ACTION_OFFSET)
            if int(joint_offset) == 2:
                return float(RIGHT_FRONT_ANKLE_ACTION_OFFSET)
        return 0.0

    def _lift_pose_debug(self, leg_id: int) -> Dict[str, float]:
        leg_id = int(max(0, min(5, leg_id)))
        hip_target = float(np.clip(self.lift_hip_forward_target * HIP_FORWARD_DIR[leg_id], -1.0, 1.0))
        knee_final_target = float(np.clip(self.lift_knee_target * KNEE_LIFT_DIR[leg_id], -1.0, 1.0))
        knee_raw_target = float(np.clip(knee_final_target - self._fixed_raw_offset(leg_id, 1), -1.0, 1.0))
        ankle_raw_target = float(np.clip(self.lift_ankle_target - self._fixed_raw_offset(leg_id, 2), -1.0, 1.0))
        return {
            "lift_hip_forward_target": hip_target,
            "lift_hip_forward_blend": self._blend01(self.lift_hip_forward_blend),
            "lift_knee_final_target": knee_final_target,
            "lift_knee_raw_target": knee_raw_target,
            "lift_knee_blend": self._blend01(self.lift_knee_blend),
            "lift_ankle_final_target": float(np.clip(self.lift_ankle_target, -1.0, 1.0)),
            "lift_ankle_raw_target": ankle_raw_target,
            "lift_ankle_blend": self._blend01(self.lift_ankle_blend),
        }

    def _update_recovery_latch(self, risk_level: int, bad_leg: int, contact_steps: int) -> int:
        triggered = (
            self.recovery_hold_steps > 0
            and risk_level >= LIFT_RECOVERY_TRIGGER_RISK_LEVEL
            and bad_leg >= 0
            and contact_steps >= LIFT_RECOVERY_TRIGGER_CONTACT_STEPS
        )
        accepted = False
        if triggered and (self.recovery_hold <= 0 or self.recovery_leg < 0 or bad_leg == self.recovery_leg):
            self.recovery_leg = int(bad_leg)
            self.recovery_hold = self.recovery_hold_steps
            accepted = True
        elif self.recovery_hold > 0:
            self.recovery_hold -= 1
            if self.recovery_hold <= 0:
                self.recovery_leg = -1

        self.recovery_last_triggered = bool(triggered)
        self.recovery_last_accepted = bool(accepted)
        return int(self.recovery_leg if self.recovery_hold > 0 else -1)

    def _ensure_2d(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    def _repeat_latent(self, latent, n):
        if isinstance(latent, dict):
            return {k: (v.repeat_interleave(n, dim=0) if torch.is_tensor(v) else v) for k, v in latent.items()}
        return latent.repeat_interleave(n, dim=0)

    def _decode_prop(self, latent):
        dynamics = self.world_model.dynamics
        feat = dynamics.get_feat(latent) if hasattr(dynamics, "get_feat") else dynamics.get_deter_feat(latent)
        pred = self.world_model.heads["decoder"](feat)
        if isinstance(pred, dict):
            pred = pred["prop"]
        mean = getattr(pred, "mean", None)
        if callable(mean):
            return mean()
        if torch.is_tensor(mean):
            return mean
        return pred

    def _lift_candidate(self, nominal: torch.Tensor, leg_id: int, risk_level: int) -> torch.Tensor:
        leg_id = int(max(0, min(5, leg_id)))
        base = leg_id * 3
        hip = base
        knee = base + 1
        ankle = base + 2
        out = nominal.clone()

        pose = self._lift_pose_debug(leg_id)

        hip_dir = float(HIP_FORWARD_DIR[leg_id])
        hip_target = float(pose["lift_hip_forward_target"])
        hip_blend = float(pose["lift_hip_forward_blend"])
        hip_aligned = out[..., hip] * hip_dir
        hip_target_aligned = abs(hip_target)
        hip_next_aligned = hip_aligned + hip_blend * (hip_target_aligned - hip_aligned)
        out[..., hip] = torch.where(
            hip_aligned < hip_target_aligned,
            hip_next_aligned * hip_dir,
            out[..., hip],
        )

        knee_dir = float(KNEE_LIFT_DIR[leg_id])
        knee_target = float(pose["lift_knee_raw_target"])
        knee_blend = float(pose["lift_knee_blend"])
        knee_aligned = out[..., knee] * knee_dir
        knee_target_aligned = knee_target * knee_dir
        knee_next_aligned = knee_aligned + knee_blend * (knee_target_aligned - knee_aligned)
        out[..., knee] = torch.where(
            knee_aligned < knee_target_aligned,
            knee_next_aligned * knee_dir,
            out[..., knee],
        )

        ankle_target = float(pose["lift_ankle_raw_target"])
        ankle_blend = float(pose["lift_ankle_blend"])
        out[..., ankle] = out[..., ankle] + ankle_blend * (ankle_target - out[..., ankle])
        return torch.clamp(out, -1.0, 1.0)

    def _build_candidates(
        self,
        nominal: torch.Tensor,
        prev_action: Optional[torch.Tensor],
        risk_state: Optional[RiskState] = None,
        risk_level_override: Optional[int] = None,
        bad_leg_override: Optional[int] = None,
    ) -> List[Tuple[str, torch.Tensor]]:
        risk_level = self._risk_level(risk_state) if risk_level_override is None else int(max(0, min(2, risk_level_override)))
        if bad_leg_override is None:
            bad_leg = self._bad_leg(risk_state)
        else:
            leg = int(bad_leg_override)
            bad_leg = leg if 0 <= leg < 6 else -1
        scale_1, scale_2, blend_prev_gain, ankle_gain = self._risk_scales(risk_level)

        cands: List[Tuple[str, torch.Tensor]] = []
        cands.append(("nominal", nominal))
        cands.append((f"scale_{scale_1:.2f}", nominal * scale_1))
        cands.append((f"scale_{scale_2:.2f}", nominal * scale_2))
        if prev_action is not None:
            cands.append(("blend_prev", blend_prev_gain * prev_action + (1.0 - blend_prev_gain) * nominal))
        ankle_protected = nominal.clone()
        ankle_protected[..., ANKLE_INDEX_LIST] = ankle_protected[..., ANKLE_INDEX_LIST] * ankle_gain
        cands.append(("ankle_protected", ankle_protected))

        lift_legs: List[int] = []
        if risk_level >= 1:
            if bad_leg >= 0:
                lift_legs.append(bad_leg)
            max_lifts = self.max_lift_candidates if risk_level >= 2 else min(self.max_lift_candidates, 2)
            for leg_id in range(max_lifts):
                if leg_id not in lift_legs:
                    lift_legs.append(leg_id)
        for leg_id in lift_legs[: self.max_lift_candidates]:
            cands.append((f"lift_leg_{leg_id}", self._lift_candidate(nominal, leg_id, risk_level)))
        return cands

    @torch.no_grad()
    def select(
        self,
        prev_latent,
        is_first: torch.Tensor,
        action_nominal: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        risk_state: Optional[RiskState] = None,
    ):
        try:
            if self.world_model is None or prev_latent is None:
                self.last_debug = {"selected": "nominal", "score": 0.0, "fallback": 1}
                return torch.clamp(action_nominal, -1.0, 1.0)
            nominal = torch.clamp(self._ensure_2d(action_nominal).to(self.device, dtype=torch.float32), -1.0, 1.0)
            prev = None if prev_action is None else torch.clamp(self._ensure_2d(prev_action).to(self.device, dtype=torch.float32), -1.0, 1.0)
            self._select_count += 1
            risk_level = self._risk_level(risk_state)
            bad_leg = self._bad_leg(risk_state)
            contact_steps = self._contact_steps(risk_state)
            latched_leg = self._update_recovery_latch(risk_level, bad_leg, contact_steps)
            latch_active = latched_leg >= 0 and self.recovery_hold > 0
            effective_bad_leg = int(latched_leg if latch_active else bad_leg)
            effective_risk_level = int(max(risk_level, LIFT_RECOVERY_TRIGGER_RISK_LEVEL if latch_active else 0))

            candidates = self._build_candidates(
                nominal,
                prev,
                risk_state,
                risk_level_override=effective_risk_level,
                bad_leg_override=effective_bad_leg,
            )
            names = [n for n, _ in candidates]
            cand_t = torch.cat([c for _, c in candidates], dim=0)

            n = cand_t.shape[0]
            latent = self._repeat_latent(prev_latent, n)
            dynamics = self.world_model.dynamics
            rssm_action_dim = int(getattr(dynamics, "_num_actions", self.action_dim))
            if rssm_action_dim != self.action_dim and rssm_action_dim % self.action_dim == 0:
                cand_rssm = cand_t.repeat(1, rssm_action_dim // self.action_dim)
            else:
                cand_rssm = cand_t
            is_first_b = is_first.reshape(-1).to(self.device, dtype=torch.float32).repeat_interleave(n)

            score = torch.zeros(n, device=self.device)
            ang_weight = 0.35 * (1.0 + 0.25 * float(effective_risk_level))
            for _ in range(self.horizon):
                latent = dynamics.img_step(latent, cand_rssm, sample=True) if hasattr(dynamics, "img_step") else dynamics.imagine_with_action(latent, cand_rssm)
                pred_prop = self._decode_prop(latent)
                base_ang = pred_prop[:, 0:3]
                grav = pred_prop[:, 3:6]
                grav_dev = torch.norm(grav - torch.tensor([0.0, 0.0, -1.0], device=self.device), dim=-1)
                ang_mag = torch.norm(base_ang, dim=-1)
                score = score + grav_dev + ang_weight * ang_mag
                is_first_b = torch.zeros_like(is_first_b)

            if prev is not None:
                prev_rep = prev.repeat(n, 1)
                smooth_weight = 0.08 * (1.0 + 0.75 * float(effective_risk_level))
                ankle_delta_weight = 0.04 * float(effective_risk_level)
                score = score + smooth_weight * torch.norm(cand_t - prev_rep, dim=-1)
                score = score + ankle_delta_weight * torch.norm((cand_t - prev_rep)[:, ANKLE_INDEX_LIST], dim=-1)

            lift_dir_t = torch.tensor(LIFT_DIR_LIST, device=self.device, dtype=cand_t.dtype)
            ankle = cand_t[:, ANKLE_INDEX_LIST]
            aligned_ankle = ankle * lift_dir_t
            down_motion = torch.relu(-aligned_ankle)
            ankle_weight = 0.15 * (1.0 + 0.80 * float(effective_risk_level))
            down_weight = 0.10 * float(effective_risk_level)
            score = score + ankle_weight * torch.norm(ankle, dim=-1)
            score = score + down_weight * torch.norm(down_motion, dim=-1)

            if effective_risk_level > 0:
                bias = torch.zeros(n, device=self.device, dtype=score.dtype)
                for idx, name in enumerate(names):
                    if name == "nominal":
                        bias[idx] += 0.15 * float(effective_risk_level)
                    if name.startswith("lift_leg_"):
                        try:
                            leg_id = int(name.rsplit("_", 1)[-1])
                        except Exception:
                            leg_id = -1
                        if effective_bad_leg >= 0 and leg_id == effective_bad_leg:
                            bias[idx] -= 0.45 * float(effective_risk_level)
                        elif effective_bad_leg >= 0:
                            bias[idx] += 0.08 * float(effective_risk_level)
                score = score + bias

            forced_lift = False
            forced_lift_name = ""
            forced_lift_reason = ""
            if (
                self._select_count >= 20
                and effective_bad_leg >= 0
                and (
                    latch_active
                    or (
                        risk_level >= LIFT_RECOVERY_TRIGGER_RISK_LEVEL
                        and contact_steps >= LIFT_RECOVERY_TRIGGER_CONTACT_STEPS
                    )
                )
            ):
                target_name = f"lift_leg_{effective_bad_leg}"
                if target_name in names:
                    best_idx = int(names.index(target_name))
                    forced_lift = True
                    forced_lift_name = target_name
                    forced_lift_reason = "recovery_latch" if latch_active else "risk_trigger"
                else:
                    best_idx = int(torch.argmin(score).detach().cpu().item())
            else:
                best_idx = int(torch.argmin(score).detach().cpu().item())
            lift_pose = self._lift_pose_debug(effective_bad_leg if effective_bad_leg >= 0 else 3)
            self.last_debug = {
                "selected": names[best_idx],
                "selected_group": self._candidate_group(names[best_idx]),
                "score": float(score[best_idx].detach().cpu().item()),
                "fallback": 0,
                "num_candidates": int(n),
                "used": bool(best_idx != 0),
                "risk_level": int(risk_level),
                "effective_risk_level": int(effective_risk_level),
                "bad_leg": int(bad_leg),
                "effective_bad_leg": int(effective_bad_leg),
                "forced_lift": bool(forced_lift),
                "forced_lift_name": forced_lift_name,
                "forced_lift_reason": forced_lift_reason,
                "recovery_latch_active": bool(latch_active),
                "recovery_latch_leg": int(latched_leg),
                "recovery_latch_hold": int(self.recovery_hold),
                "recovery_latch_triggered": bool(self.recovery_last_triggered),
                "recovery_latch_accepted": bool(self.recovery_last_accepted),
                "recovery_hold_steps": int(self.recovery_hold_steps),
                **lift_pose,
                "select_count": int(self._select_count),
                "ankle_weight": float(ankle_weight),
                "down_weight": float(down_weight),
            }
            return torch.clamp(cand_t[best_idx:best_idx + 1], -1.0, 1.0)
        except Exception as e:
            self.last_debug = {"selected": "nominal", "score": 0.0, "fallback": 1, "error": str(e)}
            return torch.clamp(action_nominal, -1.0, 1.0)
