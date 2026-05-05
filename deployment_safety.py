import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


ANKLE_INDICES = np.array([2, 5, 8, 11, 14, 17], dtype=np.int64)
HIP_KNEE_INDICES = np.array([i for i in range(18) if i not in set(ANKLE_INDICES.tolist())], dtype=np.int64)
LIFT_DIR = np.array([1, 1, 1, -1, -1, -1], dtype=np.float32)


# Deployment-side execution scale (rad per normalized action unit).
# Recommended: hip/knee ~0.12-0.18. Keep ankle base scale 1.0 when asym mapping is enabled.
DEFAULT_HIP_KNEE_SCALE_RAD = 0.15
DEFAULT_ANKLE_BASE_SCALE_RAD = 1.0

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


def default_action_scale_per_dim() -> np.ndarray:
    out = np.full(18, DEFAULT_HIP_KNEE_SCALE_RAD, dtype=np.float32)
    out[ANKLE_INDICES] = DEFAULT_ANKLE_BASE_SCALE_RAD
    return out


def _ankle_down_mask(delta: np.ndarray) -> np.ndarray:
    ankle_delta = delta[ANKLE_INDICES]
    aligned = ankle_delta * LIFT_DIR
    return aligned < 0.0


def apply_asymmetric_ankle_mapping_from_raw(
    action_raw_clipped: np.ndarray,
    lift_range_rad: float,
    sink_range_rad: float,
) -> np.ndarray:
    out = np.asarray(action_raw_clipped, dtype=np.float32).copy()
    aligned = out[ANKLE_INDICES] * LIFT_DIR
    mapped = np.where(aligned >= 0.0, aligned * lift_range_rad, aligned * sink_range_rad)
    out[ANKLE_INDICES] = mapped * LIFT_DIR
    return out


def policy_action_to_exec_rad(
    action_raw_clipped: np.ndarray,
    action_scale_per_dim: np.ndarray,
    use_asymmetric_ankle_mapping: bool,
    asym_lift_range_rad: float,
    asym_sink_range_rad: float,
) -> np.ndarray:
    raw = np.asarray(action_raw_clipped, dtype=np.float32).copy()
    raw = np.clip(raw, -1.0, 1.0)
    if use_asymmetric_ankle_mapping:
        out = apply_asymmetric_ankle_mapping_from_raw(raw, asym_lift_range_rad, asym_sink_range_rad)
        out[HIP_KNEE_INDICES] = raw[HIP_KNEE_INDICES] * np.asarray(action_scale_per_dim, dtype=np.float32)[HIP_KNEE_INDICES]
        return out
    return raw * np.asarray(action_scale_per_dim, dtype=np.float32)


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


class WorldModelCandidateSelector:
    def __init__(self, world_model, action_dim: int = 18, horizon: int = 4, max_lift_candidates: int = 2, device: str = "cpu"):
        self.world_model = world_model
        self.action_dim = int(action_dim)
        self.horizon = int(max(1, min(5, horizon)))
        self.max_lift_candidates = int(max(0, min(6, max_lift_candidates)))
        self.device = torch.device(device)
        self.last_debug = {}

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

    def _build_candidates(self, nominal: torch.Tensor, prev_action: Optional[torch.Tensor]) -> List[Tuple[str, torch.Tensor]]:
        cands: List[Tuple[str, torch.Tensor]] = []
        cands.append(("nominal", nominal))
        cands.append(("scale_0.7", nominal * 0.7))
        cands.append(("scale_0.5", nominal * 0.5))
        if prev_action is not None:
            cands.append(("blend_prev", 0.6 * prev_action + 0.4 * nominal))
        ankle_protected = nominal.clone()
        ankle_protected[..., ANKLE_INDICES] = ankle_protected[..., ANKLE_INDICES] * 0.5
        cands.append(("ankle_protected", ankle_protected))
        for leg_id in range(self.max_lift_candidates):
            base = leg_id * 3
            lift = nominal.clone()
            lift[..., base + 1] = torch.clamp(lift[..., base + 1] - 0.20, -1.0, 1.0)
            lift[..., base + 2] = torch.clamp(lift[..., base + 2] + 0.25, -1.0, 1.0)
            cands.append((f"lift_leg_{leg_id}", lift))
        return cands

    @torch.no_grad()
    def select(self, prev_latent, is_first: torch.Tensor, action_nominal: torch.Tensor, prev_action: Optional[torch.Tensor] = None):
        try:
            if self.world_model is None or prev_latent is None:
                self.last_debug = {"selected": "nominal", "score": 0.0, "fallback": 1}
                return torch.clamp(action_nominal, -1.0, 1.0)
            nominal = torch.clamp(self._ensure_2d(action_nominal).to(self.device, dtype=torch.float32), -1.0, 1.0)
            prev = None if prev_action is None else torch.clamp(self._ensure_2d(prev_action).to(self.device, dtype=torch.float32), -1.0, 1.0)
            candidates = self._build_candidates(nominal, prev)
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
            for _ in range(self.horizon):
                latent = dynamics.img_step(latent, cand_rssm, sample=True) if hasattr(dynamics, "img_step") else dynamics.imagine_with_action(latent, cand_rssm)
                pred_prop = self._decode_prop(latent)
                base_ang = pred_prop[:, 0:3]
                grav = pred_prop[:, 3:6]
                grav_dev = torch.norm(grav - torch.tensor([0.0, 0.0, -1.0], device=self.device), dim=-1)
                ang_mag = torch.norm(base_ang, dim=-1)
                score = score + grav_dev + 0.35 * ang_mag
                is_first_b = torch.zeros_like(is_first_b)

            if prev is not None:
                score = score + 0.08 * torch.norm(cand_t - prev.repeat(n, 1), dim=-1)
            score = score + 0.15 * torch.norm(cand_t[:, ANKLE_INDICES], dim=-1)

            best_idx = int(torch.argmin(score).detach().cpu().item())
            self.last_debug = {
                "selected": names[best_idx],
                "score": float(score[best_idx].detach().cpu().item()),
                "fallback": 0,
                "num_candidates": int(n),
                "used": bool(best_idx != 0),
            }
            return torch.clamp(cand_t[best_idx:best_idx + 1], -1.0, 1.0)
        except Exception as e:
            self.last_debug = {"selected": "nominal", "score": 0.0, "fallback": 1, "error": str(e)}
            return torch.clamp(action_nominal, -1.0, 1.0)
