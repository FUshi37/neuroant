#!/usr/bin/env python3
"""
Real-robot Dreamer-style training loop for a hexapod.

This file is intentionally self-contained and runnable in --dry-run mode.  For
real hardware, replace RealRobotIO.read_obs/send_action/send_safe_action with
your existing Dynamixel + IMU code.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


OBS_DIM = 45
ACTION_DIM = 18
CONTROL_HZ = 50.0
MAX_IMAGINATION_HORIZON = 5


@dataclass
class TrainConfig:
    device: str = "cpu"
    control_hz: float = CONTROL_HZ
    train_every_steps: int = 5
    imagination_horizon: int = MAX_IMAGINATION_HORIZON
    action_limit: float = 1.0
    policy_lr: float = 3e-4
    grad_clip: float = 10.0
    max_steps: int = 0
    dry_run: bool = True
    battery_voltage_threshold: float = 11.9
    battery_max_steps: int = 20000


class RobotIO(Protocol):
    def read_obs(self) -> torch.Tensor:
        """Return obs_prop with shape [45] or [1, 45]."""

    def get_battery_voltage(self) -> float:
        """Return current battery voltage."""

    def send_action(self, action: torch.Tensor) -> None:
        """Send 18-dim joint position targets to the robot."""

    def send_safe_action(self) -> None:
        """Send a conservative action before waiting for human reset."""


class RealRobotIO:
    """
    Hardware adapter placeholder.

    Connect this to the existing Dynamixel + IMU code:
    - read_obs(): build obs_prop = [ang_vel(3), projected_gravity(3),
      commands(3), joint_pos(18), joint_vel(18)]
    - send_action(): convert the 18 action targets to servo commands

    The training loop is hardware-agnostic so reset handling cannot corrupt the
    world-model latent state.
    """

    def read_obs(self) -> torch.Tensor:
        raise NotImplementedError("Please connect read_obs() to the real robot sensors.")

    def send_action(self, action: torch.Tensor) -> None:
        raise NotImplementedError("Please connect send_action() to Dynamixel targets.")

    def get_battery_voltage(self) -> float:
        raise NotImplementedError("Please connect get_battery_voltage() to servo/battery telemetry.")

    def send_safe_action(self) -> None:
        raise NotImplementedError("Please connect send_safe_action() to a safe standing pose.")


class DryRunRobotIO:
    """Small deterministic plant used only to verify that the loop runs."""

    def __init__(self, device: torch.device):
        self.device = device
        self.step = 0
        self.joint_pos = torch.zeros(ACTION_DIM, device=device)
        self.joint_vel = torch.zeros(ACTION_DIM, device=device)

    def read_obs(self) -> torch.Tensor:
        self.step += 1
        t = self.step / CONTROL_HZ
        ang_vel = torch.tensor(
            [0.02 * math.sin(t), 0.02 * math.cos(0.7 * t), 0.0],
            device=self.device,
        )
        projected_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        commands = torch.zeros(3, device=self.device)
        return torch.cat([ang_vel, projected_gravity, commands, self.joint_pos, self.joint_vel])

    def get_battery_voltage(self) -> float:
        return 12.4

    def send_action(self, action: torch.Tensor) -> None:
        action = action.detach().to(self.device).flatten().clamp(-1.0, 1.0)
        new_pos = 0.95 * self.joint_pos + 0.05 * action
        self.joint_vel = (new_pos - self.joint_pos) * CONTROL_HZ
        self.joint_pos = new_pos

    def send_safe_action(self) -> None:
        self.send_action(torch.zeros(ACTION_DIM, device=self.device))


class ResetDetector:
    def __init__(
        self,
        roll_threshold: float = 0.6,
        pitch_threshold: float = 0.6,
        anomaly_threshold: float = 0.2,
        anomaly_count_limit: int = 10,
    ):
        self.roll_threshold = float(roll_threshold)
        self.pitch_threshold = float(pitch_threshold)
        self.anomaly_threshold = float(anomaly_threshold)
        self.anomaly_count_limit = int(anomaly_count_limit)
        self.anomaly_counter = 0

    def reset(self) -> None:
        self.anomaly_counter = 0

    def check(self, obs: torch.Tensor, anomaly_error: Optional[torch.Tensor]) -> bool:
        obs = obs.detach()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        gravity = obs[..., 3:6]
        # projected_gravity uses [0, 0, -1] as the upright target, so use
        # -gravity_z as the vertical denominator.  Otherwise upright would
        # produce roll ~= pi and immediately request a reset.
        roll = torch.atan2(gravity[..., 1], -gravity[..., 2])
        pitch = torch.atan2(
            -gravity[..., 0],
            torch.sqrt(gravity[..., 1] ** 2 + gravity[..., 2] ** 2).clamp_min(1e-6),
        )
        unstable = torch.logical_or(
            torch.abs(roll) > self.roll_threshold,
            torch.abs(pitch) > self.pitch_threshold,
        ).any().item()

        if anomaly_error is not None:
            err = float(torch.as_tensor(anomaly_error).detach().mean().cpu())
            if err > self.anomaly_threshold:
                self.anomaly_counter += 1
            else:
                self.anomaly_counter = 0
        else:
            self.anomaly_counter = 0

        stuck = self.anomaly_counter > self.anomaly_count_limit
        return bool(unstable or stuck)


class BatteryManager:
    """
    Segment training into battery-limited runs.

    Battery voltage changes actuator speed/torque, so the same action can lead
    to different transitions later in the discharge curve.  Cutting a run before
    voltage sag keeps the data distribution more stationary.
    """

    def __init__(self, voltage_threshold: float = 11.0, max_steps: int = 4000):
        self.voltage_threshold = float(voltage_threshold)
        self.max_steps = int(max_steps)
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1

    def check(self, voltage: Optional[float]) -> bool:
        low_voltage = voltage is not None and float(voltage) < self.voltage_threshold
        step_limit = self.max_steps > 0 and self.step_count >= self.max_steps
        return bool(low_voltage or step_limit)

    def reason(self, voltage: Optional[float]) -> str:
        if voltage is not None and float(voltage) < self.voltage_threshold:
            return f"battery_low:{float(voltage):.2f}V"
        if self.max_steps > 0 and self.step_count >= self.max_steps:
            return f"battery_run_step_limit:{self.step_count}"
        return "battery_ok"

    def reset(self) -> None:
        self.step_count = 0


class HumanResetHandler:
    """
    Human reset is required because a real hexapod cannot be teleported like a
    simulator.  The program only detects unsafe states, stops sending training
    actions, prints clear instructions, and waits for the operator.
    """

    def __init__(self):
        self.waiting_for_reset = False

    def trigger(self, robot: RobotIO) -> None:
        self.waiting_for_reset = True
        try:
            robot.send_safe_action()
        except Exception as exc:
            print(f"Warning: failed to send safe action before reset: {exc}")

        print("\n==============================")
        print("!!! RESET REQUIRED !!!")
        print("Please manually reset the robot to a safe standing pose.")
        print("Keep hands clear after pressing ENTER.")
        print("Then press ENTER to continue...")
        print("==============================\n")
        input()
        self.waiting_for_reset = False

    def trigger_battery(self, robot: RobotIO, voltage: Optional[float]) -> None:
        self.waiting_for_reset = True
        try:
            robot.send_safe_action()
        except Exception as exc:
            print(f"Warning: failed to send safe action before battery reset: {exc}")

        print("\n==============================")
        print("!!! BATTERY RESET REQUIRED !!!")
        if voltage is None:
            print("Current voltage: unavailable")
        else:
            print(f"Current voltage: {float(voltage):.2f} V")
        print("Please replace the battery and reset the robot to a standing pose.")
        print("Then press ENTER to continue...")
        print("==============================\n")
        input()
        self.waiting_for_reset = False


def ensure_batch(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x


def soft_action_limit(action: torch.Tensor, limit: float) -> torch.Tensor:
    """Differentiable action limiting for training paths."""
    limit = float(max(limit, 1e-6))
    return limit * torch.tanh(action / limit)


def decoder_mean(pred):
    if isinstance(pred, dict):
        pred = pred["prop"]
    if torch.is_tensor(pred):
        return pred
    if hasattr(pred, "mode") and callable(pred.mode):
        return pred.mode()
    if hasattr(pred, "mean") and callable(pred.mean):
        return pred.mean()
    return pred


def compute_reward(
    obs: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    cpg_target: torch.Tensor,
) -> torch.Tensor:
    obs = ensure_batch(obs)
    action = ensure_batch(action)
    prev_action = ensure_batch(prev_action)
    cpg_target = ensure_batch(cpg_target)

    gravity = obs[..., 3:6].clamp(-2.0, 2.0)
    ang_vel = obs[..., 0:3].clamp(-8.0, 8.0)
    joint_pos = obs[..., 9:27].clamp(-3.14, 3.14)
    action = action.clamp(-1.0, 1.0)
    prev_action = prev_action.clamp(-1.0, 1.0)

    gravity_target = torch.tensor([0.0, 0.0, -1.0], device=obs.device, dtype=obs.dtype)
    # Use mean-square penalties instead of vector norms.  Norm penalties scale
    # with dimension and are non-smooth at zero; squared means give smaller,
    # steadier gradients and reduce action saturation on real hardware.
    r_upright = -torch.mean((gravity - gravity_target) ** 2, dim=-1)
    r_ang = -torch.mean(ang_vel**2, dim=-1)
    r_cpg = -torch.mean((joint_pos - cpg_target) ** 2, dim=-1)
    r_smooth = -torch.mean((action - prev_action) ** 2, dim=-1)
    r_action = -torch.mean(action**2, dim=-1)
    return 2.0 * r_upright + 0.05 * r_ang + 0.5 * r_cpg + 0.2 * r_smooth + 0.05 * r_action


def get_cpg_targets(horizon: int, device: torch.device, step: int = 0) -> torch.Tensor:
    """Simple bounded CPG target placeholder; replace with your gait generator."""
    horizon = min(int(horizon), MAX_IMAGINATION_HORIZON)
    t = torch.arange(step, step + horizon, device=device, dtype=torch.float32) / CONTROL_HZ
    phase = torch.linspace(0.0, 2.0 * math.pi, ACTION_DIM, device=device)
    return 0.35 * torch.sin(2.0 * math.pi * 1.0 * t[:, None] + phase[None, :])


class SimplePolicy(nn.Module):
    """
    Minimal policy with both required call styles:
    - forward(obs_prop, history, wm_feature) for real control
    - actor(feat) for imagination rollout
    """

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM, feat_dim: int = 64):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim + feat_dim, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        obs_prop: torch.Tensor,
        history: Optional[torch.Tensor],
        wm_feature: torch.Tensor,
    ) -> torch.Tensor:
        del history
        obs_prop = ensure_batch(obs_prop)
        wm_feature = ensure_batch(wm_feature)
        return self.obs_encoder(torch.cat([obs_prop, wm_feature], dim=-1))


class DummyDynamics(nn.Module):
    """Dry-run RSSM-like dynamics; replace with the trained Dreamer world model."""

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM, feat_dim: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.rnn = nn.GRUCell(action_dim + 32, feat_dim)
        self.obs_to_embed = nn.Linear(obs_dim, 32)
        self.decoder = nn.Linear(feat_dim, obs_dim)

    def initial(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {"deter": torch.zeros(batch_size, self.feat_dim, device=device)}

    def obs_step(
        self,
        prev_latent: Optional[Dict[str, torch.Tensor]],
        action: torch.Tensor,
        embed: torch.Tensor,
        is_first: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        del sample
        batch = action.shape[0]
        if prev_latent is None:
            prev_latent = self.initial(batch, action.device)
        if is_first.dim() > 1:
            is_first = is_first.reshape(batch)
        reset_mask = is_first.float().reshape(batch, 1)
        h0 = prev_latent["deter"] * (1.0 - reset_mask)
        prior_h = self.rnn(torch.cat([action, torch.zeros_like(embed)], dim=-1), h0)
        post_h = self.rnn(torch.cat([action, embed], dim=-1), h0)
        return {"deter": post_h}, {"deter": prior_h}

    def img_step(
        self,
        prev_latent: Dict[str, torch.Tensor],
        action: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        del sample
        zeros = torch.zeros(action.shape[0], 32, device=action.device, dtype=action.dtype)
        return {"deter": self.rnn(torch.cat([action, zeros], dim=-1), prev_latent["deter"])}

    def get_feat(self, latent: Dict[str, torch.Tensor]) -> torch.Tensor:
        return latent["deter"]


class DummyWorldModel(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM, feat_dim: int = 64):
        super().__init__()
        self.dynamics = DummyDynamics(obs_dim, action_dim, feat_dim)
        self.encoder_net = self.dynamics.obs_to_embed
        self.heads = nn.ModuleDict({"decoder": self.dynamics.decoder})

    def encoder(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder_net(obs_dict["prop"])


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def imagination_rollout(
    world_model: nn.Module,
    policy: nn.Module,
    latent: Dict[str, torch.Tensor],
    horizon: int,
    cpg_targets: torch.Tensor,
    prev_action: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Short imagination only.  On a real robot, long imagination quickly compounds
    model error and can push the policy toward unsafe actions.
    """
    horizon = min(int(horizon), MAX_IMAGINATION_HORIZON)
    rewards: List[torch.Tensor] = []
    action_prev = ensure_batch(prev_action)

    for t in range(horizon):
        feat = world_model.dynamics.get_feat(latent)
        action = soft_action_limit(policy.actor(feat), 1.0)
        latent = world_model.dynamics.img_step(latent, action, sample=True)
        feat_next = world_model.dynamics.get_feat(latent)
        pred = world_model.heads["decoder"](feat_next)
        obs_pred = decoder_mean(pred)

        reward = compute_reward(obs_pred, action, action_prev, cpg_targets[t])
        rewards.append(reward)
        action_prev = action

    return rewards


def estimate_anomaly(
    world_model: nn.Module,
    latent_prior: Optional[Dict[str, torch.Tensor]],
    obs: torch.Tensor,
) -> Optional[torch.Tensor]:
    if latent_prior is None:
        return None
    with torch.no_grad():
        feat = world_model.dynamics.get_feat(latent_prior)
        pred = world_model.heads["decoder"](feat)
        obs_pred = decoder_mean(pred)
        obs_pred = ensure_batch(obs_pred)
        obs = ensure_batch(obs)
        compare_dim = min(obs_pred.shape[-1], obs.shape[-1])
        return F.mse_loss(obs_pred[..., :compare_dim], obs[..., :compare_dim])


def train_step(
    obs_dict: Dict[str, torch.Tensor],
    prev_latent: Optional[Dict[str, torch.Tensor]],
    action_real: torch.Tensor,
    prev_action: torch.Tensor,
    policy: nn.Module,
    world_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    global_step: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
    """
    One low-frequency policy update.

    The posterior latent updates the real recurrent state.  The prior is used
    only for anomaly prediction, because it represents what the world model
    expected before seeing the new observation.
    """
    obs = obs_dict["prop"]
    is_first = obs_dict["is_first"]
    action_real = ensure_batch(action_real)
    prev_action = ensure_batch(prev_action)

    with torch.no_grad():
        embed = world_model.encoder(obs_dict)
        latent_post, latent_prior = world_model.dynamics.obs_step(
            prev_latent,
            action_real,
            embed,
            is_first,
            sample=True,
        )

    # Keep the world model fixed during online policy updates.  Gradients still
    # flow through its differentiable dynamics to the policy actions.
    for param in world_model.parameters():
        param.requires_grad_(False)

    horizon = min(cfg.imagination_horizon, MAX_IMAGINATION_HORIZON)
    cpg_targets = get_cpg_targets(horizon, obs.device, global_step)

    feat = world_model.dynamics.get_feat(latent_post)
    action_now = soft_action_limit(policy(obs, None, feat.detach()), cfg.action_limit)
    real_reward = compute_reward(obs, action_now, prev_action, cpg_targets[0])

    imag_rewards = imagination_rollout(
        world_model,
        policy,
        latent_post,
        horizon=horizon,
        cpg_targets=cpg_targets,
        prev_action=prev_action,
    )
    imag_reward = torch.stack(imag_rewards).mean()
    loss = -(real_reward.mean() + imag_reward)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
    optimizer.step()

    return latent_post, latent_prior, float(loss.detach().cpu())


class RealRobotDreamerTrainer:
    def __init__(
        self,
        robot: RobotIO,
        policy: nn.Module,
        world_model: nn.Module,
        cfg: TrainConfig,
    ):
        self.robot = robot
        self.policy = policy.to(cfg.device)
        self.world_model = world_model.to(cfg.device)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.policy_lr)
        self.reset_detector = ResetDetector()
        self.battery_manager = BatteryManager(
            voltage_threshold=cfg.battery_voltage_threshold,
            max_steps=cfg.battery_max_steps,
        )
        self.reset_handler = HumanResetHandler()
        self.prev_latent: Optional[Dict[str, torch.Tensor]] = None
        self.prev_action = torch.zeros(1, ACTION_DIM, device=self.device)
        self.need_first = True
        self.global_step = 0

        # The world model should not be corrupted by policy training gradients.
        freeze_module(self.world_model)
        self.zero_wm_feature = self._make_zero_wm_feature()

    def _make_zero_wm_feature(self) -> torch.Tensor:
        dynamics = self.world_model.dynamics
        with torch.no_grad():
            try:
                latent = dynamics.initial(1)
            except TypeError:
                latent = dynamics.initial(1, self.device)
            feat = dynamics.get_feat(latent)
            return torch.zeros_like(feat, device=self.device)

    def reset_world_model_state(self) -> None:
        """
        RSSM latent is a belief over the current physical episode.  A manual
        reset creates a discontinuity that the model did not cause with an
        action, so carrying the old latent across reset would poison training.
        """
        self.prev_latent = None
        self.prev_action.zero_()
        self.need_first = True
        self.reset_detector.reset()

    def reset_after_battery_replacement(self) -> None:
        self.reset_world_model_state()
        self.battery_manager.reset()

    def run(self) -> None:
        dt = 1.0 / float(self.cfg.control_hz)
        print(f"Starting real-robot Dreamer loop at {self.cfg.control_hz:.1f} Hz")
        if self.cfg.dry_run:
            print("Running in --dry-run mode; no hardware commands are sent.")

        while self.cfg.max_steps <= 0 or self.global_step < self.cfg.max_steps:
            loop_start = time.perf_counter()
            obs = self.robot.read_obs().to(self.device, dtype=torch.float32)
            obs = ensure_batch(obs)
            try:
                voltage = float(self.robot.get_battery_voltage())
            except Exception:
                voltage = None

            if self.battery_manager.check(voltage):
                print(f"Stopping training due to battery condition: {self.battery_manager.reason(voltage)}")
                self.reset_handler.trigger_battery(self.robot, voltage)
                self.reset_after_battery_replacement()
                continue

            if self.reset_detector.check(obs, anomaly_error=None):
                self.reset_handler.trigger(self.robot)
                self.reset_world_model_state()
                continue

            obs_dict = {
                "prop": obs,
                "is_first": torch.tensor([self.need_first], device=self.device),
            }

            with torch.no_grad():
                if self.prev_latent is None:
                    wm_feature = self.zero_wm_feature
                else:
                    wm_feature = self.world_model.dynamics.get_feat(self.prev_latent)
                action = self.policy(obs, None, wm_feature).clamp(
                    -self.cfg.action_limit,
                    self.cfg.action_limit,
                )

            self.robot.send_action(action.squeeze(0))

            should_train = self.global_step % max(1, self.cfg.train_every_steps) == 0
            latent_prior = None
            if should_train:
                self.prev_latent, latent_prior, loss = train_step(
                    obs_dict=obs_dict,
                    prev_latent=self.prev_latent,
                    # obs_t is the result of action_{t-1}; action_t has only
                    # just been sent and must not be used to update this obs.
                    action_real=self.prev_action.detach(),
                    prev_action=self.prev_action,
                    policy=self.policy,
                    world_model=self.world_model,
                    optimizer=self.optimizer,
                    cfg=self.cfg,
                    global_step=self.global_step,
                )
                if self.global_step % 50 == 0:
                    print(f"step={self.global_step} loss={loss:.4f}")
            else:
                with torch.no_grad():
                    embed = self.world_model.encoder(obs_dict)
                    self.prev_latent, latent_prior = self.world_model.dynamics.obs_step(
                        self.prev_latent,
                        self.prev_action.detach(),
                        embed,
                        obs_dict["is_first"],
                        sample=True,
                    )

            anomaly_error = estimate_anomaly(self.world_model, latent_prior, obs)
            if self.reset_detector.check(obs, anomaly_error):
                self.reset_handler.trigger(self.robot)
                self.reset_world_model_state()
                continue

            self.need_first = False
            self.prev_action.copy_(action.detach())
            self.global_step += 1
            self.battery_manager.step()

            if self.battery_manager.check(voltage):
                print(f"Stopping training due to battery condition: {self.battery_manager.reason(voltage)}")
                self.reset_handler.trigger_battery(self.robot, voltage)
                self.reset_after_battery_replacement()
                continue

            elapsed = time.perf_counter() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif self.global_step % 50 == 0:
                print(
                    f"Warning: control loop overran by {-sleep_time * 1000.0:.1f} ms; "
                    "increase train_every_steps or move training to another process."
                )


def build_components(cfg: TrainConfig) -> Tuple[RobotIO, nn.Module, nn.Module]:
    device = torch.device(cfg.device)
    if cfg.dry_run:
        world_model = DummyWorldModel(feat_dim=64)
        policy = SimplePolicy(feat_dim=64)
        robot: RobotIO = DryRunRobotIO(device)
        return robot, policy, world_model

    # For the real robot, construct your trained policy/world_model here.
    # The required APIs are:
    #   world_model.encoder(obs_dict)
    #   world_model.dynamics.obs_step(prev_latent, action, embed, is_first, sample=True)
    #   world_model.dynamics.img_step(latent, action, sample=True)
    #   world_model.dynamics.get_feat(latent)
    #   world_model.heads["decoder"](feat)
    #   policy(obs_prop, history, wm_feature)
    #   policy.actor(feat)
    raise NotImplementedError(
        "Set --dry-run for a runnable loop, or connect trained policy/world_model and RealRobotIO."
    )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--real", action="store_true", help="Use real hardware adapter.")
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    parser.add_argument("--train-every-steps", type=int, default=5)
    parser.add_argument("--imagination-horizon", type=int, default=MAX_IMAGINATION_HORIZON)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--battery-voltage-threshold", type=float, default=11.0)
    parser.add_argument("--battery-max-steps", type=int, default=4000)
    args = parser.parse_args()

    horizon = min(int(args.imagination_horizon), MAX_IMAGINATION_HORIZON)
    if args.imagination_horizon > MAX_IMAGINATION_HORIZON:
        print("Warning: imagination horizon capped at 5 for real-robot safety.")

    return TrainConfig(
        device=args.device,
        control_hz=args.control_hz,
        train_every_steps=args.train_every_steps,
        imagination_horizon=horizon,
        max_steps=args.max_steps,
        dry_run=not args.real,
        battery_voltage_threshold=args.battery_voltage_threshold,
        battery_max_steps=args.battery_max_steps,
    )


def main() -> None:
    cfg = parse_args()
    robot, policy, world_model = build_components(cfg)
    trainer = RealRobotDreamerTrainer(robot, policy, world_model, cfg)
    trainer.run()


if __name__ == "__main__":
    main()
