#!/usr/bin/env python3
"""
PC-side remote Dreamer training server.

Run this on the PC.  The Raspberry Pi sends real robot observations over UDP;
the PC returns actions and performs low-frequency policy training.  Reset is a
protocol message: the PC detects reset-needed, the Pi asks the human to reset,
then the Pi sends reset_done so this process clears RSSM latent state.
"""

import argparse
import json
import os
import socket
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from real_robot_dreamer_training import (
    ACTION_DIM,
    MAX_IMAGINATION_HORIZON,
    BatteryManager,
    ResetDetector,
    SimplePolicy,
    DummyWorldModel,
    TrainConfig,
    ensure_batch,
    estimate_anomaly,
    train_step,
)


def _float_list(x) -> list:
    return [float(v) for v in np.asarray(x, dtype=np.float32).reshape(-1)]


class TrainableRWMPolicy(nn.Module):
    """Wrap ActorCriticRWM with the API used by the remote training loop."""

    def __init__(self, actor_critic: nn.Module, history_dim: int, wm_feature_dim: int):
        super().__init__()
        self.actor_critic = actor_critic
        self.history_dim = int(history_dim)
        self.wm_feature_dim = int(wm_feature_dim)

    def _adapt_history(self, history: Optional[torch.Tensor], batch: int, device) -> torch.Tensor:
        if history is None:
            return torch.zeros(batch, self.history_dim, device=device)
        history = ensure_batch(history).to(device=device, dtype=torch.float32)
        if history.shape[-1] == self.history_dim:
            return history
        if history.shape[-1] > self.history_dim:
            return history[..., -self.history_dim :]
        pad = torch.zeros(batch, self.history_dim - history.shape[-1], device=device)
        return torch.cat([pad, history], dim=-1)

    def _adapt_wm_feature(self, wm_feature: torch.Tensor) -> torch.Tensor:
        wm_feature = ensure_batch(wm_feature)
        if wm_feature.shape[-1] == self.wm_feature_dim:
            return wm_feature
        if wm_feature.shape[-1] > self.wm_feature_dim:
            # RSSM.get_feat() is [stoch, deter]; deployed policy usually expects deter only.
            return wm_feature[..., -self.wm_feature_dim :]
        pad = torch.zeros(
            *wm_feature.shape[:-1],
            self.wm_feature_dim - wm_feature.shape[-1],
            device=wm_feature.device,
            dtype=wm_feature.dtype,
        )
        return torch.cat([wm_feature, pad], dim=-1)

    def forward(
        self,
        obs_prop: torch.Tensor,
        history: Optional[torch.Tensor],
        wm_feature: torch.Tensor,
    ) -> torch.Tensor:
        obs_prop = ensure_batch(obs_prop)
        batch = obs_prop.shape[0]
        history = self._adapt_history(history, batch, obs_prop.device)
        wm_feature = self._adapt_wm_feature(wm_feature)
        latent_vector = self.actor_critic.history_encoder(history)
        command = obs_prop[:, 6:9]
        wm_latent = self.actor_critic.wm_feature_encoder(wm_feature)
        actor_in = torch.cat([latent_vector, command, wm_latent], dim=-1)
        return self.actor_critic.actor(actor_in)

    def actor(self, wm_feature: torch.Tensor) -> torch.Tensor:
        """Action head used during imagination when no real history is available."""
        wm_feature = self._adapt_wm_feature(wm_feature)
        batch = wm_feature.shape[0]
        latent_dim = int(self.actor_critic.history_encoder[-1].out_features)
        latent_vector = torch.zeros(batch, latent_dim, device=wm_feature.device, dtype=wm_feature.dtype)
        command = torch.zeros(batch, 3, device=wm_feature.device, dtype=wm_feature.dtype)
        wm_latent = self.actor_critic.wm_feature_encoder(wm_feature)
        actor_in = torch.cat([latent_vector, command, wm_latent], dim=-1)
        return self.actor_critic.actor(actor_in)


class ActionHistoryDynamicsAdapter(nn.Module):
    """Let a RSSM trained with action history accept a live 18-dim action."""

    HISTORY_KEY = "_action_history"

    def __init__(self, dynamics: nn.Module, action_dim: int = ACTION_DIM):
        super().__init__()
        self.dynamics = dynamics
        self.action_dim = int(action_dim)
        self.rssm_action_dim = int(getattr(dynamics, "_num_actions", action_dim))
        if self.rssm_action_dim % self.action_dim != 0:
            raise ValueError(
                f"RSSM action dim {self.rssm_action_dim} is not a multiple of {self.action_dim}"
            )
        self.history_len = self.rssm_action_dim // self.action_dim

    def _strip_history(self, latent):
        if isinstance(latent, dict) and self.HISTORY_KEY in latent:
            return {k: v for k, v in latent.items() if k != self.HISTORY_KEY}
        return latent

    def _history_from_latent(self, latent, action: torch.Tensor) -> torch.Tensor:
        batch = action.shape[0]
        if isinstance(latent, dict) and self.HISTORY_KEY in latent:
            return latent[self.HISTORY_KEY]
        return torch.zeros(
            batch,
            self.history_len,
            self.action_dim,
            device=action.device,
            dtype=action.dtype,
        )

    def _reset_history_where_first(self, history: torch.Tensor, is_first: torch.Tensor) -> torch.Tensor:
        if is_first is None:
            return history
        is_first = is_first.reshape(history.shape[0], 1, 1).to(device=history.device, dtype=history.dtype)
        return history * (1.0 - is_first)

    def _append_action(self, history: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        new_history = torch.cat([history[:, 1:], action[:, None, :]], dim=1)
        return new_history.reshape(action.shape[0], -1), new_history

    def _rssm_action_and_history(self, latent, action: torch.Tensor, is_first=None) -> Tuple[torch.Tensor, torch.Tensor]:
        action = ensure_batch(action)
        if action.shape[-1] == self.rssm_action_dim:
            history = action.reshape(action.shape[0], self.history_len, self.action_dim)
            history = self._reset_history_where_first(history, is_first)
            return history.reshape(action.shape[0], -1), history
        if action.shape[-1] == self.action_dim:
            history = self._history_from_latent(latent, action)
            history = self._reset_history_where_first(history, is_first)
            return self._append_action(history, action)
        raise ValueError(
            f"Cannot map action dim {action.shape[-1]} to RSSM action dim {self.rssm_action_dim}"
        )

    def obs_step(self, prev_latent, action, embed, is_first, sample=True):
        rssm_action, new_history = self._rssm_action_and_history(prev_latent, action, is_first)
        post, prior = self.dynamics.obs_step(
            self._strip_history(prev_latent),
            rssm_action,
            embed,
            is_first,
            sample,
        )
        post[self.HISTORY_KEY] = new_history
        prior[self.HISTORY_KEY] = new_history
        return post, prior

    def img_step(self, latent, action, sample=True):
        rssm_action, new_history = self._rssm_action_and_history(latent, action)
        prior = self.dynamics.img_step(self._strip_history(latent), rssm_action, sample)
        prior[self.HISTORY_KEY] = new_history
        return prior

    def get_feat(self, latent):
        return self.dynamics.get_feat(latent)

    def get_deter_feat(self, latent):
        return self.dynamics.get_deter_feat(latent)

    def initial(self, *args, **kwargs):
        return self.dynamics.initial(*args, **kwargs)


class WorldModelActionAdapter(nn.Module):
    """World model proxy with action-dim adaptation for obs_step/img_step."""

    def __init__(self, world_model: nn.Module):
        super().__init__()
        self.world_model = world_model
        self.dynamics = ActionHistoryDynamicsAdapter(world_model.dynamics)
        self.heads = world_model.heads

    def encoder(self, obs_dict):
        return self.world_model.encoder(obs_dict)


def _set_runtime_device_attrs(module: nn.Module, device: torch.device) -> None:
    """
    Some world_model classes cache a device string in plain attributes such as
    `self.device` / `self._device`.  `module.to(cuda)` moves parameters and
    buffers, but it does not update those attributes, so RSSM.initial(),
    obs_step(), img_step() may still create CPU tensors and later fail at cat().
    """
    device_str = str(device)
    for submodule in module.modules():
        if hasattr(submodule, "device"):
            try:
                submodule.device = device
            except Exception:
                pass
        if hasattr(submodule, "_device"):
            try:
                submodule._device = device
            except Exception:
                pass
        cfg = getattr(submodule, "_config", None)
        if cfg is not None and hasattr(cfg, "device"):
            try:
                cfg.device = device_str
            except Exception:
                pass


def _make_real_components(args) -> Tuple[nn.Module, nn.Module, torch.device]:
    if not args.model_path:
        raise ValueError("Please provide --model-path for real mode.")

    if args.disable_torch_compile:
        # Remote real-robot training should prioritize predictable latency and
        # memory usage.  The deployment helper compiles WM/policy by default,
        # which can trigger CUDA allocations even when the model is loaded on
        # CPU, so disable it unless explicitly requested.
        os.environ["DISABLE_TORCH_COMPILE"] = "1"

    from test_rwm_real_robot_wm import RealRobotRWMInference

    # Load checkpoints on CPU first.  Loading directly with map_location="cuda"
    # can double-spike VRAM and fail before we can choose which modules actually
    # need to run on GPU.
    inference = RealRobotRWMInference(
        args.model_path,
        device=args.load_device,
        remove_dof_vel=args.remove_dof_vel,
    )
    if not getattr(inference, "model_loaded", False):
        raise RuntimeError(f"Failed to load checkpoint: {args.model_path}")
    if getattr(inference, "world_model", None) is None:
        raise RuntimeError("Checkpoint did not contain a usable world_model_dict.")

    actor_critic = inference.actor_critic
    actor_critic.train()
    policy = TrainableRWMPolicy(
        actor_critic=actor_critic,
        history_dim=int(inference.history_dim),
        wm_feature_dim=int(inference.wm_feature_dim),
    )
    world_model = WorldModelActionAdapter(inference.world_model)
    world_model.eval()

    runtime_device = torch.device(args.device)
    if runtime_device.type == "cuda":
        try:
            policy = policy.to(runtime_device)
            world_model = world_model.to(runtime_device)
            _set_runtime_device_attrs(world_model, runtime_device)
            torch.cuda.empty_cache()
            print(f"[PC] moved policy/world_model to {runtime_device}")
        except torch.cuda.OutOfMemoryError as exc:
            print(f"[PC] CUDA OOM while moving model to {runtime_device}: {exc}")
            print("[PC] falling back to CPU. Use --device cpu explicitly if this is expected.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            runtime_device = torch.device("cpu")
            policy = policy.to(runtime_device)
            world_model = world_model.to(runtime_device)
            _set_runtime_device_attrs(world_model, runtime_device)
            args.device = "cpu"
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"[PC] CUDA OOM while moving model to {runtime_device}: {exc}")
            print("[PC] falling back to CPU. Use --device cpu explicitly if this is expected.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            runtime_device = torch.device("cpu")
            policy = policy.to(runtime_device)
            world_model = world_model.to(runtime_device)
            _set_runtime_device_attrs(world_model, runtime_device)
            args.device = "cpu"
    else:
        policy = policy.to(runtime_device)
        world_model = world_model.to(runtime_device)
        _set_runtime_device_attrs(world_model, runtime_device)

    print(
        "[PC] loaded real checkpoint: "
        f"history_dim={inference.history_dim}, wm_feature_dim={inference.wm_feature_dim}, "
        f"rssm_action_dim={world_model.dynamics.rssm_action_dim}, runtime_device={runtime_device}"
    )
    return policy, world_model, runtime_device


class PCRemoteDreamerServer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.cfg = TrainConfig(
            device=args.device,
            train_every_steps=args.train_every_steps,
            imagination_horizon=min(args.imagination_horizon, MAX_IMAGINATION_HORIZON),
            policy_lr=args.policy_lr,
            grad_clip=args.grad_clip,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            self.policy = SimplePolicy(feat_dim=64).to(self.device)
            self.world_model = DummyWorldModel(feat_dim=64).to(self.device)
            print("[PC] dry-run trainable policy/world_model loaded")
        else:
            self.policy, self.world_model, self.device = _make_real_components(args)
            self.cfg.device = str(self.device)

        for p in self.world_model.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.policy_lr)

        self.reset_detector = ResetDetector(
            roll_threshold=args.roll_threshold,
            pitch_threshold=args.pitch_threshold,
            anomaly_threshold=args.anomaly_threshold,
            anomaly_count_limit=args.anomaly_count_limit,
        )
        self.battery_manager = BatteryManager(
            voltage_threshold=args.battery_voltage_threshold,
            max_steps=args.battery_max_steps,
        )
        self.prev_latent: Optional[Dict[str, torch.Tensor]] = None
        self.prev_action = torch.zeros(1, ACTION_DIM, device=self.device)
        self.need_first = True
        self.reset_pending = False
        self.battery_pending = False
        self.run_id = 0
        self.step_count = 0
        self.last_loss = None
        self.train_enabled = bool(args.online_train)

        self.zero_wm_feature = self._make_zero_wm_feature()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((args.host, args.port))
        self.sock.settimeout(1.0)

    def _make_zero_wm_feature(self) -> torch.Tensor:
        dynamics = self.world_model.dynamics
        with torch.no_grad():
            try:
                latent = dynamics.initial(1)
            except TypeError:
                latent = dynamics.initial(1, self.device)
            return torch.zeros_like(dynamics.get_feat(latent), device=self.device)

    def _clear_latent_after_human_reset(self) -> None:
        # A manual reset is not a physical transition predicted by the RSSM.
        # Clear all recurrent state and mark the next real observation as a new episode.
        self.prev_latent = None
        self.prev_action.zero_()
        self.need_first = True
        self.reset_pending = False
        self.battery_pending = False
        self.reset_detector.reset()
        self.battery_manager.reset()
        self.train_enabled = bool(self.args.online_train)
        self.run_id += 1
        print(
            "[PC] reset_done received: latent cleared, battery/run counter reset, "
            "next obs will use is_first=True"
        )

    def _send(self, addr, payload: dict) -> None:
        self.sock.sendto(json.dumps(payload).encode("utf-8"), addr)

    def _send_reset_required(self, addr, step: int, reason: str) -> None:
        self.reset_pending = True
        self._send(
            addr,
            {
                "type": "reset_required",
                "step": int(step),
                "ok": False,
                "reason": reason,
                "safe_action": [0.0] * ACTION_DIM,
            },
        )

    def _send_battery_reset_required(
        self,
        addr,
        step: int,
        voltage: Optional[float],
        reason: str,
    ) -> None:
        self.battery_pending = True
        self.reset_pending = True
        # Stop training immediately at the run boundary.  Battery sag changes
        # actuator response, so continuing would mix different dynamics into one
        # episode and corrupt the RSSM latent/state targets.
        self.train_enabled = False
        self._send(
            addr,
            {
                "type": "battery_reset_required",
                "step": int(step),
                "ok": False,
                "reason": reason,
                "voltage": None if voltage is None else float(voltage),
                "safe_action": [0.0] * ACTION_DIM,
            },
        )

    def _handle_obs(self, msg: dict, addr) -> None:
        step = int(msg["step"])
        obs = torch.tensor(msg["obs"], device=self.device, dtype=torch.float32).reshape(1, -1)
        voltage_msg = msg.get("voltage")
        voltage = None if voltage_msg is None else float(voltage_msg)
        prev_action_msg = msg.get("prev_action")
        if prev_action_msg is not None:
            self.prev_action = torch.tensor(
                prev_action_msg, device=self.device, dtype=torch.float32
            ).reshape(1, -1)

        if self.battery_pending:
            self._send_battery_reset_required(
                addr,
                step,
                voltage,
                "waiting_for_battery_replacement",
            )
            return

        if self.reset_pending:
            self._send_reset_required(addr, step, "waiting_for_human_reset")
            return

        if self.battery_manager.check(voltage):
            reason = self.battery_manager.reason(voltage)
            print(
                f"[PC] stopping run_id={self.run_id} for battery condition: "
                f"{reason}, voltage={voltage}, run_steps={self.battery_manager.step_count}"
            )
            self._send_battery_reset_required(addr, step, voltage, reason)
            return

        if self.reset_detector.check(obs, anomaly_error=None):
            self._send_reset_required(addr, step, "roll_pitch_unstable")
            return

        obs_dict = {
            "prop": obs,
            "is_first": torch.tensor([self.need_first], device=self.device),
        }

        t0 = time.perf_counter()
        with torch.no_grad():
            wm_feature = (
                self.zero_wm_feature
                if self.prev_latent is None
                else self.world_model.dynamics.get_feat(self.prev_latent)
            )
            history = None
            if "history" in msg and msg["history"] is not None:
                history = torch.tensor(msg["history"], device=self.device, dtype=torch.float32).reshape(1, -1)
            action = self.policy(obs, history, wm_feature).clamp(-self.args.action_limit, self.args.action_limit)

        self._send(
            addr,
            {
                "type": "act",
                "step": step,
                "ok": True,
                "action_raw": _float_list(action.detach().cpu().numpy()),
                "server_ms": float((time.perf_counter() - t0) * 1000.0),
                "is_first": bool(self.need_first),
                "last_loss": self.last_loss,
            },
        )

        latent_prior = None
        should_train = (
            self.train_enabled
            and self.step_count >= int(self.args.train_start_steps)
            and self.step_count % max(1, self.args.train_every_steps) == 0
        )
        train_ms = 0.0
        if should_train:
            train_t0 = time.perf_counter()
            self.prev_latent, latent_prior, loss = train_step(
                obs_dict=obs_dict,
                prev_latent=self.prev_latent,
                # The current observation was caused by the previous action.
                # This matches RealRobotRWMInference.update_world_model().
                action_real=self.prev_action.detach(),
                prev_action=self.prev_action,
                policy=self.policy,
                world_model=self.world_model,
                optimizer=self.optimizer,
                cfg=self.cfg,
                global_step=self.step_count,
            )
            self.last_loss = loss
            train_ms = float((time.perf_counter() - train_t0) * 1000.0)
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
            self.reset_pending = True
            print(f"[PC] reset will be requested on next packet, anomaly={anomaly_error}")

        self.prev_action.copy_(action.detach())
        self.need_first = False
        self.step_count += 1
        self.battery_manager.step()

        if self.step_count <= 5 or self.step_count % max(1, self.args.log_every) == 0:
            print(
                f"[PC] step={step} train={int(should_train)} "
                f"loss={self.last_loss} train_ms={train_ms:.1f} "
                f"voltage={voltage} run_steps={self.battery_manager.step_count} "
                f"action[min={float(action.min().detach().cpu()):.4f}, "
                f"max={float(action.max().detach().cpu()):.4f}, "
                f"mean={float(action.mean().detach().cpu()):.4f}]"
            )

    def serve_forever(self) -> None:
        print(f"[PC] listening on udp://{self.args.host}:{self.args.port}")
        print("[PC] start the Pi client after this line appears")
        while True:
            try:
                data, addr = self.sock.recvfrom(1024 * 1024)
            except socket.timeout:
                continue

            try:
                msg = json.loads(data.decode("utf-8"))
                msg_type = msg.get("type")
                if msg_type == "reset_done":
                    self._clear_latent_after_human_reset()
                    self._send(addr, {"type": "reset_ack", "step": int(msg.get("step", -1)), "ok": True})
                elif msg_type == "obs":
                    self._handle_obs(msg, addr)
            except Exception as exc:
                print(f"[PC] packet error from {addr}: {exc}")
                try:
                    self._send(addr, {"type": "error", "ok": False, "error": str(exc)})
                except Exception:
                    pass


def parse_args():
    parser = argparse.ArgumentParser(description="PC remote Dreamer training server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--load-device",
        default="cpu",
        help="Device used only for checkpoint loading. Keep this as cpu to avoid CUDA OOM.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use small dummy trainable modules.")
    parser.add_argument("--model-path", default=None, help="Checkpoint path used by your _make_real_components().")
    parser.add_argument("--remove-dof-vel", action="store_true", help="Match checkpoints trained without joint velocity.")
    parser.add_argument(
        "--disable-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable torch.compile in the imported deployment helper.",
    )
    parser.add_argument("--train-every-steps", type=int, default=5)
    parser.add_argument(
        "--online-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient updates in the UDP control server. Default off for real-time safety.",
    )
    parser.add_argument(
        "--train-start-steps",
        type=int,
        default=500,
        help="Delay online gradient updates until control/network are stable.",
    )
    parser.add_argument("--imagination-horizon", type=int, default=5)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--action-limit", type=float, default=1.0)
    parser.add_argument("--roll-threshold", type=float, default=0.6)
    parser.add_argument("--pitch-threshold", type=float, default=0.6)
    parser.add_argument("--anomaly-threshold", type=float, default=0.2)
    parser.add_argument("--anomaly-count-limit", type=int, default=10)
    parser.add_argument(
        "--battery-voltage-threshold",
        type=float,
        default=11.0,
        help="Request battery replacement below this voltage.",
    )
    parser.add_argument(
        "--battery-max-steps",
        type=int,
        default=4000,
        help="Maximum 50Hz control steps per battery run; 0 disables step limit.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if args.imagination_horizon > MAX_IMAGINATION_HORIZON:
        print("[PC] imagination horizon capped to 5")
        args.imagination_horizon = MAX_IMAGINATION_HORIZON
    return args


def main() -> None:
    args = parse_args()
    server = PCRemoteDreamerServer(args)
    server.serve_forever()


if __name__ == "__main__":
    main()
