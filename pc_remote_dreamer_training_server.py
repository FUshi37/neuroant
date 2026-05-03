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
import socket
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from real_robot_dreamer_training import (
    ACTION_DIM,
    MAX_IMAGINATION_HORIZON,
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


def _make_real_components(args) -> Tuple[nn.Module, nn.Module]:
    """
    Replace this function with your real trainable policy and world model.

    Required APIs:
      policy(obs_prop, history, wm_feature) -> [B, 18]
      policy.actor(feat) -> [B, 18] for short imagination
      world_model.encoder(obs_dict)
      world_model.dynamics.obs_step(prev_latent, action, embed, is_first, sample=True)
      world_model.dynamics.img_step(latent, action, sample=True)
      world_model.dynamics.get_feat(latent)
      world_model.heads["decoder"](feat)
    """
    raise NotImplementedError(
        "Real trainable components are not wired yet. Start with --dry-run, "
        "then edit _make_real_components() to load your policy/world_model checkpoint."
    )


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
            self.policy, self.world_model = _make_real_components(args)
            self.policy = self.policy.to(self.device)
            self.world_model = self.world_model.to(self.device)

        for p in self.world_model.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.policy_lr)

        self.reset_detector = ResetDetector(
            roll_threshold=args.roll_threshold,
            pitch_threshold=args.pitch_threshold,
            anomaly_threshold=args.anomaly_threshold,
            anomaly_count_limit=args.anomaly_count_limit,
        )
        self.prev_latent: Optional[Dict[str, torch.Tensor]] = None
        self.prev_action = torch.zeros(1, ACTION_DIM, device=self.device)
        self.need_first = True
        self.reset_pending = False
        self.step_count = 0
        self.last_loss = None

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
        self.reset_detector.reset()
        print("[PC] reset_done received: latent cleared, next obs will use is_first=True")

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

    def _handle_obs(self, msg: dict, addr) -> None:
        step = int(msg["step"])
        obs = torch.tensor(msg["obs"], device=self.device, dtype=torch.float32).reshape(1, -1)
        prev_action_msg = msg.get("prev_action")
        if prev_action_msg is not None:
            self.prev_action = torch.tensor(
                prev_action_msg, device=self.device, dtype=torch.float32
            ).reshape(1, -1)

        if self.reset_pending:
            self._send_reset_required(addr, step, "waiting_for_human_reset")
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
        should_train = self.step_count % max(1, self.args.train_every_steps) == 0
        if should_train:
            self.prev_latent, latent_prior, loss = train_step(
                obs_dict=obs_dict,
                prev_latent=self.prev_latent,
                action_real=action.detach(),
                prev_action=self.prev_action,
                policy=self.policy,
                world_model=self.world_model,
                optimizer=self.optimizer,
                cfg=self.cfg,
                global_step=self.step_count,
            )
            self.last_loss = loss
        else:
            with torch.no_grad():
                embed = self.world_model.encoder(obs_dict)
                self.prev_latent, latent_prior = self.world_model.dynamics.obs_step(
                    self.prev_latent,
                    action.detach(),
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

        if self.step_count <= 5 or self.step_count % max(1, self.args.log_every) == 0:
            print(
                f"[PC] step={step} train={int(should_train)} "
                f"loss={self.last_loss} action_mean={float(action.mean().detach().cpu()):.4f}"
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
    parser.add_argument("--dry-run", action="store_true", help="Use small dummy trainable modules.")
    parser.add_argument("--model-path", default=None, help="Checkpoint path used by your _make_real_components().")
    parser.add_argument("--train-every-steps", type=int, default=5)
    parser.add_argument("--imagination-horizon", type=int, default=5)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--action-limit", type=float, default=1.0)
    parser.add_argument("--roll-threshold", type=float, default=0.6)
    parser.add_argument("--pitch-threshold", type=float, default=0.6)
    parser.add_argument("--anomaly-threshold", type=float, default=0.2)
    parser.add_argument("--anomaly-count-limit", type=int, default=10)
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
