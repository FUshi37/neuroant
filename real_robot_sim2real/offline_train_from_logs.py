import argparse
import glob
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# Ensure parent project dir is importable when running this file directly.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from test_rwm_real_robot import RealRobotRWMInference


def load_chunks(dataset_dir: str, max_chunks: int = 0) -> List[Dict[str, np.ndarray]]:
    files = sorted(glob.glob(os.path.join(dataset_dir, "train_chunk_*.npz")))
    if max_chunks > 0:
        files = files[:max_chunks]
    chunks = []
    for f in files:
        d = np.load(f)
        chunks.append({k: d[k] for k in d.files})
    return chunks


def concat_field(chunks: List[Dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.concatenate([c[key] for c in chunks], axis=0)


def build_sequence_indices(total: int, seq_len: int, stride: int) -> np.ndarray:
    if total < seq_len:
        return np.zeros((0,), dtype=np.int64)
    return np.arange(0, total - seq_len + 1, stride, dtype=np.int64)


def sample_wm_batch(
    data: Dict[str, np.ndarray],
    starts: np.ndarray,
    batch_size: int,
    seq_len: int,
) -> Dict[str, np.ndarray]:
    sel = np.random.choice(starts, size=batch_size, replace=len(starts) < batch_size)
    prop, hist, act, rew, first = [], [], [], [], []
    for s in sel:
        e = s + seq_len
        prop.append(data["obs"][s:e])
        hist.append(data["history"][s:e])
        act.append(data["wm_action"][s:e])
        rew.append(data["reward"][s:e])
        f = np.zeros((seq_len,), dtype=np.float32)
        f[0] = 1.0
        first.append(f)
    return {
        "prop": np.stack(prop, axis=0),
        "history": np.stack(hist, axis=0),
        "action": np.stack(act, axis=0),
        "reward": np.stack(rew, axis=0),
        "is_first": np.stack(first, axis=0),
    }


def offline_train(args):
    # 离线训练在 CPU（尤其树莓派）上：torch.compile 首次构图/反向极慢，且看起来像“卡死”。
    # 默认关闭 compile；需要时再传 --torch-compile。
    if not args.torch_compile:
        import test_rwm_real_robot as trr

        trr.WM_OPT_TORCH_COMPILE = False
        trr.POLICY_OPT_TORCH_COMPILE = False

    remove_dof_vel = not args.no_remove_dof_vel
    rwm = RealRobotRWMInference(
        model_path=args.model_path,
        device="cpu",
        remove_dof_vel=remove_dof_vel,
    )
    actor_critic = rwm.actor_critic
    actor_critic.train()
    world_model = rwm.world_model
    if world_model is None:
        raise RuntimeError("checkpoint 中没有 world_model_dict，无法离线训练 WM。")

    policy_opt = torch.optim.Adam(actor_critic.parameters(), lr=args.policy_lr)
    chunks = load_chunks(args.dataset_dir, args.max_chunks)
    if len(chunks) == 0:
        raise RuntimeError(f"未在 {args.dataset_dir} 找到 train_chunk_*.npz")

    use_vel_key = "vel_proxy" if "vel_proxy" in chunks[0] else "vel_est"
    data = {
        "obs": concat_field(chunks, "obs").astype(np.float32),
        "history": concat_field(chunks, "history").astype(np.float32),
        "wm_feature": concat_field(chunks, "wm_feature").astype(np.float32),
        "action": concat_field(chunks, "action").astype(np.float32),
        "reward": concat_field(chunks, "reward").astype(np.float32),
        "vel_est": concat_field(chunks, use_vel_key).astype(np.float32),
        "wm_action": concat_field(chunks, "wm_action").astype(np.float32),
    }
    starts = build_sequence_indices(len(data["obs"]), args.wm_seq_len, args.seq_stride)
    if len(starts) == 0:
        raise RuntimeError("离线数据长度不足以构造 WM 序列。")

    def _imagination_policy_step() -> float:
        """
        Lightweight imagination update:
        - Build RSSM start state from real batch tail.
        - Rollout in latent space with actor-predicted actions.
        - Maximize predicted reward head along imagined horizon.
        """
        batch_np = sample_wm_batch(data, starts, args.imagine_batch_size, args.wm_seq_len)
        prop = torch.tensor(batch_np["prop"], dtype=torch.float32)
        hist = torch.tensor(batch_np["history"], dtype=torch.float32)
        act_hist = torch.tensor(batch_np["action"], dtype=torch.float32)
        is_first = torch.tensor(batch_np["is_first"], dtype=torch.float32)

        with torch.no_grad():
            embed = world_model.encoder({"prop": prop, "is_first": is_first})
            post, _ = world_model.dynamics.observe(embed, act_hist, is_first)
            state = {k: v[:, -1].detach() for k, v in post.items()}
            obs_seed = prop[:, -1].detach()
            hist_seed = hist[:, -1].detach()
            action_hist = act_hist[:, -1].detach()

        # Let gradients flow through actor and WM transition/reward heads.
        imag_rewards = []
        gamma = float(args.imagine_discount)
        for t in range(args.imagine_horizon):
            wm_feat = world_model.dynamics.get_deter_feat(state)
            latent = actor_critic.history_encoder(hist_seed)
            cmd = obs_seed[:, 6:9]
            wm_lat = actor_critic.wm_feature_encoder(wm_feat)
            actor_in = torch.cat((latent, cmd, wm_lat), dim=-1)
            action_now = actor_critic.actor(actor_in)  # deterministic imagined action

            action_hist = torch.cat((action_hist[..., 18:], action_now), dim=-1)
            state = world_model.dynamics.img_step(state, action_hist, sample=True)
            feat = world_model.dynamics.get_feat(state)
            reward_dist = world_model.heads["reward"](feat)
            reward_t = reward_dist.mode().squeeze(-1)
            imag_rewards.append((gamma ** t) * reward_t)

            obs_wo_cmd = torch.cat((obs_seed[:, :6], obs_seed[:, 9:], action_now), dim=-1)
            step_dim = obs_wo_cmd.shape[-1]
            hist_seed = torch.cat((hist_seed[..., step_dim:], obs_wo_cmd), dim=-1)

        if not imag_rewards:
            return 0.0
        imag_return = torch.stack(imag_rewards, dim=0).sum(dim=0).mean()
        imag_loss = -imag_return * float(args.imagine_coef)
        policy_opt.zero_grad()
        imag_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
        policy_opt.step()
        return float(imag_loss.item())

    for epoch in range(args.epochs):
        wm_losses = []
        for wi in range(args.wm_steps_per_epoch):
            t_wm = time.perf_counter()
            batch_np = sample_wm_batch(data, starts, args.wm_batch_size, args.wm_seq_len)
            _, _, metrics = world_model._train(batch_np, act_func=actor_critic.act)
            wm_losses.append(float(metrics.get("model_loss", 0.0)))
            if wi == 0 or (wi + 1) % max(1, args.wm_log_every) == 0:
                dt = time.perf_counter() - t_wm
                mloss = wm_losses[-1]
                print(
                    f"[Offline] epoch {epoch + 1}/{args.epochs} "
                    f"wm_step {wi + 1}/{args.wm_steps_per_epoch} "
                    f"model_loss={mloss:.4f} ({dt:.1f}s)",
                    flush=True,
                )

        # offline policy behavior cloning + vel_predict（实机数据稳定方案）
        idx = np.random.choice(len(data["obs"]), size=min(args.policy_batch_size, len(data["obs"])), replace=False)
        obs_b = torch.tensor(data["obs"][idx], dtype=torch.float32)
        hist_b = torch.tensor(data["history"][idx], dtype=torch.float32)
        wm_b = torch.tensor(data["wm_feature"][idx], dtype=torch.float32)
        act_target = torch.tensor(data["action"][idx], dtype=torch.float32)
        vel_target = torch.tensor(data["vel_est"][idx], dtype=torch.float32)

        latent = actor_critic.history_encoder(hist_b)
        cmd = obs_b[:, 6:9]
        wm_lat = actor_critic.wm_feature_encoder(wm_b)
        feat = torch.cat((latent, cmd, wm_lat), dim=-1)
        act_pred = actor_critic.actor(feat)
        bc_loss = F.mse_loss(act_pred, act_target)
        vel_pred = actor_critic.get_linear_vel(obs_b, hist_b)
        vel_loss = F.mse_loss(vel_pred, vel_target)
        loss = bc_loss + args.vel_predict_coef * vel_loss

        policy_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 1.0)
        policy_opt.step()

        imag_losses = []
        if args.use_imagination:
            for _ in range(args.imagine_steps_per_epoch):
                imag_losses.append(_imagination_policy_step())

        wm_mean = float(np.mean(wm_losses)) if wm_losses else 0.0
        imag_mean = float(np.mean(imag_losses)) if imag_losses else 0.0
        print(
            f"[Offline][Epoch {epoch+1}/{args.epochs}] wm_loss={wm_mean:.4f} "
            f"bc_loss={bc_loss.item():.4f} vel_loss={vel_loss.item():.4f} "
            f"imag_loss={imag_mean:.4f}"
        )

    out = {
        "actor_critic": actor_critic.state_dict(),
        "world_model": world_model.state_dict(),
        "epochs": args.epochs,
    }
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save(out, args.output_ckpt)
    print(f"离线训练完成，保存到: {args.output_ckpt}")


def parse_args():
    p = argparse.ArgumentParser("Offline sim2real training from logged chunks")
    p.add_argument("--model-path", type=str, required=True, help="在线训练前/中 checkpoint 路径")
    p.add_argument(
        "--no-remove-dof-vel",
        action="store_true",
        help="使用含关节速度的本体维数（与仿真 45+cpg 对齐）；默认去掉 dof_vel，与实机 33 维一致",
    )
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="启用 test_rwm_real_robot 内 WM/策略的 torch.compile（首次反向可能极慢；离线默认关闭）",
    )
    p.add_argument(
        "--wm-log-every",
        type=int,
        default=1,
        help="每个 epoch 内每 N 步 WM 训练打印一行进度（默认 1，即每步都打印）",
    )
    p.add_argument("--dataset-dir", type=str, required=True, help="包含 train_chunk_*.npz 的目录")
    p.add_argument("--output-ckpt", type=str, default="real_robot_sim2real/outputs/offline_refined.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-chunks", type=int, default=0, help="0 表示使用全部数据块")
    p.add_argument("--wm-steps-per-epoch", type=int, default=50)
    p.add_argument("--wm-batch-size", type=int, default=8)
    p.add_argument("--wm-seq-len", type=int, default=32)
    p.add_argument("--seq-stride", type=int, default=4)
    p.add_argument("--policy-batch-size", type=int, default=1024)
    p.add_argument("--policy-lr", type=float, default=3e-5)
    p.add_argument("--vel-predict-coef", type=float, default=0.5)
    p.add_argument("--use-imagination", action="store_true")
    p.add_argument("--imagine-steps-per-epoch", type=int, default=10)
    p.add_argument("--imagine-batch-size", type=int, default=8)
    p.add_argument("--imagine-horizon", type=int, default=8)
    p.add_argument("--imagine-discount", type=float, default=0.99)
    p.add_argument("--imagine-coef", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    offline_train(parse_args())
