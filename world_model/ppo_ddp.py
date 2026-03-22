# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from rsl_rl.modules import ActorCritic, ActorCriticWMP, ActorCriticRWM, ActorCriticRWMDDP
from rsl_rl.storage import RolloutStorage
def compare_first_grad(model):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 找第一个有 grad 的参数
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad = p.grad.detach()
            dev = grad.device  # use same device as grad (cuda:0 etc.)

            # flatten to 1D on same device
            g = grad.flatten().to(dev)

            # gather sizes (on same device)
            local_n = torch.tensor([g.numel()], dtype=torch.long, device=dev)
            sizes = [torch.zeros(1, dtype=torch.long, device=dev) for _ in range(world_size)]
            dist.all_gather(sizes, local_n)

            max_n = int(max([int(x.item()) for x in sizes]))
            if g.numel() < max_n:
                pad = torch.zeros(max_n - g.numel(), dtype=g.dtype, device=dev)
                g = torch.cat([g, pad], dim=0)

            gather = [torch.zeros_like(g) for _ in range(world_size)]
            dist.all_gather(gather, g)

            # compare to rank 0
            ref = gather[0]
            diffs = [ (tg - ref).abs().max().item() for tg in gather ]
            if rank == 0:
                print(f"[compare_grads] param: {name}, max_abs_diffs per rank: {diffs}")
            break
def average_gradients(model):
    world_size = torch.distributed.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            # 所有进程的梯度相加
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            # 求平均
            param.grad.data /= world_size

def unwrap_ddp(model):
    """解开 DDP 包裹，返回真实的模型"""
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

class PPO_DDP:
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 vel_predict_coef=1.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 min_std=None,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_std = min_std

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.vel_predict_coef = vel_predict_coef


    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape,
                     history_dim, wm_feature_dim):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape,
                                      history_dim=history_dim,
                                      wm_feature_dim=wm_feature_dim, device=self.device)

    def test_mode(self):
        unwrap_ddp(self.actor_critic).test()

    def train_mode(self):
        unwrap_ddp(self.actor_critic).train()

    def act(self, obs, critic_obs, history, wm_feature):
        ac = unwrap_ddp(self.actor_critic)
        if ac.is_recurrent:
            self.transition.hidden_states = ac.get_hidden_states()
        # Compute the actions and values
        self.transition.history = history
        self.transition.wm_feature = wm_feature.detach()
        aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()

        self.transition.actions = ac.act(aug_obs, history, wm_feature).detach()
        self.transition.values = ac.evaluate(aug_critic_obs, wm_feature).detach()
        self.transition.actions_log_prob = ac.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = ac.action_mean.detach()
        self.transition.action_sigma = ac.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        unwrap_ddp(self.actor_critic).reset(dones)

    def compute_returns(self, last_critic_obs, wm_feature):
        aug_last_critic_obs = last_critic_obs.detach()
        ac = unwrap_ddp(self.actor_critic)
        last_values = ac.evaluate(aug_last_critic_obs, wm_feature).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_vel_predict_loss = 0

        ac = unwrap_ddp(self.actor_critic)

        if ac.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, history_batch, wm_feature_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            aug_obs_batch = obs_batch.detach()
            ac.act(aug_obs_batch, history_batch, wm_feature_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = ac.get_actions_log_prob(actions_batch)

            aug_critic_obs_batch = critic_obs_batch.detach()
            value_batch = ac.evaluate(aug_critic_obs_batch, wm_feature_batch, masks=masks_batch,
                                      hidden_states=hid_states_batch[1])
            mu_batch = ac.action_mean
            sigma_batch = ac.action_std
            entropy_batch = ac.entropy

            # KL penalty
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) +
                        (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) /
                        (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # PPO surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                              1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Linear vel predict loss
            predicted_linear_vel = ac.get_linear_vel(aug_obs_batch, history_batch)
            target_linear_vel = aug_critic_obs_batch[:, ac.privileged_dim - 3: ac.privileged_dim]
            vel_predict_loss = (predicted_linear_vel - target_linear_vel).pow(2).mean()

            # Total loss
            loss = surrogate_loss + self.value_loss_coef * value_loss \
                   - self.entropy_coef * entropy_batch.mean() \
                   + self.vel_predict_coef * vel_predict_loss

            self.optimizer.zero_grad()
            
            # 在 update() 的 loss 计算之后，执行：
            # INJECT_LOSS = 1e3  # 或 1e4，越大效果越明显
            # if dist.is_available() and dist.is_initialized():
            #    rank = dist.get_rank()
            #else:
            #    rank = 0

            # 只在 rank==1 上把 loss 加一个会产生梯度的项（例如 INJECT_LOSS * target_param.sum()）
            # 找一个小参数张量 target_param 做线性组合加入 loss （不要用常数，会没有梯度）
            # target_param = None
            # for name, p in self.actor_critic.named_parameters():
            #     if p.requires_grad:
            #         target_param = p
            #         target_name = name
            #         break

            # 在 rank 1 上修改 loss
            # if rank == 1:
            #     loss = loss + INJECT_LOSS * target_param.view(-1)[0]  # 这是个 scalar * param_element => 会影响 grad
            # 现在所有 rank 调用 backward（DDP 在 backward 内会 all_reduce grads）
            #with torch.cuda.amp.autocast():
            loss.backward()

            #torch.cuda.synchronize()
            #torch.distributed.barrier()

            #torch.cuda.synchronize()
            #torch.distributed.barrier()
            #compare_first_grad(self.actor_critic)
            #for name, param in self.actor_critic.named_parameters():
            #    if param.grad is not None:
            #        grad_vel = param.grad.view(-1)[0].item()
            #        print(f"[Rank {dist.get_rank()}] actor_critic {name} grad_vel: {grad_vel}")
            #        break
            #for name, param in self._world_model.named_parameters():
            #    if param.grad is not None:
            #        grad_vel = param.grad.view(-1)[0].item()
            #        print(f"[Rank {dist.get_rank()}] world_model {name} grad_vel: {grad_vel}")
            #average_gradients(self.actor_critic)
            #g_after = target_param.grad.view(-1)[0].detach().cpu().item()
            #print(f"[Rank {rank}] after allreduce grad sample: {g_after}")
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            average_gradients(self.actor_critic)
            self.optimizer.step()

            if not ac.fixed_std and self.min_std is not None:
                ac.std.data = ac.std.data.clamp(min=self.min_std)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_vel_predict_loss += vel_predict_loss.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_vel_predict_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_vel_predict_loss