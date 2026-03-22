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

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import time
import os
from collections import deque
import statistics
import wandb
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from rsl_rl.algorithms import AMPPPO, PPO, PPO_DDP
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticRWM, ActorCriticRWMDDP
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer
from rsl_rl.modules import DepthPredictor
import torch.optim as optim

from copy import copy, deepcopy

from dreamer.models import *
import ruamel.yaml as yaml
import argparse
import pathlib
import sys
import collections
from dreamer import tools
from dreamer import exploration as expl
import datetime
import uuid
import json
import gc
import pynvml
def print_gpu_memory():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        if i == 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            # 有些版本返回 bytes，有些返回 str
            if isinstance(name, bytes):
                name = name.decode()
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU {i}: {name}")
            print(f"  Total memory: {info.total / 1024**2:.2f} MB")
            print(f"  Used memory : {info.used  / 1024**2:.2f} MB")
            print(f"  Free memory : {info.free  / 1024**2:.2f} MB\n")
    pynvml.nvmlShutdown()
def average_gradients(model):
    world_size = torch.distributed.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            # 所有进程的梯度相加
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            # 求平均
            param.grad.data /= world_size

def _get_world_info():
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
    return rank, world_size

def _all_reduce_tensor(tensor):
    """in-place all-reduce (SUM) if distributed, otherwise return same tensor"""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def print_memory_usage():
    total_memory = 0
    tensor_list = []

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    mem_size = obj.element_size() * obj.nelement() / 1024 ** 2  # MB
                    total_memory += mem_size
                    tensor_list.append((type(obj), obj.size(), mem_size))

        except Exception as e:
            pass  # 防止一些非张量对象报错

    # 按照显存占用从大到小排序
    tensor_list.sort(key=lambda x: x[2], reverse=True)

    print(f"Total CUDA memory allocated: {total_memory:.2f} MB")
    for t in tensor_list[:20]:
        print(f"Tensor Type: {t[0]}, Size: {t[1]}, Memory: {t[2]:.2f} MB")

class RWMRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 history_length=5,
                 **kwargs
                 ):
        
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.depth_predictor_cfg = train_cfg["depth_predictor"]
        self.device = device
        self.env = env
        self.history_length = history_length
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        if self.env.include_history_steps is not None:
            num_actor_obs = self.env.num_obs * self.env.include_history_steps
        else:
            num_actor_obs = self.env.num_obs
        #if dist.get_rank() == 0:
        #    print("before build world model gpu memory")
        #    print_gpu_memory()
        # build world model
        self._build_world_model()
        #if dist.get_rank() == 0:
        #    print("after build world model gpu memory")
        #    print_gpu_memory()
        # build depth predictor
        self.depth_predictor = DepthPredictor().to(self._world_model.device)
        self.depth_predictor_opt = optim.Adam(self.depth_predictor.parameters(), lr=self.depth_predictor_cfg["lr"],
                                              weight_decay=self.depth_predictor_cfg["weight_decay"])

        self.history_dim = history_length * (self.env.num_obs - self.env.privileged_dim - self.env.height_dim - 3) #exclude command
        # print("history_dim: ", self.env.num_obs - self.env.privileged_dim - self.env.height_dim - 3)
        phase_model_enabled = getattr(self.wm_config, 'phase_model', False)
        actor_critic = ActorCriticRWM(num_actor_obs=num_actor_obs,
                                          num_critic_obs=num_critic_obs,
                                          num_actions=self.env.num_actions,
                                          height_dim=self.env.height_dim,
                                          privileged_dim=self.env.privileged_dim,
                                          history_dim=self.history_dim,
                                          wm_feature_dim=self.wm_feature_dim,
                                          phase_model=phase_model_enabled,
                                          prop_dim=self.env.cfg.env.prop_dim,
                                          **self.policy_cfg).to(self.device)
        #print("self.cfg: ", self.cfg)
        if self.cfg["ddp"]:
            # ---------- Distributed setup ----------
            self.is_distributed = dist.is_available() and dist.is_initialized()
            # 如果 dist 尚未初始化，也可以从环境变量读取 LOCAL_RANK（trainMBRL.py 已 init）
            if not self.is_distributed and "LOCAL_RANK" in os.environ:
                try:
                    # 只尝试读取，确保 torchrun 已经初始化过 dist 在外部脚本中
                    _ = dist.get_rank()
                    self.is_distributed = True
                except:
                    self.is_distributed = False

            if self.is_distributed:
                try:
                    self.rank = dist.get_rank()
                    self.world_size = dist.get_world_size()
                except:
                    # 兼容性：若 dist 尚未 init，退回到 env 变量
                    self.rank = int(os.environ.get("RANK", "0"))
                    self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            else:
                self.rank = 0
                self.world_size = 1

            # Wrap actor_critic with DDP for gradient sync (per-process single-GPU)
            if self.is_distributed:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                # Ensure model is on the right cuda device already
                actor_critic = DDP(actor_critic, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
                #self._world_model = DDP(self._world_model, device_ids=[local_rank], output_device=local_rank)
                # NOTE: after this, the wrapped model is actor_critic (DDP object)
            self.ac_for_call = getattr(actor_critic, "module", actor_critic)
            # ---------- end distributed setup ----------
        # Hexapod ban amp
        if self.env.cfg.cpg.use_amp:
            amp_data = AMPLoader(
                device, time_between_frames=self.env.dt, preload_transitions=True,
                num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'],
                motion_files=self.cfg["amp_motion_files"])
            amp_normalizer = Normalizer(amp_data.observation_dim)
            print("amp_obs dim: ", amp_data.observation_dim)
            discriminator = AMPDiscriminator(
                amp_data.observation_dim * 2,
                train_cfg['runner']['amp_reward_coef'],
                train_cfg['runner']['amp_discr_hidden_dims'], device,
                train_cfg['runner']['amp_task_reward_lerp']).to(self.device)
        
        # # build estimator
        # self.estimator_cfg = train_cfg["estimator"]
        # self.depth_encoder_cfg = train_cfg["depth_encoder"]
        # estimator = Estimator(input_dim=env.cfg.env.n_proprio, output_dim=env.cfg.env.n_priv, hidden_dims=self.estimator_cfg["hidden_dims"]).to(self.device)
        # # Depth encoder
        # self.if_depth = self.depth_encoder_cfg["if_depth"]
        # if self.if_depth:
        #     depth_backbone = DepthOnlyFCBackbone58x87(env.cfg.env.n_proprio, 
        #                                             self.policy_cfg["scan_encoder_dims"][-1], 
        #                                             self.depth_encoder_cfg["hidden_dims"],
        #                                             )
        #     depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg).to(self.device)
        #     depth_actor = deepcopy(actor_critic.actor)
        # else:
        #     depth_encoder = None
        #     depth_actor = None
        
        # self.discr: AMPDiscriminator = AMPDiscriminator()
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        min_std = (
                torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
                (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])))
        if phase_model_enabled:
            # Append a default min_std for raw_omega
            # Use the same normalized min std if it's a scalar, otherwise use 0.1
            m_std = self.cfg["min_normalized_std"]
            if isinstance(m_std, (int, float)):
                omega_min_std = torch.tensor([m_std], device=self.device)
            else:
                omega_min_std = torch.tensor([0.1], device=self.device)
            min_std = torch.cat([min_std, omega_min_std])
        # Hexapod ban amp
        if self.env.cfg.cpg.use_amp:
            self.alg: PPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device,
                                    min_std=min_std, **self.alg_cfg)
        # self.alg: PPO = alg_class(actor_critic, estimator, self.estimator_cfg, depth_encoder, self.depth_encoder_cfg, depth_actor, device=self.device, **self.alg_cfg)
        elif self.cfg["ddp"]:
            self.alg = PPO_DDP(
                actor_critic=actor_critic,
                device=self.device,
                min_std = min_std,
                **self.alg_cfg
            )
        else:
            self.alg = PPO(
                actor_critic=actor_critic,
                device=self.device,
                min_std = min_std,
                **self.alg_cfg
            )
        #print("self.alg device: ", self.alg.device)
        if self.wm_config.use_imagination:
            self._task_behavior = ImagBehavior(self.wm_config, self._world_model, self.alg.actor_critic)
            self._task_behavior = self._task_behavior.to(self.device)
            self._metrics = {}
            reward = lambda f, s, a: self._world_model.heads["reward"](f).mean()
            self._expl_behavior = dict(
                greedy = lambda: self._task_behavior,
                plan2explore = lambda: expl.Plan2Explore(self.wm_config, self._world_model, reward, self.alg.actor_critic)
            )[self.wm_config.expl_behavior]().to(self.device)
            self.loss_dict = {}
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        # Determine actual action dim for storage
        storage_action_dim = self.env.num_actions + 1 if phase_model_enabled else self.env.num_actions

        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs],
                              [self.env.num_privileged_obs], [storage_action_dim], self.history_dim, self.wm_feature_dim)
        #self._task_behavior.init_storage(self.wm_config.imag_start_batch*self.wm_config.batch_size, self.wm_config.imag_horizon, [num_actor_obs],
        #                      [self.env.num_privileged_obs], [self.env.num_actions], self.history_dim, self.wm_feature_dim)
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()


    def _build_world_model(self):
        # world model
        print('Begin construct world model')
        configs = yaml.safe_load(
            (pathlib.Path(sys.argv[0]).parent.parent.parent / "dreamer/configs.yaml").read_text()
        )

        def recursive_update(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    recursive_update(base[key], value)
                else:
                    base[key] = value

        name_list = ["defaults"]
        defaults = {}
        for name in name_list:
            recursive_update(defaults, configs[name])
        parser = argparse.ArgumentParser()
        parser.add_argument("--headless", action="store_true", default=False)
        parser.add_argument("--sim_device", default='cuda:0')
        parser.add_argument("--wm_device", default='None')
        parser.add_argument("--terrain", default='climb')
        # If True: train world model purely with replay actions from wm_dataset (open-loop).
        # If False (default): use act_func to generate actions from current policy during WM training (closed-loop).
        parser.add_argument("--wm_use_replay_action", action="store_true", default=False)
        for key, value in sorted(defaults.items(), key=lambda x: x[0]):
            arg_type = tools.args_type(value)
            parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
        self.wm_config = parser.parse_args()
        # allow world model and rl env on different device
        if not self.cfg["ddp"]:
            if (self.wm_config.wm_device != 'None'):
                self.wm_config.device = self.wm_config.wm_device
        
        if getattr(self.wm_config, 'phase_model', False):
            self.wm_config.num_actions = (self.env.num_actions + 1) * self.env.cfg.depth.update_interval
        else:
            self.wm_config.num_actions = self.env.num_actions * self.env.cfg.depth.update_interval
            
        prop_dim = self.env.num_obs - self.env.privileged_dim - self.env.height_dim - self.env.num_actions
        image_shape = self.env.cfg.depth.resized + (1,)
        obs_shape = {'prop': (prop_dim,), 'image': image_shape,}
        pri_obs_shape = {'forward_height_map': (self.env.cfg.env.forward_height_dim,),
                         'height_map': (self.env.cfg.env.height_dim,),
                         'privileged_obs': (self.env.cfg.env.privileged_dim,),
                         'prop': (prop_dim,),
                         'image': image_shape,}
        if self.cfg["ddp"]:
            use_device = None
            if self.cfg.get("ddp", False) and dist.is_available() and dist.is_initialized():
                # in DDP mode, each process should create model on its own local_rank GPU
                # assume trainMBRL has already set CUDA device and self.device correctly
                try:
                    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                except:
                    local_rank = 0
                use_device = torch.device(f"cuda:{local_rank}")
            else:
                # non-distributed: honor wm_config.wm_device if set, otherwise use self.device
                if getattr(self.wm_config, "wm_device", "None") != 'None':
                    use_device = torch.device(self.wm_config.wm_device)
                else:
                    use_device = torch.device(self.device if isinstance(self.device, str) else str(self.device))

            print(f"[RWMRunner] Building world model on device: {use_device}")
        if self.cfg["ddp"]:
            if self.wm_config.decode_pri_obs:
                self._world_model = WorldModelRWM(self.wm_config, obs_shape, use_camera=self.env.cfg.depth.use_camera, pri_obs_shape = pri_obs_shape, device = use_device)
                self._world_model = self._world_model.to(use_device)
            else:
                self._world_model = WorldModelRWM(self.wm_config, obs_shape, use_camera=self.env.cfg.depth.use_camera, device = use_device)
                self._world_model = self._world_model.to(use_device)
        else:
            if self.wm_config.decode_pri_obs:
                self._world_model = WorldModelRWM(self.wm_config, obs_shape, use_camera=self.env.cfg.depth.use_camera, pri_obs_shape = pri_obs_shape)
            else:
                self._world_model = WorldModelRWM(self.wm_config, obs_shape, use_camera=self.env.cfg.depth.use_camera)
            self._world_model = self._world_model.to(self._world_model.device)
        #self._world_model = WorldModel(self.wm_config, obs_shape, use_camera=self.env.cfg.depth.use_camera, pri_obs_shape = pri_obs_shape)
        if self.cfg["ddp"]:
            if self.cfg.get("ddp", False) and dist.is_available() and dist.is_initialized():
                # Use device_ids=[local_rank] so DDP binds to the right GPU in this process.
                try:
                    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                except:
                    local_rank = 0
                # Clear GPU cache before creating DDP to prevent memory allocation errors
                # This helps avoid "Failed to CUDA host alloc" errors during DDP initialization
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device=local_rank)
                # Ensure model is on the correct device
                if next(self._world_model.parameters()).device != torch.device(f"cuda:{local_rank}"):
                    self._world_model = self._world_model.to(f"cuda:{local_rank}")
                wm = DDP(self._world_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
                # store the wrapped ddp model
                self._world_model = wm
                # store convenience pointer to underlying module for attribute access
                self._world_model_module = wm.module
                self._world_model = self._get_world_model()
            else:
                # not distributed, but ddp flag is set - just keep module as is
                # self._world_model was already created above
                self._world_model_module = self._world_model
            underlying = self._world_model_module
            self.wm_feature_dim = getattr(self.wm_config, 'dyn_deter', None) or getattr(underlying, 'deter_dim', None) or self.wm_config.dyn_deter
        else:
            self.wm_feature_dim = self.wm_config.dyn_deter #+ self.wm_config.dyn_stoch * self.wm_config.dyn_discrete
        print('Finish construct world model')
        #self.wm_feature_dim = self.wm_config.dyn_deter #+ self.wm_config.dyn_stoch * self.wm_config.dyn_discrete
        
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.cfg["ddp"]:
            if self.log_dir is not None and self.writer is None and self.rank == 0:
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)                
        else: 
            if self.log_dir is not None and self.writer is None:
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        #if dist.get_rank() == 0:
            #print("before get_observations")
            #print_memory_usage()
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        # Hexapod ban amp
        if self.env.cfg.cpg.use_amp:
            amp_obs = self.env.get_amp_observations() 
        critic_obs = privileged_obs if privileged_obs is not None else obs
        # Hexapod ban amp
        if self.env.cfg.cpg.use_amp:
            obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        else:
            obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        if self.env.cfg.cpg.use_amp:
            self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        # process trajectory history
        self.trajectory_history = torch.zeros(size=(self.env.num_envs, self.history_length, self.env.num_obs -
                                                    self.env.privileged_dim - self.env.height_dim - 3), # exclude commands
                                              device=self.device)
        # print("trajectory_history", self.trajectory_history.shape)
        # print("num_obs: ", self.env.num_obs)
        # print("privileged_dim: ", self.env.privileged_dim)
        # print("height_dim: ", self.env.height_dim)
        obs_without_command = torch.concat((obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                                            obs[:, self.env.privileged_dim + 9:self.env.num_obs-self.env.height_dim]), dim=1) #without command and hieght map and privileged_obs
        # print("obs_without_command 1: ", obs[:, self.env.privileged_dim:self.env.privileged_dim + 6].shape)
        # print("obs_without_command 2: ", obs[:, self.env.privileged_dim + 9:-self.env.height_dim].shape)
        # print("obs_without_command 3: ", obs.shape)
        self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)),
                                               dim=1)

        # init world model input
        sum_wm_dataset_size = 0
        wm_latent = wm_action = None
        wm_is_first = torch.ones(self.env.num_envs, device=self._world_model.device)
        wm_obs = {
            "prop": obs[:, self.env.privileged_dim: self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device), # TODO need to excluede lin vel
            "is_first": wm_is_first,
            #"privileged_obs": critic_obs[:, 0: self.env.privileged_dim].to(self._world_model.device),
            #"height_map":obs[:, self.env.privileged_dim + self.env.cfg.env.prop_dim + self.env.cfg.env.action_dim : self.env.privileged_dim + self.env.cfg.env.prop_dim + self.env.cfg.env.action_dim + self.env.cfg.env.height_dim].to(self._world_model.device),
        }
        if getattr(self.wm_config, "enable_foothold_prediction", False):
            wm_obs["foot_contact"] = (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.2).float().to(self._world_model.device)
        if self.wm_config.decode_pri_obs:
            wm_obs["privileged_obs"] = critic_obs[:, 0: self.env.privileged_dim].to(self._world_model.device)
            wm_obs["height_map"] = critic_obs[:, self.env.privileged_dim + self.env.cfg.env.prop_dim + self.env.cfg.env.action_dim : self.env.privileged_dim + self.env.cfg.env.prop_dim + self.env.cfg.env.action_dim + self.env.cfg.env.height_dim].to(self._world_model.device) 
        if(self.env.cfg.depth.use_camera):
            wm_obs["image"] = torch.zeros(((self.env.num_envs,) + self.env.cfg.depth.resized + (1,)), device=self._world_model.device)
            #wm_obs["forward_height_map"] = self.env.get_forward_map().to(self._world_model.device)
        # print("wm_obs: ", wm_obs["prop"].shape)
        wm_metrics = None
        phase_model = getattr(self.wm_config, 'phase_model', False)
        omega_max = getattr(self.wm_config, 'omega_max', 12.566370614359172)
        phi = torch.zeros(self.env.num_envs, 1, device=self.device)

        wm_action_dim = self.env.num_actions + 1 if phase_model else self.env.num_actions
        self.wm_update_interval = self.env.cfg.depth.update_interval
        wm_action_history = torch.zeros(size=(self.env.num_envs, self.wm_update_interval, wm_action_dim),
                                        device=self._world_model.device)
        wm_reward = torch.zeros(self.env.num_envs, device=self._world_model.device)
        wm_feature = torch.zeros((self.env.num_envs, self.wm_feature_dim))

        self.init_wm_dataset()
        # 在learn()方法的循环前添加初始化
        mean_world_model_loss = 0.0
        mean_wm_image_loss = 0.0
        mean_wm_privileged_obs_loss = 0.0
        mean_wm_prop_loss = 0.0
        mean_wm_reward_loss = 0.0

        for it in range(self.current_learning_iteration, tot_iter):
            if (self.env.cfg.rewards.reward_curriculum):
                self.env.update_reward_curriculum(it)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if (self.env.global_counter % self.wm_update_interval == 0):
                        # world model obs step
                        wm_embed = self._world_model.encoder(wm_obs)
                        if phase_model:
                            wm_embed = torch.concat((wm_embed, phi.to(self._world_model.device)), dim=-1)
                            if wm_action is not None:
                                wm_action = torch.concat((wm_action, phi.to(self._world_model.device)), dim=-1)
                        #if dist.get_rank() == 0 and i == 1:
                            #print("after wm_embed")
                            #print_memory_usage()
                        #print("wm_embed in rwm_runner: ", wm_embed.device)
                        #print("wm_action in rwm_runner: ", wm_action.device)
                        #print("wm_obs['is_first'] in rwm_runner: ", wm_obs['is_first'].device)
                        #print("world model device: ", self._world_model.device)
                        wm_latent, _ = self._world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed,
                                                                           wm_obs["is_first"])
                        wm_feature = self._world_model.dynamics.get_deter_feat(wm_latent)
                        wm_is_first[:] = 0

                    history = self.trajectory_history.flatten(1).to(self.device)
                    # Hexapod ban amp
                    if self.env.cfg.cpg.use_amp:
                        actions = self.alg.act(obs, critic_obs, amp_obs, history, wm_feature.to(self.env.device), phi=phi if phase_model else None)
                    # print("critic_obs_shpae: ", critic_obs.shape)
                    else:
                        #print("wm_feature in rwm_runner: ", wm_feature.device)
                        #print("obs in rwm_runner: ", obs.device)
                        #print("critic_obs in rwm_runner: ", critic_obs.device)
                        #print("history in rwm_runner: ", history.device)
                        #print("self.env.device: ", self.env.device)
                        actions = self.alg.act(obs, critic_obs, history, wm_feature.to(self.env.device), phi=phi if phase_model else None)
                        #if dist.get_rank() == 0 and i == 1:
                        #    print("after act")
                        #    print_memory_usage()
                    
                    if phase_model:
                        # Update phi
                        raw_omega = actions[:, -1:]
                        omega = omega_max + torch.tanh(raw_omega)
                        # Use self.env.dt or self.env.cfg.sim.dt depending on structure
                        dt = getattr(self.env, 'dt', 0.02)
                        phi = phi + omega * dt
                        # Slice actions for environment
                        env_actions = actions[:, :18]
                    else:
                        env_actions = actions

                    if self.env.cfg.cpg.use_amp:
                        obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(
                             env_actions)
                    else:
                        obs, privileged_obs, rewards, dones, infos, reset_env_ids = self.env.step(
                        env_actions)
                        #if dist.get_rank() == 0 and i == 1:
                        #    print("after step")
                        #    print_memory_usage()
                    # Hexapod ban amp
                    if self.env.cfg.cpg.use_amp:
                        next_amp_obs = self.env.get_amp_observations()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    # Hexapod ban amp
                    if self.env.cfg.cpg.use_amp:
                        obs, critic_obs, next_amp_obs, rewards, dones = obs.to(self.device), critic_obs.to(
                             self.device), next_amp_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    else:
                        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(
                        self.device), rewards.to(self.device), dones.to(self.device)
                    # update world model input
                    if phase_model:
                        wm_actions_to_cat = actions.unsqueeze(1).to(self._world_model.device)
                    else:
                        wm_actions_to_cat = actions.unsqueeze(1).to(self._world_model.device)
                        
                    wm_action_history = torch.concat(
                        (wm_action_history[:, 1:], wm_actions_to_cat), dim=1)
                    wm_obs = {
                        "prop": obs[:, self.env.privileged_dim: self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device),
                        "is_first": wm_is_first,
                        #"privileged_obs": critic_obs[:, 0: self.env.privileged_dim].to(self._world_model.device),
                        # "heght_map": obs[:, self.env.privileged_dim + self.env.cfg.env.prop_dim + self.env.cfg.env.action_dim : self.env.privileged_dim + self.env.cfg.env.prop_dim + self.env.cfg.env.action_dim + self.env.cfg.env.height_dim].to(self._world_model.device),
                    }
                    if getattr(self.wm_config, "enable_foothold_prediction", False):
                        wm_obs["foot_contact"] = (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.2).float().to(self._world_model.device)
                    if self.wm_config.decode_pri_obs:
                        wm_obs["privileged_obs"] = critic_obs[:, 0: self.env.privileged_dim].to(self._world_model.device)
                        wm_obs["height_map"] = critic_obs[:, self.env.privileged_dim + self.env.cfg.env.prop_dim + self.env.cfg.env.action_dim : self.env.privileged_dim + self.env.cfg.env.prop_dim + self.env.cfg.env.action_dim + self.env.cfg.env.height_dim].to(self._world_model.device) 
                    # print("wm_obs: ", wm_obs["prop"].shape)
                    # if (self.env.cfg.depth.use_camera):
                    #     wm_obs["forward_height_map"] = self.env.get_forward_map().to(self._world_model.device)

                    # store the data in buffer into the dataset before reset
                    reset_env_ids = reset_env_ids.cpu().numpy()
                    if (len(reset_env_ids) > 0):
                        for k, v in self.wm_dataset.items():
                            if(k == "image"):
                                for id in reset_env_ids:
                                    idx_in_buffer = np.where(self.env.depth_index == id)[0]
                                    if(len(idx_in_buffer) > 0):
                                        v[idx_in_buffer, :] = self.wm_buffer[k][idx_in_buffer].to(self._world_model.device)
                            else:
                                v[reset_env_ids, :] = self.wm_buffer[k][reset_env_ids].to(self._world_model.device)

                        self.wm_dataset_size[reset_env_ids] = self.wm_buffer_index[reset_env_ids]
                        self.wm_buffer_index[reset_env_ids] = 0
                        sum_wm_dataset_size = np.sum(self.wm_dataset_size)

                        wm_action_history[reset_env_ids, :] = 0
                        wm_is_first[reset_env_ids] = 1
                        if phase_model:
                            phi[reset_env_ids] = 0

                    wm_action = wm_action_history.flatten(1)
                    wm_reward += rewards.to(self._world_model.device)

                    # store current step into buffer
                    if (self.env.global_counter % self.wm_update_interval == 0):
                        if (self.env.cfg.depth.use_camera):
                            forward_heightmap = self.env.get_forward_map().to(self._world_model.device)
                            pred_depth_image = self.depth_predictor(forward_heightmap, wm_obs["prop"])
                            wm_obs["image"] = pred_depth_image
                            self.wm_buffer["forward_height_map"][range(self.env.num_envs), self.wm_buffer_index,:] = forward_heightmap[:].to('cpu')
                            wm_obs["image"][self.env.depth_index] = infos["depth"].unsqueeze(-1).to(self._world_model.device)
                            self.wm_buffer["image"][range(self.env.cfg.depth.camera_num_envs),
                            self.wm_buffer_index[self.env.depth_index], :] = wm_obs["image"][self.env.depth_index].to(
                                'cpu')
                        not_reset_env_ids = (1 - wm_is_first).nonzero(as_tuple=False).flatten().cpu().numpy()
                        if (len(not_reset_env_ids) > 0):
                            for k, v in wm_obs.items():
                                if(k != "is_first" and k != "image"):
                                    self.wm_buffer[k][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = v[not_reset_env_ids].to('cpu')
                            self.wm_buffer["action"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = \
                                wm_action[not_reset_env_ids, :].to('cpu')
                            self.wm_buffer["reward"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids]] = \
                                wm_reward[not_reset_env_ids].to('cpu')
                            self.wm_buffer_index[not_reset_env_ids] += 1
                            # Store history
                            self.wm_buffer["history"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = \
                                self.trajectory_history[not_reset_env_ids].flatten(1).to('cpu')
                            # self.wm_buffer_index[not_reset_env_ids] += 1

                        wm_reward[:] = 0
                        
                    # Hexapod ban amp
                    # # Account for terminal states.
                    if self.env.cfg.cpg.use_amp:
                        next_amp_obs_with_term = torch.clone(next_amp_obs)
                        next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                        #rewards = self.alg.discriminator.predict_amp_reward(
                        #     amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)[0]
                        #print("amp_obs: ", amp_obs.shape)
                        #print("next_amp_obs_with_term: ", next_amp_obs_with_term.shape)
                        task_reward = rewards
                        amp_reward = self.alg.discriminator.predict_amp_reward(
                            amp_obs, next_amp_obs_with_term, task_reward, normalizer=self.alg.amp_normalizer)[0]

                        rewards = task_reward + 0.3 * amp_reward
                        amp_obs = torch.clone(next_amp_obs)
                        self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
                    else:
                        rewards = rewards.to(self.device)
                        self.alg.process_env_step(rewards, dones, infos)
                    
                    # process trajectory history
                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.trajectory_history[env_ids] = 0
                    obs_without_command = torch.concat((obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                                                        obs[:, self.env.privileged_dim + 9:self.env.num_obs-self.env.height_dim]),
                                                       dim=1)
                    self.trajectory_history = torch.concat(
                        (self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, wm_feature.to(self.env.device), phi=phi if phase_model else None)
            if self.env.cfg.cpg.use_amp:
                mean_value_loss, mean_surrogate_loss, mean_vel_predict_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            else:
                mean_value_loss, mean_surrogate_loss, mean_vel_predict_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            #if self.log_dir is not None:
            #    self.log(locals())
            #if it % self.save_interval == 0:
            #    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            #ep_infos.clear()

            start_time = time.time()
            if (sum_wm_dataset_size > self.wm_config.train_start_steps):

                if(it % self.depth_predictor_cfg["training_interval"] == 0 and self.env.cfg.depth.use_camera):
                # Train Depth Predictor
                    depth_mse_loss = self.train_depth_predictor()
                    #if self.writer is not None:
                    self.writer.add_scalar('DepthPredictor/loss', depth_mse_loss, it)
                #if dist.get_rank() == 0:
                #    print("before train_world_model")
                #    print("allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
                #    print("reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")
                # Train World Model
                wm_metrics = self.train_world_model(it)
                #if dist.get_rank() == 0:
                #    print("after train_world_model")
                #    print("allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
                #    print("reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")
                #if dist.get_rank() == 0:
                    #print("after train_world_model")
                    #print_memory_usage()
                world_model_loss = wm_metrics["model_loss"]
                mean_world_model_loss = np.mean(world_model_loss)
                if (self.env.cfg.depth.use_camera):
                    wm_image_loss = wm_metrics["image_loss"]
                    mean_wm_image_loss = np.mean(wm_image_loss)
                wm_reward_loss = wm_metrics["reward_loss"]
                mean_wm_reward_loss = np.mean(wm_reward_loss)
                if (self.wm_config.decode_pri_obs):
                    wm_privileged_obs_loss = wm_metrics["privileged_obs_loss"]
                    mean_wm_privileged_obs_loss = np.mean(wm_privileged_obs_loss)
                wm_prop_loss = wm_metrics["prop_loss"]
                mean_wm_prop_loss = np.mean(wm_prop_loss)
                for name, values in wm_metrics.items():
                    if self.writer is not None:
                        self.writer.add_scalar('World_model/' + name, float(np.mean(values)), it)
                #if self.cfg["ddp"] and dist.get_rank() == 0:
                #    print_memory_usage()
            train_world_model_time = time.time() - start_time
            print('training world model time:', train_world_model_time)
            #if dist.get_rank() == 0:
            #    print("allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
            #    print("reserved :", torch.cuda.memory_reserved() / 1024**2, "MB")

            if self.log_dir is not None:
                if self.cfg["ddp"]:
                    self.log_ddp(locals())
                else:
                    self.log(locals())
                # self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
            
            # copy the config file
            if(it == 0):
                # os.system("cp ./legged_gym/envs/a1/a1_amp_config.py " + self.log_dir + "/")
                os.system("cp ./gym/envs/hexapodMBRL_config.py " + self.log_dir + "/")
            # for key, value in wm_obs.items():
            #     print(f"wm_obs[{key}].shape: {value.shape}")
            # for key, value in self.wm_buffer.items():
            #     print(f"wm_buffer[{key}].shape: {value.shape}")
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
    
    def init_wm_dataset(self):
        phase_model = getattr(self.wm_config, 'phase_model', False)
        wm_action_dim = self.env.num_actions + 1 if phase_model else self.env.num_actions
        
        self.wm_dataset = {
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, self.env.cfg.env.prop_dim),
                                device=self._world_model.device),
            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   wm_action_dim * self.wm_update_interval), device=self._world_model.device),
            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,),
                                  device=self._world_model.device),
            "history": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.history_length * (self.env.num_obs - self.env.privileged_dim - self.env.height_dim - 3)),
                                  device=self._world_model.device),
        }
        if getattr(self.wm_config, "enable_foothold_prediction", False):
            self.wm_dataset["foot_contact"] = torch.zeros(
                (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, len(self.env.feet_indices)),
                device=self._world_model.device,
            )
        if(self.env.cfg.depth.use_camera):
            self.wm_dataset["image"] = torch.zeros(((self.env.cfg.depth.camera_num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                               + self.env.cfg.depth.resized + (1,)), device=self._world_model.device)
            self.wm_dataset["forward_height_map"] = torch.zeros(
                (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                 self.env.cfg.env.forward_height_dim), device=self._world_model.device)
        if self.wm_config.decode_pri_obs:
            self.wm_dataset["privileged_obs"] = torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                                            self.env.cfg.env.privileged_dim), device=self._world_model.device)
            self.wm_dataset["height_map"] = torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                                            self.env.cfg.env.height_dim), device=self._world_model.device)
        self.wm_dataset_size = np.zeros(self.env.num_envs)

        self.wm_buffer = {
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, self.env.cfg.env.prop_dim),
                                device='cpu'),
            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   wm_action_dim * self.wm_update_interval), device='cpu'),
            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,),
                                  device='cpu'),
            "history": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.history_length * (self.env.num_obs - self.env.privileged_dim - self.env.height_dim - 3)),
                                  device='cpu'),
                                  
        }
        if getattr(self.wm_config, "enable_foothold_prediction", False):
            self.wm_buffer["foot_contact"] = torch.zeros(
                (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, len(self.env.feet_indices)),
                device='cpu',
            )
        if(self.env.cfg.depth.use_camera):
            self.wm_buffer["image"] = torch.zeros(((self.env.cfg.depth.camera_num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                               + self.env.cfg.depth.resized + (1,)), device='cpu')
            self.wm_buffer["forward_height_map"] = torch.zeros(
                (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                 self.env.cfg.env.forward_height_dim), device='cpu')
        if self.wm_config.decode_pri_obs:
            self.wm_buffer["privileged_obs"] = torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                                            self.env.cfg.env.privileged_dim), device='cpu')
            self.wm_buffer["height_map"] = torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                                            self.env.cfg.env.height_dim), device='cpu')
        self.wm_buffer_index = np.zeros(self.env.num_envs)

    def train_depth_predictor(self):
        total_mse_loss = 0
        for _ in range(self.depth_predictor_cfg["training_iters"]):
            batch_idx = np.random.choice(self.env.depth_index, self.depth_predictor_cfg["batch_size"],
                                         replace=True)
            time_index = [np.random.randint(0, self.wm_dataset_size[idx] + 1) for idx in batch_idx]
            forward_heightmap = self.wm_dataset["forward_height_map"][batch_idx, time_index]
            prop = self.wm_dataset["prop"][batch_idx, time_index]
            depth_image = self.wm_dataset["image"][self.env.depth_index_inverse[batch_idx], time_index]

            predict_depth_image = self.depth_predictor(forward_heightmap, prop)
            depth_predict_loss = (depth_image - predict_depth_image).pow(2).mean() * self.depth_predictor_cfg[
                "loss_scale"]
            # Gradient step
            self.depth_predictor_opt.zero_grad()
            depth_predict_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_predictor.parameters(), 1)
            if dist.is_initialized():
                average_gradients(self.depth_predictor)  # 调用你提供的第二个版本
            self.depth_predictor_opt.step()
            total_mse_loss += depth_predict_loss.detach() / self.depth_predictor_cfg["loss_scale"]
        return float(total_mse_loss / self.depth_predictor_cfg["training_iters"])

    def train_world_model(self, it):
        metrics = {}
        wm_metrics = {}
        mets = {}
        post = None  # 初始化 post 变量
        context = None  # 初始化 context 变量
        actual_train_steps = 0  # 实际执行的训练步数

        for i in range(self.wm_config.train_steps_per_iter):
            p = self.wm_dataset_size / np.sum(self.wm_dataset_size)
            batch_idx = np.random.choice(range(self.env.num_envs), self.wm_config.batch_size, replace=True,
                                         p=p)
            batch_length = min(int(self.wm_dataset_size[batch_idx].min()), self.wm_config.batch_length)
            if (batch_length <= self._world_model.k_steps):
                print(f"Step {i}: Skipping due to insufficient data. batch_length: {batch_length}, k_steps: {self._world_model.k_steps}")
                continue  # an error occur about the predict loss if batch_length < 1
            batch_end_idx = [np.random.randint(batch_length, self.wm_dataset_size[idx] + 1) for idx in batch_idx]
            batch_data = {}
            for k, v in self.wm_dataset.items():
                # print("wm_dataset key: ", k)
                if (k == "forward_height_map"):
                # if ((k == "forward_height_map") and (not self.env.cfg.depth.use_camera)):
                #if (k == "forward_height_map" or k == "privileged_obs"):
                    continue
                value = []
                for idx, end_idx in zip(batch_idx, batch_end_idx):
                    if (k == "image"):
                        idx_in_buffer = np.where(self.env.depth_index == idx)[0]
                        if (len(idx_in_buffer) == 0):
                            # not in the buffer, use the predicted ones
                            tmp_forward_heightmap = self.wm_dataset["forward_height_map"][idx,
                                                    end_idx - batch_length: end_idx]
                            tmp_prop = self.wm_dataset["prop"][idx, end_idx - batch_length: end_idx]
                            pred_depth_image = self.depth_predictor(tmp_forward_heightmap, tmp_prop)
                            value.append(pred_depth_image)
                        else:
                            value.append(v[idx_in_buffer[0], end_idx - batch_length: end_idx])
                    else:
                        value.append(v[idx, end_idx - batch_length: end_idx])
                value = torch.stack(value)
                batch_data[k] = value
            is_first = torch.zeros((self.wm_config.batch_size, batch_length))
            is_first[:, 0] = 1
            batch_data["is_first"] = is_first
            # print("batch_data key: ", batch_data.keys())
            if self.cfg["ddp"]:
                # -------- get world-model and its device (你已有逻辑) ----------
                if isinstance(self._world_model, torch.nn.parallel.DistributedDataParallel):
                    #wm_model = self._world_model.module
                    wm_model = self._world_model
                    wm_device = next(self._world_model.parameters()).device
                    #print("train wm_device: ", wm_device)
                else:
                    wm_model = self._world_model
                    wm_device = next(wm_model.parameters()).device
                    #print("train wm_device2: ", wm_device)   
                #print("wm_device: ", wm_device)
                # 将 batch_data 全部转到 world-model 的 device
                #for k, v in batch_data.items():
                #    if isinstance(v, torch.Tensor):
                #        batch_data[k] = v.to(wm_device)
                #print("world model device: ", wm_device)
                # -------- actor_critic module & device ----------
                # some times self.alg.actor_critic may itself be DDP; 取出 module（如果有）
                if isinstance(self.alg.actor_critic, torch.nn.parallel.DistributedDataParallel):
                    #ac_module = self.alg.actor_critic.module
                    ac_module = self.alg.actor_critic
                    #print("ac_module: ", ac_module)
                else:
                    ac_module = self.alg.actor_critic
                    #print("ac_module2: ", ac_module)
                #ac_device = next(ac_module.parameters()).device
                ac_device = wm_device
                ac_module = ac_module.to(ac_device)
                
                # -------- define wrapper that ensures inputs/outputs are on correct devices ----------
                def act_wrapper(prop, history, *args, **kwargs):
                    # move inputs to actor device
                    prop_ac = prop.to(ac_device) if isinstance(prop, torch.Tensor) else prop
                    history_ac = history.to(ac_device) if isinstance(history, torch.Tensor) else history
                    #prop_ac = prop.to(wm_device) if isinstance(prop, torch.Tensor) else prop
                    #history_ac = history.to(wm_device) if isinstance(history, torch.Tensor) else history
                    # choose underlying module (local module if DDP)
                    ac_local = ac_module.module if isinstance(ac_module, torch.nn.parallel.DistributedDataParallel) else ac_module
                    with torch.no_grad():
                        #prop_ac = prop_ac.detach()
                        #proc_history = history_ac.detach()
                        action_ac = ac_local.act(prop_ac, history_ac, *args, **kwargs)
                    #print("ac_device: ", ac_device)
                    #print("wm_device: ", wm_device)
                    #print("history_ac: ", history_ac.device)
                    # call actor (actor returns actions on actor device)
                    #action_ac = ac_module.act(prop_ac, history_ac, *args, **kwargs)
                    
                    # 如果 actor_critic 是 DDP，调用它的 module.act
                    #if isinstance(ac_module, torch.nn.parallel.DistributedDataParallel):
                    #    action_ac = ac_module.module.act(prop_ac, history_ac, *args, **kwargs)
                    #else:
                    #    action_ac = ac_module.act(prop_ac, history_ac, *args, **kwargs)

                    # move action back to world-model device
                    if isinstance(action_ac, torch.Tensor):
                        #return action_ac.to(wm_device)
                        return action_ac.detach().to(wm_device)
                    else:
                        # 如果 actor 返回不是 Tensor（很少），尽量处理可迭代结构
                        try:
                            #return torch.as_tensor(action_ac).to(wm_device)
                            return torch.as_tensor(action_ac).detach().to(wm_device)
                        except Exception:
                            return action_ac

                # Call WM train. Optionally disable act_func to use replay actions only.
                if getattr(self.wm_config, "wm_use_replay_action", False):
                    post, context, mets = wm_model._train(batch_data, act_func=None)
                else:
                    post, context, mets = wm_model._train(batch_data, act_func=act_wrapper)
            else:
                if getattr(self.wm_config, "wm_use_replay_action", False):
                    post, context, mets = self._world_model._train(batch_data, act_func=None)
                else:
                    post, context, mets = self._world_model._train(batch_data, act_func=self.alg.actor_critic.act)
            #print("post: ", post.keys())
            #print("post['stoch'].shape: ", post["stoch"].shape)
            actual_train_steps += 1
            
        # 如果没有执行任何训练步骤，记录警告
        if actual_train_steps == 0:
            print(f"Warning: No training steps executed in this iteration. wm_dataset_size: {self.wm_dataset_size}")
            print(f"wm_config.k_steps: {self._world_model.k_steps}")
            print(f"wm_config.batch_length: {self.wm_config.batch_length}")
            
        wm_metrics.update(mets)
        #del batch_data, mets
        #torch.cuda.synchronize()
        #torch.cuda.empty_cache()
        # print key
        print("wm_metrics: ", wm_metrics.keys())
        if self.wm_config.use_imagination and it > 5000:
            # 确保 post 和 context 有值
            print("post: ", post.keys())
            print("post['stoch'].shape: ", post["stoch"].shape)
            print("post['deter'].shape: ", post["deter"].shape)
            print("post['logit'].shape: ", post["logit"].shape)
            if post is None or context is None:
                print("Error: post or context is None, cannot proceed with imagination training")
                return wm_metrics
            if post["stoch"].shape[1] < self.wm_config.imag_start_batch:
                print(f"Warning: post['stoch'].shape[0] < self.wm_config.imag_start_batch, cannot proceed with imagination training")
                return wm_metrics
            elif post["stoch"].shape[1] > self.wm_config.imag_start_batch:
                post["stoch"] = post["stoch"][:, :self.wm_config.imag_start_batch,:,:]
                post["deter"] = post["deter"][:, :self.wm_config.imag_start_batch,:]
                post["logit"] = post["logit"][:, :self.wm_config.imag_start_batch,:,:]
                
            # 构造 reward function: 从 world model 中预测
            reward_fn = lambda f, s, a: self._world_model.heads["reward"](
                self._world_model.dynamics.get_feat(s)
            ).mode()
            #for name, param in self._world_model.heads["reward"].named_parameters():
            #    print(name, param.requires_grad, param.grad is None)

            #print("self._world_model.heads: ", self._world_model.heads.keys())

            # post 是 world model 的 posterior latent state，作为 rollout 的起点
            _, _, _, _,policy_mets = self._task_behavior._train(start=post, objective=reward_fn)
            metrics.update(policy_mets)

            if self.wm_config.expl_behavior != "greedy":
                _, expl_mets = self._expl_behavior.train(
                    start=post, context=context, data=batch_data
                )
                for key, value in expl_mets.items():
                    metrics[f"expl_{key}"] = value

            metrics.update(wm_metrics)
            for name, value in metrics.items():
                if name not in self._metrics:
                    self._metrics[name] = [value]
                else:
                    self._metrics[name].append(value)
            #print("metrics: ", metrics.keys())
            for name, value in metrics.items():
                if name.endswith("loss"):
                    self.loss_dict[name] = value
                if name == "imag_reward":
                    self.loss_dict[name] = value
            print("loss_dict: ", self.loss_dict.keys())
            return metrics
        #if isinstance(post, dict):
        #    post = {k: v.detach() for k, v in post.items() if isinstance(v, torch.Tensor)}
        #if isinstance(context, dict):
        #    context = {k: v.detach() for k, v in context.items() if isinstance(v, torch.Tensor)}
        #del post, context
        #torch.cuda.empty_cache()
        return wm_metrics

    def log_normal(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/vel_predict', locs['mean_vel_predict_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Vel predict loss:':>{pad}} {locs['mean_vel_predict_loss']:.4f}\n"""
                        #   f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                        #   f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                        #   f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                        #   f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def log_ddp(self, locs, width=80, pad=35):
        # ensure we have world_size and rank
        rank, world_size = _get_world_info()

        # update tot_timesteps with global step count
        # one iteration each rank collected self.num_steps_per_env * self.env.num_envs steps
        steps_this_iter_per_rank = int(self.num_steps_per_env * self.env.num_envs)
        steps_this_iter_global = steps_this_iter_per_rank * world_size
        self.tot_timesteps += steps_this_iter_global

        # aggregate times: collection_time and learn_time are per-rank times
        # we'll take the average across ranks (could also take max)
        coll_time = torch.tensor([locs['collection_time']], device=f"cuda:{dist.get_rank()}", dtype=torch.float32)
        learn_time = torch.tensor([locs['learn_time']], device=f"cuda:{dist.get_rank()}", dtype=torch.float32)
        #print("device: ", f"cuda:{dist.get_rank()}")
        if dist.is_available() and dist.is_initialized():
            coll_time = _all_reduce_tensor(coll_time)
            learn_time = _all_reduce_tensor(learn_time)
            coll_time = coll_time / world_size
            learn_time = learn_time / world_size
        collection_time = float(coll_time.item())
        learn_time = float(learn_time.item())

        # iteration_time = average across ranks
        iteration_time = collection_time + learn_time
        self.tot_time += iteration_time

        # Prepare wandb dict
        wandb_dict = {}
        ep_string = ''

        # Aggregate episode buffers (rewbuffer, lenbuffer)
        # local sums and counts
        local_rew_sum = 0.0
        local_rew_count = 0
        local_len_sum = 0.0
        local_len_count = 0

        if len(locs['rewbuffer']) > 0:
            local_rew_sum = float(sum(locs['rewbuffer']))
            local_rew_count = len(locs['rewbuffer'])
        if len(locs['lenbuffer']) > 0:
            local_len_sum = float(sum(locs['lenbuffer']))
            local_len_count = len(locs['lenbuffer'])

        # make tensors and all_reduce sums over ranks
        rew_sum_t = torch.tensor([local_rew_sum], device=f"cuda:{dist.get_rank()}", dtype=torch.float64)
        rew_cnt_t = torch.tensor([local_rew_count], device=f"cuda:{dist.get_rank()}", dtype=torch.float64)
        len_sum_t = torch.tensor([local_len_sum], device=f"cuda:{dist.get_rank()}", dtype=torch.float64)
        len_cnt_t = torch.tensor([local_len_count], device=f"cuda:{dist.get_rank()}", dtype=torch.float64)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(rew_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(rew_cnt_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(len_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(len_cnt_t, op=dist.ReduceOp.SUM)

        global_rew_sum = float(rew_sum_t.item())
        global_rew_cnt = int(rew_cnt_t.item())
        global_len_sum = float(len_sum_t.item())
        global_len_cnt = int(len_cnt_t.item())

        # compute global means safely
        global_mean_reward = None
        global_mean_episode_length = None
        if global_rew_cnt > 0:
            global_mean_reward = global_rew_sum / global_rew_cnt
            wandb_dict['Train/mean_reward'] = global_mean_reward
            wandb_dict['Train/mean_reward/time'] = global_mean_reward
        if global_len_cnt > 0:
            global_mean_episode_length = global_len_sum / global_len_cnt
            wandb_dict['Train/mean_episode_length'] = global_mean_episode_length
            wandb_dict['Train/mean_episode_length/time'] = global_mean_episode_length

        # losses (these are scalars already, but they might be local; we'll average across ranks)
        # Create tensors for the loss scalars and average them across ranks.
        def _avg_scalar_across_ranks(val):
            t = torch.tensor([float(val)], device=f"cuda:{dist.get_rank()}", dtype=torch.float32)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                t = t / world_size
            return float(t.item())

        wandb_dict['Loss/value_function'] = _avg_scalar_across_ranks(locs.get('mean_value_loss', 0.0))
        wandb_dict['Loss/surrogate'] = _avg_scalar_across_ranks(locs.get('mean_surrogate_loss', 0.0))
        wandb_dict['Loss/vel_predict'] = _avg_scalar_across_ranks(locs.get('mean_vel_predict_loss', 0.0))
        wandb_dict['Loss/learning_rate'] = _avg_scalar_across_ranks(self.alg.learning_rate)
        wandb_dict['Loss/world_model_loss'] = _avg_scalar_across_ranks(locs.get('mean_world_model_loss', 0.0))
        wandb_dict['Loss/wm_image_loss'] = _avg_scalar_across_ranks(locs.get('mean_wm_image_loss', 0.0))
        wandb_dict['Loss/wm_privileged_obs_loss'] = _avg_scalar_across_ranks(locs.get('mean_wm_privileged_obs_loss', 0.0))
        wandb_dict['Loss/wm_prop_loss'] = _avg_scalar_across_ranks(locs.get('mean_wm_prop_loss', 0.0))
        wandb_dict['Loss/wm_reward_loss'] = _avg_scalar_across_ranks(locs.get('mean_wm_reward_loss', 0.0))

        # Policy std: may be a tensor on GPU; gather and average across ranks
        try:
            mu_std_local = torch.tensor([float(self.ac_for_call.std.mean().item())], device=f"cuda:{dist.get_rank()}", dtype=torch.float32)
        except Exception:
            mu_std_local = torch.tensor([0.0], device=f"cuda:{dist.get_rank()}", dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(mu_std_local, op=dist.ReduceOp.SUM)
            mu_std_local = mu_std_local / world_size
        wandb_dict['Policy/mean_noise_std'] = float(mu_std_local.item())

        # Perf
        # total_fps based on global steps during this iteration divided by average iteration_time across ranks
        total_steps = steps_this_iter_global
        if iteration_time > 0:
            fps = total_steps / iteration_time
        else:
            fps = 0.0
        wandb_dict['Perf/total_fps'] = float(fps)
        wandb_dict['Perf/collection time'] = collection_time
        wandb_dict['Perf/learning_time'] = learn_time
        wandb_dict['Perf/train_world_model_time'] = locs.get('train_world_model_time', 0.0)

        # If you have extra loss items in self.loss_dict, add them:
        if self.wm_config.use_imagination:
            for name, value in self.loss_dict.items():
                wandb_dict[f'Loss/{name}'] = _avg_scalar_across_ranks(value)

        # Episode info: if ep_infos exists, aggregate keys the same way as above (sum/count -> mean)
        if locs['ep_infos']:
            # create dict of sums & counts per key
            local_key_sums = {}
            local_key_counts = {}
            for ep_info in locs['ep_infos']:
                for k, v in ep_info.items():
                    val = v.item() if isinstance(v, torch.Tensor) else float(v)
                    local_key_sums[k] = local_key_sums.get(k, 0.0) + val
                    local_key_counts[k] = local_key_counts.get(k, 0) + 1
            # aggregate across ranks
            for k in local_key_sums:
                s_t = torch.tensor([local_key_sums[k]], device=f"cuda:{dist.get_rank()}", dtype=torch.float64)
                c_t = torch.tensor([local_key_counts[k]], device=f"cuda:{dist.get_rank()}", dtype=torch.float64)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(s_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(c_t, op=dist.ReduceOp.SUM)
                gs = float(s_t.item())
                gc = int(c_t.item())
                if gc > 0:
                    wandb_dict['Episode_rew/' + k] = gs / gc
                    ep_string += f"""{f'Mean episode {k}:':>{pad}} {gs/gc:.4f}\n"""

        # If rank 0 then log to wandb and writer
        if rank == 0:
            wandb.log(wandb_dict, step=locs['it'])
            if self.writer is not None:
                for k, v in wandb_dict.items():
                    # writer expects scalar floats
                    try:
                        self.writer.add_scalar(k, float(v), locs['it'])
                    except Exception:
                        pass

        # Build log string (use global values for display)
        str_header = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        if global_rew_cnt > 0:
            log_string = (f"""{'#' * width}\n"""
                        f"""{str_header.center(width, ' ')}\n\n"""
                        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                        f"""{'Value function loss:':>{pad}} {wandb_dict['Loss/value_function']:.4f}\n"""
                        f"""{'Surrogate loss:':>{pad}} {wandb_dict['Loss/surrogate']:.4f}\n"""
                        f"""{'Vel predict loss:':>{pad}} {wandb_dict['Loss/vel_predict']:.4f}\n"""
                        f"""{'Mean action noise std:':>{pad}} {wandb_dict['Policy/mean_noise_std']:.2f}\n"""
                        f"""{'Mean reward (total):':>{pad}} {global_mean_reward:.2f}\n"""
                        f"""{'Mean episode length:':>{pad}} {global_mean_episode_length:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                        f"""{str_header.center(width, ' ')}\n\n"""
                        f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"""
                        f"""{'Value function loss:':>{pad}} {wandb_dict['Loss/value_function']:.4f}\n"""
                        f"""{'Surrogate loss:':>{pad}} {wandb_dict['Loss/surrogate']:.4f}\n"""
                        f"""{'Mean action noise std:':>{pad}} {wandb_dict['Policy/mean_noise_std']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                    f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                    f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                    f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                    f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        # only print on rank0 to avoid duplication
        if rank == 0:
            print(log_string)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        if self.cfg["ddp"]:
            mean_std = self.ac_for_call.std.mean()
        else:
            mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        if self.wm_config.stage == "wm":
            locs['mean_value_loss'] = 0.0
            locs['mean_surrogate_loss'] = 0.0
            locs['mean_vel_predict_loss'] = 0.0
            #mean_noise_std = 0.0
        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        # wandb_dict['Loss/estimator'] = locs['mean_estimator_loss']
        wandb_dict['Loss/vel_predict'] = locs['mean_vel_predict_loss']
        # wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        # wandb_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
        # wandb_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']
        # wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        if self.env.cfg.cpg.use_amp:
            wandb_dict['Loss/AMP'] = locs['mean_amp_loss']
            wandb_dict['Loss/AMP_grad'] = locs['mean_grad_pen_loss']
            wandb_dict['Loss/AMP_mean_policy_pred'] = locs['mean_policy_pred']
            wandb_dict['Loss/AMP_mean_expert_pred'] = locs['mean_expert_pred']
        # wandb_dict['Loss/discriminator'] = locs['mean_disc_loss']
        # wandb_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']
        # wandb_dict['Loss/world_model_loss'] = locs['mean_world_model_loss']
        # wandb_dict['Loss/wm_image_loss'] = locs['mean_wm_image_loss']
        # wandb_dict['Loss/wm_privileged_obs_loss'] = locs['mean_wm_privleged_obs_loss']
        # wandb_dict['Loss/wm_prop_loss'] = locs['mean_wm_prop_loss']
        # wandb_dict['Loss/wm_reward_loss'] = locs['mean_wm_reward_loss']
        wandb_dict['Loss/world_model_loss'] = locs.get('mean_world_model_loss', 0.0)
        wandb_dict['Loss/wm_image_loss'] = locs.get('mean_wm_image_loss', 0.0)
        wandb_dict['Loss/wm_privileged_obs_loss'] = locs.get('mean_wm_privileged_obs_loss', 0.0)
        wandb_dict['Loss/wm_prop_loss'] = locs.get('mean_wm_prop_loss', 0.0)
        wandb_dict['Loss/wm_reward_loss'] = locs.get('mean_wm_reward_loss', 0.0)

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        wandb_dict['Perf/train_world_model_time'] = locs['train_world_model_time']
        if self.wm_config.use_imagination:
            for name, value in self.loss_dict.items():
                wandb_dict[f'Loss/{name}'] = value
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            # wandb_dict['Train/mean_reward_explr'] = statistics.mean(locs['rew_explr_buffer'])
            # wandb_dict['Train/mean_reward_task'] = wandb_dict['Train/mean_reward'] - wandb_dict['Train/mean_reward_explr']
            # wandb_dict['Train/mean_reward_entropy'] = statistics.mean(locs['rew_entropy_buffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            wandb_dict['Train/mean_reward/time'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length/time'] = statistics.mean(locs['lenbuffer'])

        if 'frames' in locs:  # 假设 locs 里包含 'frames' (一个Numpy数组列表)
            video_frames = np.array(locs['frames'])  # 形状：(num_frames, height, width, 3)
            video_path = "training_video.mp4"

            # 保存视频
            import imageio
            imageio.mimsave(video_path, video_frames, fps=30)

            # 将视频上传到 wandb
            wandb_dict["Train/video"] = wandb.Video(video_path, fps=30, format="mp4")
        if self.cfg["ddp"]:
            if self.rank == 0:
                wandb.log(wandb_dict, step=locs['it'])
            else:
                wandb.log(wandb_dict, step=locs['it'])
        else:
            wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Vel predict loss:':>{pad}} {locs['mean_vel_predict_loss']:.4f}\n"""
                        #   f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                        #   f"""{'Discriminator accuracy:':>{pad}} {locs['mean_disc_acc']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                        #   f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['rewbuffer']) - statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                        #   f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                        #   f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                        #   f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string) 
            
    
    def save(self, path, infos=None):
        if self.cfg["ddp"]:
            try:
                actor_state = self.alg.actor_critic.module.state_dict()
            except AttributeError:
                actor_state = self.alg.actor_critic.state_dict()
            try:
                world_model_state = self._world_model.module.state_dict()
            except AttributeError:
                world_model_state = self._world_model.state_dict()
        else:
            actor_state = self.alg.actor_critic.state_dict()
            world_model_state = self._world_model.state_dict()
        
        torch.save({
            'model_state_dict': actor_state,
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'world_model_dict': world_model_state,
            'wm_optimizer_state_dict': self._world_model._model_opt._opt.state_dict(),
            'depth_predictor': self.depth_predictor.state_dict(),
            # 'discriminator_state_dict': self.alg.discriminator.state_dict(),
            # 'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True, load_wm_optimizer = False):
        loaded_dict = torch.load(path, map_location=self.device)

        def strip_module(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            return new_state_dict

        # Debug: Check if weights are actually changing
        if hasattr(self.alg.actor_critic, 'actor') and len(self.alg.actor_critic.actor) > 0:
            before_mean = self.alg.actor_critic.actor[0].weight.data.mean().item()
            print(f"[DEBUG] Before load - Actor layer 0 weight mean: {before_mean}")

        if self.cfg["ddp"]:
            # DDP 模式下，先尝试直接加载
            try:
                ac_module = self.alg.actor_critic.module
                result = ac_module.load_state_dict(loaded_dict['model_state_dict'], strict=False)
            except AttributeError:
                ac_module = self.alg.actor_critic
                result = ac_module.load_state_dict(loaded_dict['model_state_dict'], strict=False)
            
            # 如果加载失败（missing keys 太多），尝试 strip module 前缀
            if len(result.missing_keys) > len(result.unexpected_keys) * 0.5:  # 如果 missing 比 unexpected 多很多，说明 key 不匹配
                print(f"[DEBUG] DDP mode - First load failed. Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
                print(f"[DEBUG] Attempting to strip 'module.' prefix and retry...")
                model_dict_stripped = strip_module(loaded_dict['model_state_dict'])
                result = ac_module.load_state_dict(model_dict_stripped, strict=False)
                print(f"[DEBUG] After strip - Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
            
            if len(result.missing_keys) > 0 or len(result.unexpected_keys) > 0:
                print(f"[DEBUG] DDP mode - Final Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
                if len(result.missing_keys) > 0:
                    print(f"[DEBUG] First 5 missing keys: {result.missing_keys[:5]}")
            
            try:
                wm_module = self._world_model.module
                wm_result = wm_module.load_state_dict(loaded_dict['world_model_dict'], strict=False)
            except AttributeError:
                wm_module = self._world_model
                wm_result = wm_module.load_state_dict(loaded_dict['world_model_dict'], strict=False)
            
            # 同样处理 world model
            if len(wm_result.missing_keys) > len(wm_result.unexpected_keys) * 0.5:
                print(f"[DEBUG] World Model - First load failed. Attempting to strip 'module.' prefix...")
                wm_dict_stripped = strip_module(loaded_dict['world_model_dict'])
                wm_result = wm_module.load_state_dict(wm_dict_stripped, strict=False)
                print(f"[DEBUG] World Model after strip - Missing: {len(wm_result.missing_keys)}, Unexpected: {len(wm_result.unexpected_keys)}")
        else:
            # 如果不是 DDP 模式，但 checkpoint 是 DDP 训练的（带 module.），则去除前缀
            model_dict = loaded_dict['model_state_dict']
            
            # Debug: 打印 checkpoint 中的前几个 key
            checkpoint_keys = list(model_dict.keys())
            print(f"[DEBUG] Checkpoint has {len(checkpoint_keys)} keys")
            print(f"[DEBUG] First 5 checkpoint keys: {checkpoint_keys[:5]}")
            
            # 打印当前模型的 key
            current_model_keys = list(self.alg.actor_critic.state_dict().keys())
            print(f"[DEBUG] Current model has {len(current_model_keys)} keys")
            print(f"[DEBUG] First 5 current model keys: {current_model_keys[:5]}")
            
            if any(k.startswith('module.') for k in model_dict.keys()):
                print("[DEBUG] Detected 'module.' prefix in checkpoint. Stripping...")
                model_dict = strip_module(model_dict)
                print(f"[DEBUG] After strip - First 5 keys: {list(model_dict.keys())[:5]}")
            
            result = self.alg.actor_critic.load_state_dict(model_dict, strict=False)
            if len(result.missing_keys) > 0 or len(result.unexpected_keys) > 0:
                print(f"[DEBUG] Non-DDP mode - Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
                if len(result.missing_keys) > 0:
                    print(f"[DEBUG] First 10 missing keys: {result.missing_keys[:10]}")
                if len(result.unexpected_keys) > 0:
                    print(f"[DEBUG] First 10 unexpected keys: {result.unexpected_keys[:10]}")

            wm_dict = loaded_dict['world_model_dict']
            if any(k.startswith('module.') for k in wm_dict.keys()):
                print("[DEBUG] Detected 'module.' prefix in world_model checkpoint. Stripping...")
                wm_dict = strip_module(wm_dict)
            wm_result = self._world_model.load_state_dict(wm_dict, strict=False)
            if len(wm_result.missing_keys) > 0 or len(wm_result.unexpected_keys) > 0:
                print(f"[DEBUG] World Model - Missing keys: {len(wm_result.missing_keys)}, Unexpected keys: {len(wm_result.unexpected_keys)}")
        
        if hasattr(self.alg.actor_critic, 'actor') and len(self.alg.actor_critic.actor) > 0:
            after_mean = self.alg.actor_critic.actor[0].weight.data.mean().item()
            print(f"[DEBUG] After load - Actor layer 0 weight mean: {after_mean}")
            if abs(after_mean - before_mean) < 1e-6:
                print("[WARNING] ⚠️  Weight did NOT change! Model loading likely failed!")
        
        if(load_wm_optimizer):
            self._world_model._model_opt._opt.load_state_dict(loaded_dict['wm_optimizer_state_dict'])
        # self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'], strict=False)
        # self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    

    def learn_world_model_only(self, num_learning_iterations, init_at_random_ep_len=False):
        """第一阶段：只训练世界模型，不训练策略"""
        print("开始世界模型训练阶段...")
        
        # 初始化 writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        obs, critic_obs = obs.to(self.device), privileged_obs.to(self.device) if privileged_obs is not None else obs.to(self.device)
        
        # 禁用策略网络的梯度计算
        for param in self.alg.actor_critic.parameters():
            param.requires_grad = False

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        # process trajectory history
        self.trajectory_history = torch.zeros(size=(self.env.num_envs, self.history_length, self.env.num_obs -
                                                    self.env.privileged_dim - self.env.height_dim - 3), # exclude commands
                                              device=self.device)
        
        obs_without_command = torch.concat((obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                                            obs[:, self.env.privileged_dim + 9:self.env.num_obs-self.env.height_dim]), dim=1) #without command and hieght map and privileged_obs
        
        self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)),
                                               dim=1)

        # init world model input
        sum_wm_dataset_size = 0
        wm_latent = wm_action = None
        wm_is_first = torch.ones(self.env.num_envs, device=self._world_model.device)
        wm_obs = {
            "prop": obs[:, self.env.privileged_dim: self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device), # TODO need to excluede lin vel
            "is_first": wm_is_first,
        }
        if getattr(self.wm_config, "enable_foothold_prediction", False):
            wm_obs["foot_contact"] = (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.2).float().to(self._world_model.device)
        if self.wm_config.decode_pri_obs:
            wm_obs["privileged_obs"] = critic_obs[:, 0: self.env.privileged_dim].to(self._world_model.device)
            wm_obs["height_map"] = critic_obs[:, self.env.privileged_dim + self.env.cfg.env.prop_dim + self.env.cfg.env.action_dim : self.env.privileged_dim + self.env.cfg.env.prop_dim + self.env.cfg.env.action_dim + self.env.cfg.env.height_dim].to(self._world_model.device) 
        if(self.env.cfg.depth.use_camera):
            wm_obs["image"] = torch.zeros(((self.env.num_envs,) + self.env.cfg.depth.resized + (1,)), device=self._world_model.device)
        
        wm_metrics = None
        phase_model = getattr(self.wm_config, 'phase_model', False)
        omega_max = getattr(self.wm_config, 'omega_max', 12.566370614359172)
        phi = torch.zeros(self.env.num_envs, 1, device=self.device)

        wm_action_dim = self.env.num_actions + 1 if phase_model else self.env.num_actions
        self.wm_update_interval = self.env.cfg.depth.update_interval
        wm_action_history = torch.zeros(size=(self.env.num_envs, self.wm_update_interval, wm_action_dim),
                                        device=self._world_model.device)
        wm_reward = torch.zeros(self.env.num_envs, device=self._world_model.device)
        wm_feature = torch.zeros((self.env.num_envs, self.wm_feature_dim))

        self.init_wm_dataset()
        
        # 初始化指标
        mean_world_model_loss = 0.0
        mean_wm_image_loss = 0.0
        mean_wm_privileged_obs_loss = 0.0
        mean_wm_prop_loss = 0.0
        mean_wm_reward_loss = 0.0

        for it in range(self.current_learning_iteration, tot_iter):
            if (self.env.cfg.rewards.reward_curriculum):
                self.env.update_reward_curriculum(it)
            start = time.time()
            
            # Rollout - 使用CPG生成动作
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if (self.env.global_counter % self.wm_update_interval == 0):
                        # world model obs step
                        wm_embed = self._world_model.encoder(wm_obs)
                        if phase_model:
                            wm_embed = torch.concat((wm_embed, phi.to(self._world_model.device)), dim=-1)
                            if wm_action is not None:
                                wm_action = torch.concat((wm_action, phi.to(self._world_model.device)), dim=-1)
                        wm_latent, _ = self._world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed,
                                                                           wm_obs["is_first"])
                        wm_feature = self._world_model.dynamics.get_deter_feat(wm_latent)
                        wm_is_first[:] = 0

                    history = self.trajectory_history.flatten(1).to(self.device)
                    
                    # 使用CPG生成动作而不是策略网络
                    if self.env.cfg.cpg.use_CPG:
                        actions = self.env.get_cpg_actions()  # 假设环境中有这个方法
                    else:
                        # 如果没有CPG，使用随机动作
                        actions = torch.rand((self.env.num_envs, 18), device=self.device) * 2 - 1
                    
                    if phase_model:
                        # Append dummy raw_omega if not using actor
                        raw_omega = torch.zeros((self.env.num_envs, 1), device=self.device)
                        omega = omega_max + torch.tanh(raw_omega)
                        dt = getattr(self.env, 'dt', 0.02)
                        phi = phi + omega * dt
                        # Actions for world model include raw_omega
                        wm_step_actions = torch.concat((actions, raw_omega), dim=-1)
                    else:
                        wm_step_actions = actions

                    obs, privileged_obs, rewards, dones, infos, reset_env_ids = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(
                        self.device), rewards.to(self.device), dones.to(self.device)
                    
                    # update world model input
                    if phase_model:
                        wm_actions_to_cat = wm_step_actions.unsqueeze(1).to(self._world_model.device)
                    else:
                        wm_actions_to_cat = actions.unsqueeze(1).to(self._world_model.device)
                        
                    wm_action_history = torch.concat(
                        (wm_action_history[:, 1:], wm_actions_to_cat), dim=1)
                    wm_obs = {
                        "prop": obs[:, self.env.privileged_dim: self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device),
                        "is_first": wm_is_first,
                    }
                    if getattr(self.wm_config, "enable_foothold_prediction", False):
                        wm_obs["foot_contact"] = (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.2).float().to(self._world_model.device)
                    if self.wm_config.decode_pri_obs:
                        wm_obs["privileged_obs"] = critic_obs[:, 0: self.env.privileged_dim].to(self._world_model.device)
                        wm_obs["height_map"] = critic_obs[:, self.env.privileged_dim + self.env.cfg.env.prop_dim + self.env.cfg.env.action_dim : self.env.privileged_dim + self.env.cfg.env.prop_dim + self.env.cfg.env.action_dim + self.env.cfg.env.height_dim].to(self._world_model.device) 
                    
                    # store the data in buffer into the dataset before reset
                    reset_env_ids = reset_env_ids.cpu().numpy()
                    if (len(reset_env_ids) > 0):
                        for k, v in self.wm_dataset.items():
                            if(k == "image"):
                                for id in reset_env_ids:
                                    idx_in_buffer = np.where(self.env.depth_index == id)[0]
                                    if(len(idx_in_buffer) > 0):
                                        v[idx_in_buffer, :] = self.wm_buffer[k][idx_in_buffer].to(self._world_model.device)
                            else:
                                v[reset_env_ids, :] = self.wm_buffer[k][reset_env_ids].to(self._world_model.device)

                        self.wm_dataset_size[reset_env_ids] = self.wm_buffer_index[reset_env_ids]
                        self.wm_buffer_index[reset_env_ids] = 0
                        sum_wm_dataset_size = np.sum(self.wm_dataset_size)

                        wm_action_history[reset_env_ids, :] = 0
                        wm_is_first[reset_env_ids] = 1
                        if phase_model:
                            phi[reset_env_ids] = 0

                    wm_action = wm_action_history.flatten(1)
                    wm_reward += rewards.to(self._world_model.device)

                    # store current step into buffer
                    if (self.env.global_counter % self.wm_update_interval == 0):
                        if (self.env.cfg.depth.use_camera):
                            forward_heightmap = self.env.get_forward_map().to(self._world_model.device)
                            pred_depth_image = self.depth_predictor(forward_heightmap, wm_obs["prop"])
                            wm_obs["image"] = pred_depth_image
                            self.wm_buffer["forward_height_map"][range(self.env.num_envs), self.wm_buffer_index,:] = forward_heightmap[:].to('cpu')
                            wm_obs["image"][self.env.depth_index] = infos["depth"].unsqueeze(-1).to(self._world_model.device)
                            self.wm_buffer["image"][range(self.env.cfg.depth.camera_num_envs),
                            self.wm_buffer_index[self.env.depth_index], :] = wm_obs["image"][self.env.depth_index].to(
                                'cpu')
                        not_reset_env_ids = (1 - wm_is_first).nonzero(as_tuple=False).flatten().cpu().numpy()
                        if (len(not_reset_env_ids) > 0):
                            for k, v in wm_obs.items():
                                if(k != "is_first" and k != "image"):
                                    self.wm_buffer[k][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = v[not_reset_env_ids].to('cpu')
                            self.wm_buffer["action"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = \
                                wm_action[not_reset_env_ids, :].to('cpu')
                            self.wm_buffer["reward"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids]] = \
                                wm_reward[not_reset_env_ids].to('cpu')
                            # Store history
                            self.wm_buffer["history"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = \
                                self.trajectory_history[not_reset_env_ids].flatten(1).to('cpu')
                            self.wm_buffer_index[not_reset_env_ids] += 1

                        wm_reward[:] = 0
                    
                    # 不更新策略，只收集数据
                    rewards = rewards.to(self.device)
                    
                    # process trajectory history
                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.trajectory_history[env_ids] = 0
                    obs_without_command = torch.concat((obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                                                        obs[:, self.env.privileged_dim + 9:self.env.num_obs-self.env.height_dim]),
                                                       dim=1)
                    self.trajectory_history = torch.concat(
                        (self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # 不更新策略网络，只训练世界模型
                # mean_value_loss, mean_surrogate_loss, mean_vel_predict_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            #if self.log_dir is not None:
            #    self.log(locals())
            #if it % self.save_interval == 0:
            #    self.save_world_model(os.path.join(self.log_dir, 'world_model_{}.pt'.format(it)))
            #ep_infos.clear()

            start_time = time.time()
            if (sum_wm_dataset_size > self.wm_config.train_start_steps):

                if(it % self.depth_predictor_cfg["training_interval"] == 0 and self.env.cfg.depth.use_camera):
                # Train Depth Predictor
                    depth_mse_loss = self.train_depth_predictor()
                    self.writer.add_scalar('DepthPredictor/loss', depth_mse_loss, it)

                # Train World Model
                wm_metrics = self.train_world_model(it)
                world_model_loss = wm_metrics["model_loss"]
                mean_world_model_loss = np.mean(world_model_loss)
                if (self.env.cfg.depth.use_camera):
                    wm_image_loss = wm_metrics["image_loss"]
                    mean_wm_image_loss = np.mean(wm_image_loss)
                wm_reward_loss = wm_metrics["reward_loss"]
                mean_wm_reward_loss = np.mean(wm_reward_loss)
                if (self.wm_config.decode_pri_obs) :
                    wm_privileged_obs_loss = wm_metrics["privileged_obs_loss"]
                    mean_wm_privileged_obs_loss = np.mean(wm_privileged_obs_loss)
                wm_prop_loss = wm_metrics["prop_loss"]
                mean_wm_prop_loss = np.mean(wm_prop_loss)
                for name, values in wm_metrics.items():
                    self.writer.add_scalar('World_model/' + name, float(np.mean(values)), it)
            train_world_model_time = time.time() - start_time
            print('training world model time:', train_world_model_time)
            
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save_world_model(os.path.join(self.log_dir, 'world_model_{}.pt'.format(it)))
                #self.collect_cpg_dataset(self.env, 1000, os.path.join('/datasets/motion_files', 'cpg_motion_{}.json'.format(it)))
            ep_infos.clear()
            # copy the config file
            if(it == 0):
                os.system("cp ./gym/envs/hexapodMBRL_config.py " + self.log_dir + "/")
                
        self.current_learning_iteration += num_learning_iterations
        self.save_world_model(os.path.join(self.log_dir, 'world_model_final.pt'))
        
        # 重新启用策略网络的梯度计算
        for param in self.alg.actor_critic.parameters():
            param.requires_grad = True


    def save_world_model(self, path):
        torch.save(self._world_model.state_dict(), path)
        print(f"World model saved to {path}")

    def load_world_model(self, path, load_wm_optimizer = False):
        state_dict = torch.load(path, map_location=self._world_model.device)
        
        # 同样添加 strip module 逻辑
        def strip_module(sd):
            new_state_dict = {}
            for k, v in sd.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            return new_state_dict

        if any(k.startswith('module.') for k in state_dict.keys()):
            print("[load_world_model] Detected 'module.' prefix in checkpoint. Stripping...")
            state_dict = strip_module(state_dict)

        self._world_model.load_state_dict(state_dict, strict=False) # 建议使用 strict=False 以避免版本差异导致 crash
        
        if(load_wm_optimizer):
            self._world_model._model_opt._opt.load_state_dict(state_dict['wm_optimizer_state_dict'])
        print(f"World model loaded from {path}")
        #return state_dict['infos']

    def collect_cpg_dataset(self, env, num_steps=1000, save_path="datasets/motion_files/cpg_motion.json"):
        frames = []

        for i in range(num_steps):
            # 执行一步 CPG 动作
            actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
            obs, _, _, _ = env.step(actions)

            # 取 root 状态
            root_pos = env.root_states[0, 0:3].cpu().numpy()         # (3,)
            root_rot = env.root_states[0, 3:7].cpu().numpy()         # (4,)
            lin_vel  = env.root_states[0, 7:10].cpu().numpy()        # (3,)
            ang_vel  = env.root_states[0, 10:13].cpu().numpy()       # (3,)

            # 关节状态
            joint_pos = env.dof_pos[0].cpu().numpy()                 # (18,)
            joint_vel = env.dof_vel[0].cpu().numpy()                 # (18,)

            # toe pos / vel 如果环境能提供，就补上，否则用 0 占位
            toe_pos_local = env.toe_pos[0].cpu().numpy()    # (18,)
            toe_vel_local = env.toe_vel[0].cpu().numpy()    # (18,)

            # 拼成一帧
            frame = np.concatenate([
                root_pos, root_rot,
                joint_pos, toe_pos_local,
                lin_vel, ang_vel,
                joint_vel, toe_vel_local
            ])
            frames.append(frame.tolist())

        # 存储 json
        motion_data = {
            "Frames": frames,
            "FrameDuration": float(env.dt),  # 每一步的时间间隔
            "MotionWeight": 1.0
        }

        with open(save_path, "w") as f:
            json.dump(motion_data, f)

        print(f"✅ Saved CPG dataset to {save_path}, total {len(frames)} frames")

    def _get_world_model(self):
        """安全地获取world model，处理DDP包装"""
        if isinstance(self._world_model, torch.nn.parallel.DistributedDataParallel):
            return self._world_model.module
        return self._world_model

    def _get_world_model_device(self):
        """获取world model的设备"""
        if isinstance(self._world_model, torch.nn.parallel.DistributedDataParallel):
            return next(self._world_model.parameters()).device
        return self._world_model.device