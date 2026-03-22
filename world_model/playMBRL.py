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

import os
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from gym import GYM_ROOT_DIR

import isaacgym
from gym.envs import *
from gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import json

# -----------------------------
# 测试开关控制 (需与训练时的配置保持一致)
# -----------------------------
REMOVE_DOF_VEL_OBS = True  # 是否在 Observation 中去掉关节速度
USE_DELTA_ACTIONS = False   # 是否开启增量动作模式 (Action = Delta)
DELTA_ACTION_SCALE = 0.5    # 增量缩放系数

def quat_to_euler(quat):
    """
    Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw).
    """
    single = False
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
        single = True
    x = quat[:,0]; y = quat[:,1]; z = quat[:,2]; w = quat[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4) + torch.pi / 2
    yaw_z = torch.where(yaw_z > 2 * torch.pi, yaw_z - 2 * torch.pi, yaw_z)
    
    if single:
        return roll_x[0], pitch_y[0], yaw_z[0]
    return roll_x, pitch_y, yaw_z # in radians

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.depth.camera_num_envs = min(env_cfg.depth.camera_num_envs, 1)
    env_cfg.terrain.num_cols = env_cfg.env.num_envs
    env_cfg.terrain.curriculum = True#False
    env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_restitution = False
    # env_cfg.commands.heading_command = True

    env_cfg.domain_rand.friction_range = [1.0, 1.0]
    env_cfg.domain_rand.restitution_range = [0.0, 0.0]
    env_cfg.domain_rand.added_mass_range = [0., 0.]  # kg
    env_cfg.domain_rand.com_x_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_y_pos_range = [-0.0, 0.0]
    env_cfg.domain_rand.com_z_pos_range = [-0.0, 0.0]

    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = True
    # env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    # env_cfg.domain_rand.randomize_com_pos = False
    env_cfg.domain_rand.randomize_motor_strength = False

    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.domain_rand.stiffness_multiplier_range = [1.0, 1.0]
    env_cfg.domain_rand.damping_multiplier_range = [1.0, 1.0]


    # env_cfg.terrain.mesh_type = 'plane'
    if(env_cfg.terrain.mesh_type == 'plane'):
        env_cfg.rewards.scales.feet_edge = 0
        env_cfg.rewards.scales.feet_stumble = 0
    if env_cfg.terrain.is_plane:
       env_cfg.depth.use_camera = False
       env_cfg.env.height_dim = 0 

    if(args.terrain not in ['slope', 'stair', 'gap', 'climb', 'crawl', 'tilt']):
        print('terrain should be one of slope, stair, gap, climb, crawl, and tilt, set to climb as default')
        args.terrain = 'climb'
    env_cfg.terrain.terrain_proportions = {
        'slope': [0, 1.0, 0.0, 0, 0, 0, 0, 0, 0],
        'stair': [0, 0, 1.0, 0, 0, 0, 0, 0, 0],
        'gap': [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],
        'climb': [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0],
        'tilt': [0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0],
        'crawl': [0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0],
     }[args.terrain]

    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.055, 0.055]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0, 0]

    env_cfg.commands.ranges.flat_lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.flat_lin_vel_y = [0.055, 0.055]
    env_cfg.commands.ranges.flat_ang_vel_yaw = [0.0, 0.0]

    env_cfg.depth.use_camera = False#True
    # 同步本地开关到环境配置
    env_cfg.env.remove_dof_vel = REMOVE_DOF_VEL_OBS
    env_cfg.control.use_delta_actions = USE_DELTA_ACTIONS
    env_cfg.control.delta_action_scale = DELTA_ACTION_SCALE

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # env.terrain_levels = torch.ones((env.num_envs,), device=env.device).long()
    _, _ = env.reset()
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = 'MP-HWM-rwm-decode-priobsALL-ddp-18-cpg-reward'#'MP-HWM-rwm-decode-priobsALL-ddp-17-cpg-reward-resume'#'MP-HWM-rwm-decode-priobsALL-ddp-16-cpg-reward'#-resume'#'MP-HWM-rwm-decode-priobsALL-ddp-cpgrew'#'CPG-HWM-rwm-decode-priobs-imag'#'WMP-Settings'#'Real_robot_test_extraobs_privileged_obs'#'Real_robot_test_privileged_obs_camera'#'Real_robot_test_extraobs_no_privileged_obs'#'Hexapod_terrain_055speed_ft_wm_camera_test_camera'#'Hexapod_terrain_with_measureheight_02_clearance_nobaseheight_minfat_nevel' #'Hexapod_terrain_with_measureheight_nonmeanvel'#'Hexapod_plane_randvel'
    #'MP-HWM-rwm-decode-priobsALL-ddp-2'
    #'Real_robot_test_no_privileged_obs' # Real_robot_test_privileged_obs
    train_cfg.runner.checkpoint = 8000
    train_cfg.runner.ddp = False  # 显式设置为 False，确保 play 模式下不使用 DDP
    #ppo_runner, train_cfg = task_registry.make_wmp_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    ppo_runner, train_cfg = task_registry.make_rwm_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    history_length = 5
    # ---- indices (robust to remove_dof_vel which changes prop_dim) ----
    # obs layout (as used by this project):
    # [ privileged (env.privileged_dim),
    #   prop (env.cfg.env.prop_dim) : [base_ang_vel(3), gravity(3), commands(3), dof_pos(18), (dof_vel(18)?), prev_action(18), (phase(6)?)]
    #   action (env.num_actions),
    #   height_map (env.height_dim) ]
    prop_start = env.privileged_dim
    cmd_start = prop_start + 6
    cmd_end = cmd_start + 3
    non_height_end = prop_start + env.cfg.env.prop_dim + env.cfg.env.action_dim

    trajectory_history = torch.zeros(
        size=(env.num_envs, history_length, env.num_obs - env.privileged_dim - env.height_dim - 3),  # exclude commands
        device=env.device,
    )
    # remove commands from prop; keep (ang_vel+gravity) and (dof_pos + (dof_vel?) + prev_action + phase) plus action segment
    obs_without_command = torch.concat((obs[:, prop_start:cmd_start],
                                        obs[:, cmd_end:non_height_end]), dim=1)
    trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

    world_model = ppo_runner._world_model.to(env.device)
    wm_latent = wm_action = None
    wm_is_first = torch.ones(env.num_envs, device=env.device)
    wm_update_interval = env.cfg.depth.update_interval
    wm_action_history = torch.zeros(size=(env.num_envs, wm_update_interval, env.num_actions),
                                    device=env.device)
    wm_obs = {
        "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
        "is_first": wm_is_first,
    }

    if (env.cfg.depth.use_camera):
        wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                      device=world_model.device)

    wm_feature = torch.zeros((env.num_envs, ppo_runner.wm_feature_dim), device=env.device)

    total_reward = 0
    not_dones = torch.ones((env.num_envs,), device=env.device)
    
    frames = []
    contact_frames = []
    actions_output = []
    # Initialize storage for reconstruction visualization
    recorded_data = {
        "real_pri_obs": [],
        "recon_pri_obs": [],
        "real_height_map": [],
        "recon_height_map": []
    }
    
    wm_obs_prop = []
    imu_data = []
    wm_obs_prop.append(wm_obs["prop"].detach().cpu().numpy())
    imu_data.append(quat_to_euler(env.root_states[0, 3:7]))
    #actions_output.append(actions.detach().cpu().numpy())
    
    for i in range(1*int(env.max_episode_length) + 3):
    #for i in range(96):
    #while True:
        #i = 1
        if (env.global_counter % wm_update_interval == 0):
            if (env.cfg.depth.use_camera):
                # print("wm_obs shape: ", wm_obs["image"][env.depth_index].shape)
                # print("info shape: ", infos["depth"].unsqueeze(-1).to(world_model.device).shape)
                wm_obs["image"][env.depth_index] = infos["depth"].unsqueeze(-1).to(world_model.device)

            wm_embed = world_model.encoder(wm_obs)
            wm_latent, _ = world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"], sample=True)
            wm_feature = world_model.dynamics.get_deter_feat(wm_latent)
            
            # Reconstruct privileged information
            feat = world_model.dynamics.get_feat(wm_latent)
            decoded = world_model.heads["decoder"](feat)
            if "privileged_obs" in decoded:
                # 保存recon_pri_obs和real_pri_obs到文件
                recon_pri_obs = decoded["privileged_obs"].mean()
                real_pri_obs = obs[:, :env.privileged_dim]
                pri_obs_error = torch.mean((recon_pri_obs - real_pri_obs) ** 2)
                #print(f"Step {i} | Privileged Obs Reconstruction Error: {pri_obs_error.item():.6f}")
                
                recorded_data["real_pri_obs"].append(real_pri_obs.detach().cpu().numpy())
                recorded_data["recon_pri_obs"].append(recon_pri_obs.detach().cpu().numpy())
            else:
                if i == 0:
                    print(f"Warning: 'privileged_obs' key not found in decoder output. Available keys: {decoded.keys()}")
            if "height_map" in decoded:
                recon_height_map = decoded["height_map"].mean()
                real_height_map = obs[:, env.privileged_dim + env.cfg.env.prop_dim + env.cfg.env.action_dim: env.privileged_dim + env.cfg.env.prop_dim + env.cfg.env.action_dim + env.cfg.env.height_dim]
                height_map_error = torch.mean((recon_height_map - real_height_map) ** 2)
                #print(f"Step {i} | Height Map Reconstruction Error: {height_map_error.item():.6f}, recon_height_map: {recon_height_map.shape}, real_height_map: {real_height_map.shape}")
                
                recorded_data["real_height_map"].append(real_height_map.detach().cpu().numpy())
                recorded_data["recon_height_map"].append(recon_height_map.detach().cpu().numpy())
            else:
                if i == 0:
                    print(f"Warning: 'height_map' key not found in decoder output. Available keys: {decoded.keys()}")
            
            wm_is_first[:] = 0

        history = trajectory_history.flatten(1).to(env.device)
        actions = policy(obs.detach(), history.detach(), wm_feature.detach())
        #print("actions: ", actions)
        #single_action = torch.tensor([0, 0, -6.0, 0, 0, 0.0, 0, 0, 0.0,
        #                            0, 0, -0.0, 0, 0, -0.0, 0, 0, -0.0],
        #                            dtype=torch.float32, device=env.device)

        #actions = single_action.unsqueeze(0).repeat(env.num_envs, 1)

        #while True:
        #    env.render()
        obs, _, rews, dones, infos, reset_env_ids = env.step(actions.detach())

        not_dones *= (~dones)
        total_reward += torch.mean(rews * not_dones)

        # update world model input
        wm_action_history = torch.concat(
            (wm_action_history[:, 1:], actions.unsqueeze(1)), dim=1)
        wm_obs = {
            "prop": obs[:, env.privileged_dim: env.privileged_dim + env.cfg.env.prop_dim],
            "is_first": wm_is_first,
        }
        if (env.cfg.depth.use_camera):
            wm_obs["image"] = torch.zeros(((env.num_envs,) + env.cfg.depth.resized + (1,)),
                                          device=world_model.device)
        #print("env.privileged_dim: ", env.privileged_dim)
        #print("env.cfg.env.prop_dim: ", env.cfg.env.prop_dim)
        wm_obs_prop.append(wm_obs["prop"].detach().cpu().numpy())
        imu_data.append(quat_to_euler(env.root_states[0, 3:7]))
        actions_output.append(actions.detach().cpu().numpy())

        reset_env_ids = reset_env_ids.cpu().numpy()
        if (len(reset_env_ids) > 0):
            wm_action_history[reset_env_ids, :] = 0
            wm_is_first[reset_env_ids] = 1

        wm_action = wm_action_history.flatten(1)


        # process trajectory history
        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        # keep consistent with initialization (robust to remove_dof_vel)
        obs_without_command = torch.concat((obs[:, prop_start:cmd_start],
                                            obs[:, cmd_end:non_height_end]), dim=1)
        trajectory_history = torch.concat(
            (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            lootat = env.root_states[8, :3]
            camara_position = lootat.detach().cpu().numpy() + [0, 1, 0]
            env.set_camera(camara_position, lootat)

        
        if i < stop_state_log:
            # 计算当前显示的关节目标值
            if getattr(env.cfg.control, "use_delta_actions", False):
                # 增量模式：显示累加后的绝对目标角度
                actual_target = env.cur_targets[robot_index, joint_index].item() * env.cfg.control.action_scale
            else:
                # 绝对模式：显示当前 Action 直接映射的角度
                actual_target = actions[robot_index, joint_index].item() * env.cfg.control.action_scale

            logger.log_states(
                {
                    'dof_pos_target': actual_target,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
            
        root_pos = env.root_states[0, 0:3].cpu().numpy()         # (3,)
        root_rot = env.root_states[0, 3:7].cpu().numpy()         # (4,)
        lin_vel_global = env.root_states[0, 7:10].cpu().numpy()        # (3,)
        ang_vel_global = env.root_states[0, 10:13].cpu().numpy()       # (3,)
        #print("lin_vel_global: ", lin_vel_global)
        #print("ang_vel_global: ", ang_vel_global)
        lin_vel = env.base_lin_vel[0].cpu().numpy()        # (3,)
        ang_vel = env.base_ang_vel[0].cpu().numpy()       # (3,)
        #print("lin_vel: ", lin_vel)
        #print("ang_vel: ", ang_vel)
        # 关节状态
        joint_pos = env.dof_pos[0].cpu().numpy()                 # (18,)
        joint_vel = env.dof_vel[0].cpu().numpy()                 # (18,)
        # toe pos / vel 如果环境能提供，就补上，否则用 0 占位
        #toe_pos_local = env.toe_pos[0].flatten().cpu().numpy()    # (18,)
        #toe_vel_local = env.toe_vel[0].flatten().cpu().numpy()    # (18,)
        toe_pos_local = np.zeros(18)    # (18,)
        toe_vel_local = np.zeros(18)    # (18,)
        # 拼成一帧
        frame = np.concatenate([
            root_pos, root_rot,
            joint_pos, toe_pos_local,
            lin_vel, ang_vel,
            joint_vel, toe_vel_local
        ])
        frames.append(frame.tolist())

        # 记录足端接触情况 (1表示接触, 0表示未接触)
        contact = (env.contact_forces[0, env.feet_indices, 2] > 1.2).cpu().numpy().astype(int)
        #print("contact forces: ", env.contact_forces[0, env.feet_indices, 2])
        contact_frames.append(contact.tolist())

    save_path = "./datasets/motion_files/cpg_motion.json"
    # 存储 json
    motion_data = {
        "Frames": frames,
        "FrameDuration": float(env.dt),  # 每一步的时间间隔
        "MotionWeight": 1.0
    }

    with open(save_path, "w") as f:
        json.dump(motion_data, f)

    print(f"✅ Saved CPG dataset to {save_path}, total {len(frames)} frames")

    # 保存接触情况到 CSV 文件
    contact_save_path = "./datasets/motion_files/contact_data.csv"
    np.savetxt(contact_save_path, np.array(contact_frames), delimiter=",", fmt='%d')
    print(f"✅ Saved contact data to {contact_save_path}")
    
    # Save reconstruction data
    if False:
        recon_save_path = "./experiments/reconstruction_data.npz"
        # Ensure directory exists
        os.makedirs(os.path.dirname(recon_save_path), exist_ok=True)
        np.savez(recon_save_path, 
                real_pri_obs=np.array(recorded_data["real_pri_obs"]),
                recon_pri_obs=np.array(recorded_data["recon_pri_obs"]),
                real_height_map=np.array(recorded_data["real_height_map"]),
                recon_height_map=np.array(recorded_data["recon_height_map"]))
        print(f"✅ Saved reconstruction data to {recon_save_path}")
    
    # 保存本体感知信息到txt文件中，也就是wm_obs["prop"]，同时保存IMU的数据
    if False:
        with open("wm_obs_prop.txt", "a") as f:
            for i in range(len(wm_obs_prop)):
                f.write(str(wm_obs_prop[i]) + "\n")    
        with open("imu_data.txt", "a") as f:
            for i in range(len(imu_data)):
                f.write(str(imu_data[i]) + "\n")
        with open("actions_output.txt", "a") as f:
            for i in range(len(actions_output)):
                f.write(str(actions_output[i]) + "\n")
    
    
    print('total reward:', total_reward)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    args.rl_device = args.sim_device
    play(args)
