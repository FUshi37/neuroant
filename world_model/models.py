# MIT License

# Copyright (c) 2023 NM512

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import copy
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Callable
from . import tools
from . import networks
import time
import numpy as np
#from rsl_rl.rsl_rl.storage import RolloutStorage

to_np = lambda x: x.detach().cpu().numpy()

class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class FootholdPredictionHead(nn.Module):
    def __init__(self, latent_dim, hind_leg_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, hind_leg_dim),
        )

    def forward(self, z):
        return self.net(z)

class WorldModel(nn.Module):
    def __init__(self, config, obs_shape, use_camera, pri_obs_shape = 0, device = None):
        super(WorldModel, self).__init__()
        # self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        if device is not None:
            self.device = device
        else:
            self.device = self._config.device

        self.encoder = networks.MultiEncoder(obs_shape, **config.encoder, use_camera=use_camera)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            #config.device,
            self.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        # self.heads["decoder"] = networks.MultiDecoder(
        #     feat_size, obs_shape, **config.decoder, use_camera=use_camera
        # )
        if config.decode_pri_obs:
            self.heads["decoder"] = networks.MultiDecoder(
                feat_size, pri_obs_shape, **config.decoder, use_camera=use_camera
            )
        else:
            self.heads["decoder"] = networks.MultiDecoder(
                feat_size, obs_shape, **config.decoder, use_camera=use_camera
            )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        # self.heads["cont"] = networks.MLP(
        #     feat_size,
        #     (),
        #     config.cont_head["layers"],
        #     config.units,
        #     config.act,
        #     config.norm,
        #     dist="binary",
        #     outscale=config.cont_head["outscale"],
        #     device=config.device,
        #     name="Cont",
        # )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        # can set different scale for terms in decoder here
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            image = 1.0,
            # privileged_obs = 0.5,
            # forward_height_map = 0.5,
            # clean_prop = 0,
            # cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        # print("data key:", data.keys())
        data = self.preprocess(data)
        # print("data key:", data.keys())
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                # print("heads:", self.heads.keys())
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                # print("preds:", preds.keys())
                losses = {}
                for name, pred in preds.items():
                    # print("pred name:", name)
                    #print(f"data['{name}'] shape: ", data[name].shape)
                    pred_mean = getattr(pred, "mean", None)
                    if callable(pred_mean):
                        pred_mean = pred_mean()
                    mean_shape = getattr(pred_mean, "shape", None)
                    #print(name, type(pred), mean_shape, data[name].shape)
                    #loss = -pred.log_prob(data[name])
                    target = data[name]
                    if mean_shape is not None and target.shape != mean_shape:
                        if target.shape + (1,) == mean_shape:
                            target = target.unsqueeze(-1)
                        else:
                            raise ValueError(
                                f"Target shape {target.shape} does not match prediction shape {mean_shape} for head '{name}'."
                            )
                    target_device = None
                    if torch.is_tensor(pred_mean):
                        target_device = pred_mean.device
                    else:
                        pred_mode = getattr(pred, "mode", None)
                        if callable(pred_mode):
                            pred_mode = pred_mode()
                        if torch.is_tensor(pred_mode):
                            target_device = pred_mode.device
                    if target_device is None:
                        target_device = feat.device
                    pred_dist = pred._dist if hasattr(pred, "_dist") else None
                    base_dist = pred_dist
                    while base_dist is not None and hasattr(base_dist, "base_dist"):
                        base_dist = base_dist.base_dist

                    def _to_device(tensor, dev):
                        return tensor.to(dev)

                    target = target.to(target_device)
                    if base_dist is not None:
                        for attr in ("loc", "scale", "mean"):
                            param = getattr(base_dist, attr, None)
                            if torch.is_tensor(param) and param.device != target_device:
                                setattr(base_dist, attr, _to_device(param, target_device))
                    if hasattr(pred, "_mode") and torch.is_tensor(pred._mode) and pred._mode.device != target_device:
                        pred._mode = pred._mode.to(target_device)
                    loss = -pred.log_prob(target)
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        # obs = obs.copy()
        # obs["image"] = torch.Tensor(obs["image"]) / 255.0

        # discount in obs seems useless
        # if "discount" in obs:
        #     obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            # obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        # assert "is_terminal" in obs
        # obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        #obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        current_device = next(self.parameters()).device
        if self.device != current_device:
            self.device = current_device
        obs = {k: torch.as_tensor(v, device=self.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model], 2)
    

class WorldModelRWM(nn.Module):
    def __init__(self, config, obs_shape, use_camera, k_steps=3, pri_obs_shape = 0, device = None):
        super(WorldModelRWM, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        if device is not None:
            self.device = device
        else:
            self.device = self._config.device
        self.k_steps = k_steps
        self.phase_model = getattr(config, 'phase_model', False)
        self.omega_max = getattr(config, 'omega_max', 12.566370614359172)

        # Initialize components similar to WorldModel
        self.encoder = networks.MultiEncoder(obs_shape, **config.encoder, use_camera=use_camera)
        self.embed_size = self.encoder.outdim
        
        rssm_embed_size = self.embed_size
        rssm_num_actions = config.num_actions
        if self.phase_model:
            rssm_embed_size += 1
            # Add phi to action input of RSSM. 
            rssm_num_actions += 1 

        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            rssm_num_actions,
            rssm_embed_size,
            #config.device,
            self.device,
        )
        
        # Initialize heads
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.feat_size = feat_size
        
        decoder_feat_size = feat_size
        if self.phase_model:
            decoder_feat_size += 1

        if config.decode_pri_obs:
            self.heads["decoder"] = networks.MultiDecoder(
                decoder_feat_size, pri_obs_shape, **config.decoder, use_camera=use_camera
            )
        else:
            self.heads["decoder"] = networks.MultiDecoder(
                decoder_feat_size, obs_shape, **config.decoder, use_camera=use_camera
            )
        self.heads["reward"] = networks.MLP(
            decoder_feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            #device=config.device,
            device=self.device,
            name="Reward",
        )
        self.enable_foothold_prediction = bool(getattr(config, "enable_foothold_prediction", False))
        hind_leg_indices = getattr(config, "foothold_hind_leg_indices", [2, 3, 4, 5])
        self.foothold_hind_leg_indices = [int(idx) for idx in hind_leg_indices]
        self.foothold_delay = int(getattr(config, "foothold_delay", 5))
        if self.enable_foothold_prediction:
            self.foothold_head = FootholdPredictionHead(
                self.feat_size, len(self.foothold_hind_leg_indices)
            ).to(self.device)
        else:
            self.foothold_head = None

        # Initialize optimizer
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )

        # Loss scales
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            image=1.0,
        )
        if self.enable_foothold_prediction:
            self._scales["foothold"] = float(getattr(config, "foothold_loss_scale", 0.5))

    def _compute_foothold_loss(self, post_feat, foot_contact, start_idx, time_dim):
        if not self.enable_foothold_prediction or self.foothold_head is None:
            return None
        if foot_contact is None:
            return None
        target_contact = foot_contact[
            :, start_idx + self.foothold_delay : start_idx + self.foothold_delay + time_dim
        ]
        valid_steps = target_contact.shape[1]
        if valid_steps <= 0:
            return None
        pred_contact = self.foothold_head(post_feat[:, :valid_steps])
        target_contact = target_contact[..., self.foothold_hind_leg_indices]
        loss = F.binary_cross_entropy_with_logits(
            pred_contact,
            target_contact,
            reduction="none",
        ).mean(dim=-1)
        padded_loss = torch.zeros(post_feat.shape[:2], device=post_feat.device, dtype=post_feat.dtype)
        padded_loss[:, :valid_steps] = loss
        return padded_loss

    def _train(self, data, act_func: Optional[Callable]=None):
        # act_funt: self.actor_critic.act(obs, history, wm_feature)
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        # print("data key:", data.keys())
        data = self.preprocess(data)
        # print("data key:", data.keys())
        original_data = data.copy()
        data = {k: v[:, :-self.k_steps+1] for k, v in data.items()}
        # print("data key:", data.keys())
        model_loss = []

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                for i in range(self.k_steps):
                    # print("i: ", i)
                    if i + 1 == self.k_steps:
                        data["is_first"] = original_data["is_first"][:, i:]
                    else:
                        data["is_first"] = original_data["is_first"][:, i:-self.k_steps+1+i]
                    
                    if self.phase_model:
                        # Compute phi sequence for the current data["action"]
                        # data["action"] shape: (batch, time, dim)
                        B, T, _ = data["action"].shape
                        raw_omega = data["action"][..., -1:]
                        omega = self.omega_max + torch.tanh(raw_omega)
                        dt = getattr(self._config, 'dt', 0.02)
                        phi_seq = torch.zeros((B, T, 1), device=self.device)
                        current_phi = torch.zeros((B, 1), device=self.device)
                        is_first_t = data["is_first"]
                        if is_first_t.dim() == 2:
                            is_first_t = is_first_t.unsqueeze(-1)
                        
                        for t in range(T):
                            current_phi = current_phi * (1.0 - is_first_t[:, t])
                            phi_seq[:, t] = current_phi
                            current_phi = current_phi + omega[:, t] * dt
                        
                        # Concat phi to embed
                        embed = self.encoder(data)
                        embed = torch.concat((embed, phi_seq), dim=-1)
                    else:
                        embed = self.encoder(data)
                        phi_seq = None

                    # generate action -> obs -> action
                    rssm_action = data["action"]
                    if self.phase_model:
                        rssm_action = torch.concat((rssm_action, phi_seq), dim=-1)
                    
                    post, prior = self.dynamics.observe(embed, rssm_action, data["is_first"])
                    if act_func is not None:
                        # If phase_model, we need to pass phi_t to act_func
                        # But act_func might be called inside loop? No, act_func here is applied to sequence.
                        # Wait, the act_func call below:
                        if self.phase_model:
                            action = act_func(data["prop"], data["history"], 
                                        self.dynamics.get_deter_feat(post), phi=phi_seq, proprioception_only=True)
                            data["action"] = torch.concat((data["action"][..., 19:], action), dim=-1)
                        else:
                            action = act_func(data["prop"], data["history"], 
                                        self.dynamics.get_deter_feat(post), proprioception_only=True)
                            data["action"] = torch.concat((data["action"][..., 18:], action), dim=-1)
                    else:
                        print("[WARNING] act_func is not provided, using data['action']")

                    kl_free = self._config.kl_free
                    dyn_scale = self._config.dyn_scale
                    rep_scale = self._config.rep_scale
                    kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                        post, prior, kl_free, dyn_scale, rep_scale
                    )
                    assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                    preds = {}
                    base_post_feat = self.dynamics.get_feat(post)
                    # print("heads:", self.heads.keys())
                    for name, head in self.heads.items():
                        grad_head = name in self._config.grad_heads
                        post_feat = base_post_feat
                        
                        if self.phase_model:
                            # Concat phi to post_feat for decoder
                            post_feat = torch.concat((post_feat, phi_seq), dim=-1)
                        
                        post_feat = post_feat if grad_head else post_feat.detach()
                        pred = head(post_feat)

                        if name in data.keys():
                            data[name] = pred

                        if type(pred) is dict:
                            preds.update(pred)
                        else:
                            preds[name] = pred
                    
                    # For history update, we only include the joint angles (first 18 dims of action)
                    # to keep it consistent with the history collected from the environment.
                    hist_action = action[..., :18]
                    obs_without_command = torch.concat((data["prop"][..., :6], 
                                                        data["prop"][..., 9:],
                                                        hist_action), dim=-1)
                    
                    step_dim = obs_without_command.shape[-1]
                    data["history"] = torch.concat((data["history"][..., step_dim:], obs_without_command),
                                               dim=-1)
                    #data["history"] = torch.concat((data["history"][..., 66:], obs_without_command),
                    #                               dim=-1)
                    ## --- diagnostic prints for reward debugging ---
                    #if "reward" in preds:
                    #    pred = preds["reward"]
                        # type check
                        #print("DEBUG reward pred class:", type(pred))
                        # if torch.distributions.Distribution-like object, try to inspect params
                        #try:
                        #    if hasattr(pred, "mean"):
                        #        print("DEBUG pred.mean() sample:", pred.mean().detach().cpu().numpy().flatten()[:6])
                        #    if hasattr(pred, "scale") or hasattr(pred, "stddev"):
                        #        s = getattr(pred, "scale", None) or getattr(pred, "stddev", None)
                        #        if s is not None:
                        #            print("DEBUG pred.scale() sample:", s.detach().cpu().numpy().flatten()[:6])
                        #    except Exception as e:
                        #        print("DEBUG inspecting pred params failed:", e)

                        # inspect target slice
                        #if i + 1 == self.k_steps:
                        #    target = original_data["reward"][:, i:]
                        #else:
                        #    target = original_data["reward"][:, i:-self.k_steps+1+i]
                        #print("DEBUG target.shape, min, max, mean:", target.shape, target.min().item(), target.max().item(), target.mean().item())

                        # compute a raw log_prob (catch shape errors)
                        #try:
                        #    lp = pred.log_prob(target)
                        #    print("DEBUG pred.log_prob() sample:", lp.detach().cpu().numpy().flatten()[:6])
                        #    print("DEBUG pred.log_prob() mean:", float(lp.mean().detach().cpu().numpy()))
                        #except Exception as e:
                        #    print("DEBUG pred.log_prob() error:", e)

                    # print("preds:", preds.keys())
                    losses = {}
                    for name, pred in preds.items():
                        #print("pred name:", name)
                        if i + 1 == self.k_steps:
                            target = original_data[name][:, i:]
                            #loss = -pred.log_prob(original_data[name][:, i:])
                        else:
                            target = original_data[name][:, i:-self.k_steps+1+i]
                        #print(f"target['{name}'] shape: ", target.shape)
                            #loss = -pred.log_prob(original_data[name][:, i:-self.k_steps+1+i])
                        ## --- 设备检查 ---
                        #pred_device = None
                        #if isinstance(pred, torch.distributions.Distribution):
                        #    if hasattr(pred, "loc"):
                        #        pred_device = pred.loc.device
                        #    elif hasattr(pred, "mean") and isinstance(pred.mean, torch.Tensor):
                        #        pred_device = pred.mean.device
                        #elif isinstance(pred, torch.Tensor):
                        #    pred_device = pred.device
                        #else:
                            #print(f"[WARNING] pred for head {name} is type={type(pred)}, cannot infer device directly")
                        #print(f"[DEBUG] step={i}, head={name}, pred_type={type(pred)}, pred_device={pred_device}, target_device={target.device}")

                        try:
                            loss = -pred.log_prob(target)
                        except ValueError as e:
                            # ValueError 常因形状不匹配（例如需要最后一维），尝试 unsqueeze(-1)
                            try:
                                loss = -pred.log_prob(target.unsqueeze(-1))
                            except Exception:
                                # 若仍然失败，抛出原始错误，便于定位
                                raise e
                        assert loss.shape == embed.shape[:2], (name, loss.shape)
                        losses[name] = loss

                    foothold_loss = self._compute_foothold_loss(
                        base_post_feat,
                        original_data.get("foot_contact"),
                        i,
                        embed.shape[1],
                    )
                    if foothold_loss is not None:
                        losses["foothold"] = foothold_loss

                    scaled = {
                        key: value * self._scales.get(key, 1.0)
                        for key, value in losses.items()
                    }
                    model_loss.append(sum(scaled.values()) + kl_loss)

            model_loss = torch.stack(model_loss, dim=1)
            model_loss = torch.mean(model_loss, dim=1)
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def preprocess(self, obs):
        assert "is_first" in obs
        #obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        #current_device = next(self.parameters()).device
        #if self.device != current_device:
        #    self.device = current_device
        #obs = {k: torch.as_tensor(v, device=self.device) for k, v in obs.items()}
        obs = {k: torch.Tensor(v).to(self.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model], 2)
    
class ImagBehavior(nn.Module):
    def __init__(self, config, world_model, actor_critic):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self.actor_critic = actor_critic  # actor_critic is an instance of ActorCriticRWM
        self.actor = actor_critic.actor
        self.critic = actor_critic.critic
        self.max_grad_norm = 1.0
        self.value_loss_coef = 0.5  # ↓ 降低critic对总损失的影响
        self.critic_grad_clip = 0.5 # ↓ 单独更严的critic梯度裁剪
        self.gamma = 0.998
        self.lam = 0.95

        # -------- New: 冻结的旧策略，用于计算 PPO ratio --------
        # 注意：只有在第二次及之后的迭代时，ratio 才可能 != 1
        self._old_actor = copy.deepcopy(self.actor_critic.actor).eval()
        for p in self._old_actor.parameters():
            p.requires_grad = False

        if self._config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(actor_critic.critic)
            self._updates = 0

        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=config.actor["lr"])

        self.desired_kl = 0.01
        self.learning_rate = 1e-3
        self.flatten_mode = False
        
    #def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, history_dim, wm_feature_dim):
    #    self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, history_dim=history_dim,
    #                                  wm_feature_dim = wm_feature_dim, device = self.device)
        
    # -------------------- 工具函数 --------------------
    def _check_numerics(self, tensor, name, extra_info=""):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            print(f"!!! Numerical error detected in {name}:")
            print(f"  NaN count: {nan_count}, Inf count: {inf_count}")
            print(f"  Tensor shape: {tensor.shape}")
            print(f"  Context: {extra_info}")
            indices = torch.where(torch.isnan(tensor) | torch.isinf(tensor))
            for i in range(min(5, len(indices[0]))):
                idx = tuple(indices[j][i] for j in range(len(indices)))
                print(f"  Bad value at {idx}: {tensor[idx].item()}")
            raise ValueError(f"Numerical error in {name}")
        max_val = tensor.max().item(); min_val = tensor.min().item()
        mean_val = tensor.mean().item(); std_val = tensor.std().item()
        print(f"[Numerics] {name}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f} {extra_info}")

    @torch.no_grad()
    def _standardize(self, x):
        # 按批次标准化，避免极端输入冲击 critic
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + 1e-6)

    def _normal_log_prob(self, action, mean, std):
        # 计算给定高斯参数下对 actions 的 log_prob，逐维相加
        var = std ** 2 + 1e-8
        log_scale = torch.log(std + 1e-8)
        return -0.5 * (((action - mean) ** 2) / var + 2 * log_scale + np.log(2 * np.pi)).sum(dim=-1, keepdim=True)

    def _actor_forward(self, actor_module, obs, hist_flat, wm_feature):
        # 兼容你的 ActorCriticRWM 接口：拼接后过 actor，返回 mean/std
        # 这里假设 actor.forward(concat) -> (mean, std)，若你的实现不同，请在此适配
        concat = torch.cat((obs, hist_flat, wm_feature), dim=-1)
        out = actor_module(concat)
        if isinstance(out, tuple) and len(out) == 2:
            mean, std = out
        else:
            # 若 actor 直接输出均值，log_std 为可学习参数（常见做法）
            mean = out
            log_std = getattr(actor_module, 'log_std', None)
            if log_std is None:
                raise RuntimeError("actor_module需返回(mean,std)或包含log_std参数")
            std = torch.exp(log_std).expand_as(mean)
        return mean, std

    #def _batch_generator(self, num_mini_batches, num_epochs=8):
    #    batch_size = self._config.batch_size * self._config.imag_start_batch * self._config.horizon
    #    mini_batch_size = batch_size // num_mini_batches
    #    indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self._config.device)

        #observations = self.observations.flatten(0, 1)
        #if self.privileged_observations is not None:
        #    critic_observations = self.privileged_observations.flatten(0, 1)
        #else:
        #    critic_observations = observations

        #actions = self.actions.flatten(0, 1)
        #values = self.values.flatten(0, 1)
        #returns = self.returns.flatten(0, 1)
        #old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        #advantages = self.advantages.flatten(0, 1)
        #old_mu = self.mu.flatten(0, 1)
        #old_sigma = self.sigma.flatten(0, 1)

        #history = self.history.flatten(0, 1)
        #wm_feature = self.wm_features.flatten(0, 1)

        #for epoch in range(num_epochs):
        #    for i in range(num_mini_batches):
        #        start = i*mini_batch_size
        #        end = (i+1)*mini_batch_size
        #        batch_idx = indices[start:end]
                
                
        
    def _train(self, start, objective):
        metrics = {}
        self._update_slow_target()
        self.actor_critic.train()

        # 生成想象轨迹（来自你原始实现）
        imag_feat, imag_state, imag_obs, obs_without_command, imag_action, action_log_prob, mu, sigma, entropy, phis = self._imagine(
            start, self.actor_critic, self._config.imag_horizon
        )
        
        #num_mini_batches = getattr(self, 'num_mini_batches', None) or getattr(self._config, 'num_mini_batches', None) or 32
        #num_learning_epochs = getattr(self, 'num_learning_epochs', None) or getattr(self._config, 'num_learning_epochs', None) or 4
        
        #print("imag_feat shape:", imag_feat.shape)
        # 奖励 & 目标
        reward = objective(imag_feat, imag_state, imag_action)
        self._check_numerics(reward, "reward", "after objective")
        target, weights, value_boot = self._compute_target(imag_feat, imag_obs, reward, phis=phis[:-1])
        self._check_numerics(target, "target", "after compute_target")
        self._check_numerics(weights, "weights", "after compute_target")
        self._check_numerics(value_boot, "value", "after compute_target")

        # ===== 计算想象轨迹上的 value 与 bootstrap（与 PPO 对齐） =====
        # v_t：评估在时刻 t 的状态（想象）
        # imag_obs is [obs_0, ..., obs_{horizon-1}], so imag_obs[:-1] is [obs_0, ..., obs_{horizon-2}]
        # phis is [phi_0, ..., phi_horizon], so phis[:-2] is [phi_0, ..., phi_{horizon-2}]
        v_t   = self.actor_critic.evaluate(imag_obs[:-1], imag_feat[:-1], phi=phis[:-2])    # [T-1,B,1]
        # imag_obs[-1:] is [obs_{horizon-1}], so phi should be [phi_{horizon-1}]
        v_T   = self.actor_critic.evaluate(imag_obs[-1:].detach(), imag_feat[-1:].detach(), phi=phis[-2:-1].detach())  # [1,B,1] 作为 bootstrap

        # r_t：与 PPO 对齐（时刻 t 的 reward 发生在 action_t 之后）
        r_t = reward[1:]                                                     # [T-1,B,1]

        # step 折扣（可选）：若你有连续性头 cont，按 Dreamer 的风格用它来当“非终止概率”
        if "cont" in self._world_model.heads:
            cont = self._world_model.heads["cont"](imag_feat).mean           # [T,B,1]
            cont_t = cont[1:]                                                # 对齐 r_t 的 T-1 步
            discount_step = cont_t                                           # 交给 _gae_returns 里与 gamma 相乘
        else:
            discount_step = None

        # ===== 用与 PPO 一致的 GAE 计算 returns / advantages =====
        returns, advantages = self._gae_returns(r_t.detach(), v_t.detach(), v_T.detach(), discount_step)

        # ===== 跟 PPO 一样做优势标准化（建议按整个 batch 维度标准化）=====
        adv_mean = advantages.mean()
        adv_std  = advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        advantages = advantages.squeeze(-1)  # [B,T-1]

        ## GAE/优势（此处按你原逻辑）
        #target = target.permute(1, 0, 2)   # [B,T-1,1]
        #returns = target
        #advantages = (returns - value_boot)
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        #advantages = advantages.squeeze(-1)  # [B,T-1]
        self._check_numerics(advantages, "advantages", "after normalize")

        # -------- PPO ratio：用冻结的旧策略计算 old_log_prob --------
        # 准备和 imagine 时相同的输入（注意对齐 T 维度）
        T, B, _ = imag_obs.shape  # [T+1, B, D_obs+] 你上面传入evaluate时也是[:-1]
        hist_len = 5
        # 还原 imagine 中用于 actor 的 hist_flat，保持和 _imagine 一致
        # 这里复用 obs_without_command 来构造历史
        history = []
        for t in range(1, T):
            start_t = max(0, t - hist_len)
            h = obs_without_command[start_t:t, :, :]  # [len,B,F]
            if h.shape[0] < hist_len:
                pad = torch.zeros(hist_len - h.shape[0], h.shape[1], h.shape[2], device=h.device)
                h = torch.cat([pad, h], dim=0)
            history.append(h)
        history = torch.stack(history, dim=0)              # [T-1,len,B,F]
        history = history.permute(1, 0, 2, 3).reshape(T-1, B, -1)  # [T-1,B,len*F]

        obs_t = imag_obs[:-1]            # [T-1,B,D_obs_for_critic]（含pri/act/height_map等）
        # 但 actor 用的是 decoder(obs) 而非 critic_obs，这里需要与你 _imagine 中 actor 的真实输入一致。
        # 因为 _imagine 里 actor 用的是：obs(decoder prop.mean)、hist_flat、wm_feature
        # 因此我们需要从 imag_feat 推 wm_feature，再通过 decoder 得到 obs。
        # imag_feat 的形状在 _imagine 返回的是 wm_feature（第二个返回是 states），这里直接使用 imag_feat[:-1] 即 wm_feature[:-1]
        wm_feature_t = imag_feat[:-1]    # [T-1,B,D_wm]

        # 通过旧策略计算 old_log_prob（与当前动作对齐）
        with torch.no_grad():
            # 旧策略必须基于与采样动作相同的输入来评估对数概率
            dynamics = self._world_model.dynamics
            def get_prop_from_state(state):
                feat = dynamics.get_feat(state)
                return self._world_model.heads["decoder"](feat)["prop"].mean()

            # 对齐到 T-1 个动作时刻
            Tm1 = T - 1

            states_t = {k: v[:-1] for k, v in imag_state.items()}  # [T-1,B,...]
            obs_actor_list = []
            for t_idx in range(states_t["stoch"].shape[0]):
                s_t = {k: states_t[k][t_idx] for k in states_t}
                obs_actor_list.append(get_prop_from_state(s_t))
            obs_actor = torch.stack(obs_actor_list, dim=0)  # [T-1,B,obs_dim_for_actor]

            # 临时替换为旧策略，走一遍 act → get_actions_log_prob 流程
            saved_actor = self.actor_critic.actor
            self.actor_critic.actor = self._old_actor
            old_logprobs = []
            try:
                for t_idx in range(obs_actor.shape[0]):
                    _ = self.actor_critic.act(obs_actor[t_idx], history[t_idx], wm_feature_t[t_idx], phi=phis[:-2][t_idx], imagine=True)
                    lp = self.actor_critic.get_actions_log_prob(imag_action[t_idx])  # [B,1] 或 [B]
                    if lp.dim() == 1:
                        lp = lp.unsqueeze(-1)
                    old_logprobs.append(lp)
                old_action_log_prob = torch.stack(old_logprobs, dim=0)  # 期望形状 [T-1,B,1]
            finally:
                self.actor_critic.actor = saved_actor
        old_action_log_prob = action_log_prob[:-1].detach()
        old_mu = mu[:-1].detach()
        old_sigma = sigma[:-1].detach()
        # 当前策略的 log_prob 使用 mu/sigma 计算，避免来自 action_log_prob 的维度不一致
        #curr_action_log_prob = self._normal_log_prob(imag_action[:-1], mu[:-1], sigma[:-1])  # 期望形状 [T-1,B,1]
        curr_action_log_prob = self.actor_critic.get_actions_log_prob(imag_action[:-1])
        curr_mu = self.actor_critic.action_mean
        curr_sigma = self.actor_critic.action_std
        
        # --- 统一使用 batch-major 形状 [T-1, B, 1] 进行后续计算，减少 permute/reshape 错误
        def to_batch_major(x, name):
            # 支持 x 为 [T-1,B,1] 或 [B,T-1,1] 或扁平的 [N,1]
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            if x.dim() != 3:
                raise RuntimeError(f"{name} unexpected rank {x.dim()}, got shape {tuple(x.shape)}")
            # 已经是 time-major [T-1,B,1]
            if x.shape[0] == Tm1 and x.shape[1] == B:
                return x
            # 已经是 batch-major [B,T-1,1]
            if x.shape[0] == B and x.shape[1] == Tm1:
                return x.permute(1, 0, 2)
            # 尝试以总元素数重排为 [B,T-1,-1]
            if x.numel() == B * Tm1 * x.shape[-1]:
                return x.reshape(B, Tm1, -1)
            raise RuntimeError(f"{name} shape {tuple(x.shape)} cannot be aligned to (B={B}, T-1={Tm1}, 1)")

        curr_log_b = to_batch_major(curr_action_log_prob, 'curr_action_log_prob')
        old_log_b = to_batch_major(old_action_log_prob, 'old_action_log_prob')
        self._check_numerics(curr_log_b, "curr_log_b", "curr_log_b")
        self._check_numerics(old_log_b, "old_log_b", "old_log_b")

        # batch-major ratio: [B,T-1,1]
        ratio_b = torch.exp(curr_log_b - old_log_b)
        self._check_numerics(ratio_b, "ratio", "batch-major before surrogate loss")

        # advantages: originally [B, T-1] -> [B, T-1, 1]
        adv_b = advantages.unsqueeze(-1)

        # weights: 标量或向量 -> 扩展为 [T-1,B,1]
        w = weights[:-1]
        if w.dim() == 1:
            weights_b = w.view(1, Tm1, 1).expand(B, Tm1, 1)
        elif w.dim() == 2:
            # 可能为 [T-1, B] 或 [B, T-1]
            if w.shape[0] == Tm1 and w.shape[1] == B:
                weights_b = w.unsqueeze(-1)
            elif w.shape[0] == B and w.shape[1] == Tm1:
                weights_b = w.permute(1, 0).unsqueeze(-1)
            else:
                weights_b = w.reshape(1, Tm1, -1).expand(B, Tm1, -1)
        elif w.dim() == 3:
            if w.shape[0] == Tm1 and w.shape[1] == B:
                weights_b = w
            elif w.shape[0] == B and w.shape[1] == Tm1:
                weights_b = w.permute(1, 0, 2)
            else:
                weights_b = w.permute(1, 0, 2)
        else:
            raise RuntimeError(f"Unsupported weights dim: {w.dim()} with shape {tuple(w.shape)}")

        # 计算 surrogate loss（batch-major）
        if torch.allclose(ratio_b.mean(), torch.ones_like(ratio_b.mean()), atol=1e-6):
            surr1 = -adv_b
            surr2 = -adv_b
        else:
            surr1 = ratio_b * (-adv_b)
            surr2 = torch.clamp(ratio_b, 1.0 - self._config.ppo_clip, 1.0 + self._config.ppo_clip) * (-adv_b)
        self._check_numerics(surr1, "surr1", "surrogate loss 1 batch-major")
        self._check_numerics(surr2, "surr2", "surrogate loss 2 batch-major")
        # actor_loss 按 batch-major 聚合
        surrogate_loss = torch.max(surr1, surr2).mean()
        actor_loss = surrogate_loss
        #actor_loss = torch.max(surr1, surr2) * weights_b
        #actor_loss = -actor_loss.mean()


        # 熵正则（按维度求和→时间/批次求均值）
        #entropy_term = 0.5 * (torch.log(2 * np.pi * sigma[:-1]**2 + 1e-8) + 1)
        #entropy_term = entropy_term.sum(dim=-1).mean()
        entropy_term = self.actor_critic.entropy.mean()
        self._check_numerics(entropy_term, "entropy", "entropy loss")
        actor_loss = actor_loss - self._config.entropy_coef * entropy_term

        # 线速度预测损失（与你原逻辑一致）
        ac_unwrapped = self.actor_critic.module if hasattr(self.actor_critic, 'module') else self.actor_critic
        predicted_linear_vel = self.actor_critic.get_linear_vel(imag_obs[:-1], history, imagine=True)
        priv_dim = getattr(ac_unwrapped, 'privileged_dim', 69)
        target_linear_vel = imag_obs[:-1, :, priv_dim-3:priv_dim]
        vel_predict_loss = (predicted_linear_vel - target_linear_vel).pow(2).mean()
        self._check_numerics(predicted_linear_vel, "predicted_linear_vel", "vel_predict_loss")
        actor_loss = actor_loss + self._config.vel_predict_coef * vel_predict_loss

        # KL 整流 + 自适应lr（与你原逻辑一致）
        #kl = torch.sum(
        #    torch.log(sigma / (sigma.detach() + 1e-5) + 1e-5)
        #    + (sigma.detach().pow(2) + (mu.detach() - mu).pow(2)) / (2.0 * sigma.pow(2) + 1e-8)
        #    - 0.5,
        #    dim=-1,
        #)
        kl = torch.sum(
            torch.log(curr_sigma.detach() / (old_sigma + 1e-5) + 1e-5)
            + (old_sigma.pow(2) + (old_mu - curr_mu.detach()).pow(2)) / (2.0 * curr_sigma.detach().pow(2) + 1e-8)
            - 0.5,
            dim=-1,
        )
        kl_mean = kl.mean()
        self._check_numerics(kl_mean, "kl", "kl loss")
        if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        # --------- Critic 训练 ---------
        value_pred = self.actor_critic.evaluate(imag_obs[:-1], imag_feat[:-1], phi=phis[:-2])  # [T-1,B,1]

        ## 批次标准化输入（仅用于 critic，提升稳定性）
        #imag_obs_std = self._standardize(imag_obs[:-1].reshape(-1, imag_obs.shape[-1])).reshape_as(imag_obs[:-1])
        #wm_feat_std = self._standardize(imag_feat[:-1].reshape(-1, imag_feat.shape[-1])).reshape_as(imag_feat[:-1])
        #value_pred = self.actor_critic.evaluate(imag_obs_std, wm_feat_std)

        if self._config.use_clipped_value_loss:
            value_clipped = returns + (value_pred - returns).clamp(-self._config.clip_param, self._config.clip_param)
            value_losses = (value_pred - returns).pow(2)
            value_losses_clipped = (value_clipped - returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (value_pred - returns).pow(2).mean()
        self._check_numerics(value_loss, "value_loss", "before slow_target")

        if self._config.critic["slow_target"]:
            wm_latent_vector = self.actor_critic.critic_wm_feature_encoder(imag_feat[:-1])
            if getattr(self._config, 'phase_model', False):
                concat_obs = torch.cat((imag_obs[:-1], wm_latent_vector, phis[:-2]), dim=-1)
            else:
                concat_obs = torch.cat((imag_obs[:-1], wm_latent_vector), dim=-1)
            slow_value = self._slow_value(concat_obs)
            slow_value_loss = (value_pred - slow_value).pow(2).mean()
            value_loss = value_loss + slow_value_loss

        value_loss = (weights[:-1] * value_loss).mean() if value_loss.dim() > 0 else value_loss
        self._check_numerics(value_loss, "value_loss", "after slow_target")

        # --------- 反向传播与更新 ---------
        self.optimizer.zero_grad()
        total_loss = actor_loss + self.value_loss_coef * value_loss
        self._check_numerics(total_loss, "total_loss", "before backward")
        total_loss.backward()

        # 单独裁剪 critic 梯度
        nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.critic_grad_clip)
        # 全局裁剪
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        # 监控异常梯度
        for name, param in self.actor_critic.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1000:
                    print(f"!!! Exploding gradient in {name}: {grad_norm}")
                    raise RuntimeError(f"Exploding gradient in {name}: {grad_norm}")
                if torch.isnan(param.grad).any():
                    print(f"!!! NaN gradient in {name}")

        self.optimizer.step()

        # 成功更新后，同步旧策略参数，用于下一轮 ratio 计算
        with torch.no_grad():
            for p_old, p_new in zip(self._old_actor.parameters(), self.actor_critic.actor.parameters()):
                p_old.copy_(p_new)

        # 记录指标
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_loss"] = actor_loss.item()
        metrics["value_loss"] = value_loss.item()
        metrics["total_loss"] = total_loss.item()

        for p in self.actor_critic.parameters():
            p.requires_grad = True

        return imag_feat, imag_state, imag_action, weights, metrics

    # -------------- 下方保持你的原实现（除非特别标注） --------------
    def _imagine(self, start, actor_critic, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        B = start["stoch"].shape[0]
        #print("B: ", B)
        history_len = 5
        update_interval = 5
        
        # Robustly get history dimension from actor_critic
        ac_unwrapped = self.actor_critic.module if hasattr(self.actor_critic, 'module') else self.actor_critic
        hist_step_dim = ac_unwrapped.history_encoder[0].in_features // history_len
        self.hist = torch.zeros((B, history_len, hist_step_dim), device=start["stoch"].device)
        
        # action history for RSSM uses 19-dim actions if phase_model is True
        wm_action_dim = 19 if getattr(self._config, 'phase_model', False) else 18
        self.action_history = torch.zeros((B, update_interval, wm_action_dim), device=start["stoch"].device)
        
        current_phi = torch.zeros((B, 1), device=start["stoch"].device)
        dt = getattr(self._config, 'dt', 0.02)
        omega_max = getattr(self._config, 'omega_max', 12.566370614359172)

        def step(prev, _):
            state, _, _, _, _, _, _, _, _, phi = prev
            feat = dynamics.get_feat(state)
            wm_feature = dynamics.get_deter_feat(state)
            
            # If phase_model, concat phi to feat for decoder
            if getattr(self._config, 'phase_model', False):
                feat_for_dec = torch.concat((feat, phi), dim=-1)
            else:
                feat_for_dec = feat
                
            dec = self._world_model.heads["decoder"](feat_for_dec)
            obs = dec["prop"].mean()
            pri_obs = dec["privileged_obs"].mean()
            height_map = dec["height_map"].mean()
            hist_flat = self.hist.flatten(1)
            
            if getattr(self._config, 'phase_model', False):
                action = actor_critic.act(obs, hist_flat, wm_feature, phi=phi, imagine=True)
            else:
                action = actor_critic.act(obs, hist_flat, wm_feature, imagine=True)
                
            action_log_prob = actor_critic.get_actions_log_prob(action)
            mu = actor_critic.action_mean
            sigma = actor_critic.action_std
            entropy = actor_critic.entropy

            # Use full action for RSSM action history if phase_model is True
            action_for_wm = action if getattr(self._config, 'phase_model', False) else action[:, :18]
            self.action_history = torch.cat([self.action_history[:, 1:], action_for_wm.unsqueeze(1)], dim=1)

            # Use 18-dim action for critic_obs and actor history
            action_for_env = action[:, :18]
            critic_obs = torch.cat((pri_obs, obs, action_for_env, height_map), dim=-1)
            obs_without_command = torch.cat((
                obs[:,  : 6],
                obs[:, 9 : ], action_for_env
            ), dim=-1)
            self.hist = torch.cat([self.hist[:, 1:], obs_without_command.unsqueeze(1)], dim=1)

            action_flat = self.action_history.flatten(1)
            
            # Update phi for next step
            if getattr(self._config, 'phase_model', False):
                raw_omega = action[:, -1:]
                omega = omega_max + torch.tanh(raw_omega)
                next_phi = phi + omega * dt
                # Also RSSM update might need phi if we added it to action_flat
                # But here action_flat is used for transition. 
                # Does transition depend on phi? The user said "RSSM...新加入phi作为输入"
                # If we want phi to affect transitions in imagine, we should concat it to action_flat.
                action_flat = torch.concat((action_flat, phi), dim=-1)
            else:
                next_phi = phi

            succ = dynamics.img_step(state, action_flat)
            return succ, wm_feature, critic_obs, obs_without_command, action, action_log_prob, mu, sigma, entropy, next_phi

        succ, feats, critic_obs, obs_without_command, actions, action_log_prob, mu, sigma, entropy, phis = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None, None, None, None, None, None, None, current_phi)
        )
        # phis contains [phi_1, ..., phi_horizon]
        # Prepend current_phi to get [phi_0, ..., phi_{horizon-1}]
        # Actually we might need all of them up to phi_horizon for bootstrap
        phis = torch.cat([current_phi[None], phis], 0)
        
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        return feats, states, critic_obs, obs_without_command, actions, action_log_prob, mu, sigma, entropy, phis

    def _compute_target(self, imag_feat, imag_obs, reward, phis=None):
        if "cont" in self._world_model.heads:
            discount = self._config.discount * self._world_model.heads["cont"](imag_feat).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)

        value = self.actor_critic.evaluate(imag_obs, imag_feat, phi=phis)
        target = tools.lambda_return(
            reward[1:], value[:-1], discount[1:], bootstrap=value[-1],
            lambda_=self._config.discount_lambda, axis=0
        )
        target = torch.stack(target, dim=0)
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]
    
    def _gae_returns(self, rewards, values, last_value, discount):
        """
        与 RolloutStorage.compute_returns 完全一致的 GAE/returns 计算（time-major）。
        参数:
        rewards:     [T-1, B, 1]    —— 对应时刻 t 的 reward (此处我们会用 reward[1:])
        values:      [T-1, B, 1]    —— v_t，评估于 imag_obs[:-1]
        last_value:  [1,   B, 1]    —— v_T，评估于 imag_obs[-1]
        discount:    [T-1, B, 1]    —— 每步折扣因子（若没有 cont head 就用常数 gamma）

        返回:
        returns:     [T-1, B, 1]
        advantages:  [T-1, B, 1]
        """
        Tm1 = rewards.shape[0]
        device = rewards.device

        # 拼接 next_values: v_{t+1}，最后一个用 bootstrap v_T
        next_values = torch.cat([values[1:], last_value], dim=0)              # [T-1,B,1]
        gamma = self.gamma                                                     # 与 PPO 一致
        lam   = self.lam

        # 如果传入的是模型的 cont 概率，先把它转成 “not_done” 意义的 step 折扣（等价于 PPO 里的 next_is_not_terminal）
        # 写法：discount_step = gamma * cont_t；若没有 cont，用常数 gamma
        if discount is None:
            discount_step = torch.full_like(rewards, gamma)
        else:
            discount_step = gamma * discount

        # 与 RolloutStorage 相同的递推
        advantages = torch.zeros_like(rewards, device=device)
        gae = torch.zeros((rewards.shape[1], 1), device=device)               # [B,1]
        for t in reversed(range(Tm1)):
            delta = rewards[t] + discount_step[t] * next_values[t] - values[t]
            gae = delta + discount_step[t] * lam * gae
            advantages[t] = gae

        returns = advantages + values
        return returns, advantages

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.actor_critic.critic.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
