import argparse
import json
import socket
import time

import numpy as np
import torch

from test_rwm_real_robot_wm import (
    ImaginationStabilityFilter,
    RealRobotRWMInference,
    get_action_limits,
)


def _to_float_list(arr):
    return [float(x) for x in np.asarray(arr, dtype=np.float32).reshape(-1)]


def run_server(host, port, model_path, remove_dof_vel=False):
    print(f"[PC] starting WM server at {host}:{port}")
    inference = RealRobotRWMInference(model_path, device="cpu", remove_dof_vel=remove_dof_vel)
    policy = inference.get_inference_policy()
    stability_filter = (
        ImaginationStabilityFilter(
            world_model=inference.world_model,
            action_dim=18,
            horizon=5,
            num_samples=8,
            noise_scale=0.05,
            device="cpu",
        )
        if inference.world_model is not None
        else None
    )
    action_limits = get_action_limits()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(1.0)

    while True:
        try:
            data, addr = sock.recvfrom(1024 * 1024)
        except socket.timeout:
            continue

        t0 = time.perf_counter()
        try:
            msg = json.loads(data.decode("utf-8"))
            if msg.get("type") != "obs":
                continue

            step = int(msg["step"])
            obs = np.asarray(msg["obs"], dtype=np.float32)
            history = np.asarray(msg["history"], dtype=np.float32)
            prev_action = msg.get("prev_action")
            prev_action = None if prev_action is None else np.asarray(prev_action, dtype=np.float32)

            prop_t = torch.from_numpy(obs).unsqueeze(0)
            history_t = torch.from_numpy(history).unsqueeze(0)
            obs_dict = {"prop": prop_t, "is_first": inference.wm_is_first}

            with torch.no_grad():
                wm_feature = inference.update_world_model(obs_dict, prev_action)
                actions = policy(obs_dict["prop"], history_t, wm_feature)
                if stability_filter is not None and inference.wm_latent is not None:
                    prev_action_t = (
                        None
                        if prev_action is None
                        else torch.from_numpy(prev_action).unsqueeze(0)
                    )
                    actions = stability_filter.select_action(
                        obs_prop=obs_dict["prop"],
                        prev_latent=inference.wm_latent,
                        action_nominal=actions,
                        is_first=inference.wm_is_first,
                        prev_action=prev_action_t,
                    )

            action_raw = actions.detach().cpu().numpy().reshape(-1)
            action_raw = np.clip(action_raw, action_limits["min"], action_limits["max"])

            resp = {
                "type": "act",
                "step": step,
                "action_raw": _to_float_list(action_raw),
                "server_ms": float((time.perf_counter() - t0) * 1000.0),
                "ok": True,
            }
        except Exception as e:
            resp = {
                "type": "act",
                "step": int(msg["step"]) if "msg" in locals() and "step" in msg else -1,
                "action_raw": [0.0] * 18,
                "ok": False,
                "error": str(e),
            }

        sock.sendto(json.dumps(resp).encode("utf-8"), addr)


def main():
    parser = argparse.ArgumentParser(description="PC world-model inference UDP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--remove-dof-vel", action="store_true")
    args = parser.parse_args()
    run_server(args.host, args.port, args.model_path, remove_dof_vel=args.remove_dof_vel)


if __name__ == "__main__":
    main()
