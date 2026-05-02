import argparse
import csv
import json
import os
import socket
import shutil
import time

import numpy as np
import torch

# On Windows hosts without MSVC, force eager mode before importing WM modules.
if os.name == "nt":
    has_cl = (shutil.which("cl") is not None) or (shutil.which("cl.exe") is not None)
    if not has_cl:
        os.environ["DISABLE_TORCH_COMPILE"] = "1"
        print("[PC] cl.exe not found, forcing eager mode (DISABLE_TORCH_COMPILE=1)")

from test_rwm_real_robot_wm import (
    CONTACT_ANOMALY_EMA_ALPHA,
    CONTACT_ANOMALY_THRESHOLD,
    CONTACT_ANOMALY_TRIGGER_COUNT,
    ContactAnomalyDetector,
    ImaginationStabilityFilter,
    RealRobotRWMInference,
    get_action_limits,
)


def _to_float_list(arr):
    return [float(x) for x in np.asarray(arr, dtype=np.float32).reshape(-1)]


def run_server(
    host,
    port,
    model_path,
    remove_dof_vel=False,
    log_every=50,
    use_stability_filter=False,
    filter_debug_path=None,
    log_contact_anomaly=True,
):
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
        if (use_stability_filter and inference.world_model is not None)
        else None
    )
    contact_detector = (
        ContactAnomalyDetector(
            world_model=inference.world_model,
            threshold=CONTACT_ANOMALY_THRESHOLD,
            ema_alpha=CONTACT_ANOMALY_EMA_ALPHA,
            trigger_count=CONTACT_ANOMALY_TRIGGER_COUNT,
            action_dim=18,
            device="cpu",
        )
        if (log_contact_anomaly and inference.world_model is not None)
        else None
    )
    detector_latent = inference.wm_latent
    action_limits = get_action_limits()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(1.0)

    if filter_debug_path is None:
        filter_debug_path = os.path.join("validation_outputs", "filter_debug.csv")
    os.makedirs(os.path.dirname(filter_debug_path) or ".", exist_ok=True)
    debug_file = open(filter_debug_path, "w", newline="")
    debug_writer = csv.DictWriter(
        debug_file,
        fieldnames=[
            "step",
            "filter_enabled",
            "filter_used",
            "best_idx",
            "nominal_score",
            "best_score",
            "improvement",
            "delta",
            "num_samples",
            "horizon",
            "noise_scale",
            "server_ms",
            "action_min",
            "action_max",
            "action_mean",
            "error",
        ],
    )
    debug_writer.writeheader()
    debug_file.flush()
    print(f"[PC] filter debug will be logged to: {filter_debug_path}")

    packet_count = 0
    try:
        while True:
            try:
                data, addr = sock.recvfrom(1024 * 1024)
            except socket.timeout:
                continue

            t0 = time.perf_counter()
            filter_debug = {"used": False, "error": ""}
            anomaly_debug = {"enabled": False, "is_anomaly": False, "error": "", "steps": 0}
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
                    if contact_detector is not None:
                        if detector_latent is None:
                            detector_latent = inference.wm_latent
                        if detector_latent is not None:
                            try:
                                is_anomaly, ema_error, detector_latent, anomaly_steps = contact_detector.detect(
                                    prev_latent=detector_latent,
                                    action=actions,
                                    obs_dict=obs_dict,
                                )
                                anomaly_debug = {
                                    "enabled": True,
                                    "is_anomaly": bool(is_anomaly),
                                    "error": float(torch.max(ema_error).detach().cpu().item()),
                                    "steps": int(anomaly_steps),
                                }
                            except Exception as e:
                                anomaly_debug = {
                                    "enabled": True,
                                    "is_anomaly": False,
                                    "error": f"detect_error:{e}",
                                    "steps": 0,
                                }
                                print(f"[PC] contact anomaly detector error: {e}")
                    if stability_filter is not None and inference.wm_latent is not None:
                        try:
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
                            filter_debug = dict(getattr(stability_filter, "last_debug", filter_debug))
                        except Exception as e:
                            # Keep control alive if imagination branch fails.
                            filter_debug = {"used": False, "error": str(e)}
                            print(f"[PC] stability filter disabled for this step: {e}")

                action_raw = actions.detach().cpu().numpy().reshape(-1)
                action_raw = np.clip(action_raw, action_limits["min"], action_limits["max"])
                server_ms = float((time.perf_counter() - t0) * 1000.0)

                resp = {
                    "type": "act",
                    "step": step,
                    "action_raw": _to_float_list(action_raw),
                    "server_ms": server_ms,
                    "ok": True,
                }
                packet_count += 1

                row = {
                    "step": step,
                    "filter_enabled": int(stability_filter is not None),
                    "filter_used": int(bool(filter_debug.get("used", False))),
                    "best_idx": filter_debug.get("best_idx", ""),
                    "nominal_score": filter_debug.get("nominal_score", ""),
                    "best_score": filter_debug.get("best_score", ""),
                    "improvement": filter_debug.get("improvement", ""),
                    "delta": filter_debug.get("delta", ""),
                    "num_samples": filter_debug.get("num_samples", ""),
                    "horizon": filter_debug.get("horizon", ""),
                    "noise_scale": filter_debug.get("noise_scale", ""),
                    "server_ms": server_ms,
                    "action_min": float(action_raw.min()),
                    "action_max": float(action_raw.max()),
                    "action_mean": float(action_raw.mean()),
                    "error": filter_debug.get("error", ""),
                }
                debug_writer.writerow(row)
                if packet_count <= 5 or (packet_count % max(1, int(log_every))) == 0:
                    debug_file.flush()
                    print(
                        "[PC] "
                        f"from={addr[0]}:{addr[1]} step={step} "
                        f"act[min={action_raw.min():.4f}, max={action_raw.max():.4f}, mean={action_raw.mean():.4f}] "
                        f"server={server_ms:.2f}ms "
                        f"filter_used={row['filter_used']} improve={row['improvement']} delta={row['delta']} "
                        f"contact_enabled={int(anomaly_debug['enabled'])} "
                        f"contact_anomaly={int(anomaly_debug['is_anomaly'])} "
                        f"contact_error={anomaly_debug['error']} "
                        f"contact_steps={anomaly_debug['steps']}"
                    )
            except Exception as e:
                resp = {
                    "type": "act",
                    "step": int(msg["step"]) if "msg" in locals() and "step" in msg else -1,
                    "action_raw": [0.0] * 18,
                    "ok": False,
                    "error": str(e),
                }
                print(f"[PC] inference error: {e}")

            sock.sendto(json.dumps(resp).encode("utf-8"), addr)
    finally:
        debug_file.flush()
        debug_file.close()


def main():
    parser = argparse.ArgumentParser(description="PC world-model inference UDP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--remove-dof-vel", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--use-stability-filter", action="store_true")
    parser.add_argument("--filter-debug-path", default=None)
    parser.add_argument("--no-contact-anomaly-log", action="store_true")
    args = parser.parse_args()
    run_server(
        args.host,
        args.port,
        args.model_path,
        remove_dof_vel=args.remove_dof_vel,
        log_every=args.log_every,
        use_stability_filter=args.use_stability_filter,
        filter_debug_path=args.filter_debug_path,
        log_contact_anomaly=not args.no_contact_anomaly_log,
    )


if __name__ == "__main__":
    main()
