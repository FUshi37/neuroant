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
    RealRobotRWMInference,
    compute_leg_errors,
    detect_stuck_leg,
)
from deployment_safety import BadLegTracker, RiskLevelEstimator, WorldModelCandidateSelector


def _to_float_list(arr):
    return [float(x) for x in np.asarray(arr, dtype=np.float32).reshape(-1)]


def run_server(
    host,
    port,
    model_path,
    remove_dof_vel=False,
    log_every=50,
    use_stability_filter=True,
    filter_debug_path=None,
    log_contact_anomaly=True,
):
    print(f"[PC] starting WM server at {host}:{port}")
    inference = RealRobotRWMInference(model_path, device="cpu", remove_dof_vel=remove_dof_vel)
    policy = inference.get_inference_policy()
    candidate_selector = (
        WorldModelCandidateSelector(
            world_model=inference.world_model,
            action_dim=18,
            horizon=4,
            max_lift_candidates=6,
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
    risk_estimator = RiskLevelEstimator()
    bad_leg_tracker = BadLegTracker()
    if candidate_selector is None:
        print("[PC][WARNING] WM candidate selector is DISABLED. Risk can only tighten the final rate limiter; no lift candidates will be selected.")
    else:
        print("[PC] WM candidate selector ENABLED: risk-coupled candidates and lift recovery are active.")
        print(
            "[PC] Lift recovery config: "
            f"hold_steps={candidate_selector.recovery_hold_steps}, "
            f"hip_target={candidate_selector.lift_hip_forward_target:.3f}, "
            f"knee_target={candidate_selector.lift_knee_target:.3f}, "
            f"ankle_target={candidate_selector.lift_ankle_target:.3f}"
        )

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
            "candidate_selected",
            "candidate_group",
            "candidate_score",
            "num_candidates",
            "risk_level",
            "risk_ema",
            "detected_bad_leg",
            "bad_leg",
            "effective_bad_leg",
            "effective_risk_level",
            "forced_lift",
            "forced_lift_reason",
            "recovery_latch_active",
            "recovery_latch_leg",
            "recovery_latch_hold",
            "recovery_latch_triggered",
            "recovery_latch_accepted",
            "lift_hip_forward_target",
            "lift_hip_forward_blend",
            "lift_knee_final_target",
            "lift_knee_raw_target",
            "lift_knee_blend",
            "lift_ankle_final_target",
            "lift_ankle_raw_target",
            "lift_ankle_blend",
            "horizon",
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
    last_event_key = None
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
                    detected_bad_leg = None
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

                    if (
                        contact_detector is not None
                        and anomaly_debug.get("enabled")
                        and int(anomaly_debug.get("steps", 0)) >= 1
                    ):
                        obs_real_for_leg = contact_detector.last_obs_real
                        obs_pred_for_leg = contact_detector.last_obs_pred
                        if (obs_real_for_leg is not None) and (obs_pred_for_leg is not None):
                            leg_errors = compute_leg_errors(obs_real_for_leg, obs_pred_for_leg)
                            detected_bad_leg = detect_stuck_leg(leg_errors, threshold=CONTACT_ANOMALY_THRESHOLD)
                    stable_bad_leg = bad_leg_tracker.update(
                        detected_bad_leg,
                        active=int(anomaly_debug.get("steps", 0)) >= 1,
                    )

                    risk_state = risk_estimator.update(
                        wm_error=float(anomaly_debug.get("error")) if isinstance(anomaly_debug.get("error"), (float, int)) else 0.0,
                        contact_anomaly=bool(anomaly_debug.get("is_anomaly", False)),
                        contact_steps=int(anomaly_debug.get("steps", 0)),
                        bad_leg=stable_bad_leg,
                    )

                    if candidate_selector is not None and inference.wm_latent is not None:
                        try:
                            prev_action_t = (
                                None
                                if prev_action is None
                                else torch.from_numpy(prev_action).unsqueeze(0)
                            )
                            actions = candidate_selector.select(
                                prev_latent=inference.wm_latent,
                                is_first=inference.wm_is_first,
                                action_nominal=actions,
                                prev_action=prev_action_t,
                                risk_state=risk_state,
                            )
                            filter_debug = dict(getattr(candidate_selector, "last_debug", filter_debug))
                        except Exception as e:
                            # Keep control alive if imagination branch fails.
                            filter_debug = {"used": False, "error": str(e)}
                            print(f"[PC] stability filter disabled for this step: {e}")

                action_raw = actions.detach().cpu().numpy().reshape(-1)
                action_raw = np.clip(action_raw, -1.0, 1.0)
                server_ms = float((time.perf_counter() - t0) * 1000.0)

                resp = {
                    "type": "act",
                    "step": step,
                    "action_raw": _to_float_list(action_raw),
                    "server_ms": server_ms,
                    "ok": True,
                    "contact_anomaly": bool(anomaly_debug.get("is_anomaly", False)),
                    "contact_error": float(anomaly_debug.get("error")) if isinstance(anomaly_debug.get("error"), (float, int)) else 0.0,
                    "contact_steps": int(anomaly_debug.get("steps", 0)),
                    "risk_level": int(risk_state.level),
                    "risk_ema": float(risk_state.ema_error),
                    "risk_baseline_mean": float(risk_state.baseline_mean),
                    "risk_baseline_std": float(risk_state.baseline_std),
                    "detected_bad_leg": int(-1 if detected_bad_leg is None else detected_bad_leg),
                    "bad_leg": int(risk_state.bad_leg),
                    "effective_bad_leg": int(filter_debug.get("effective_bad_leg", risk_state.bad_leg)),
                    "effective_risk_level": int(filter_debug.get("effective_risk_level", risk_state.level)),
                    "candidate_selected": str(filter_debug.get("selected", "nominal")),
                    "candidate_group": str(filter_debug.get("selected_group", "")),
                    "candidate_score": float(filter_debug.get("score", 0.0)) if isinstance(filter_debug.get("score", 0.0), (float, int)) else 0.0,
                    "forced_lift": bool(filter_debug.get("forced_lift", False)),
                    "forced_lift_reason": str(filter_debug.get("forced_lift_reason", "")),
                    "recovery_latch_active": bool(filter_debug.get("recovery_latch_active", False)),
                    "recovery_latch_leg": int(filter_debug.get("recovery_latch_leg", -1)),
                    "recovery_latch_hold": int(filter_debug.get("recovery_latch_hold", 0)),
                    "recovery_latch_triggered": bool(filter_debug.get("recovery_latch_triggered", False)),
                    "recovery_latch_accepted": bool(filter_debug.get("recovery_latch_accepted", False)),
                    "lift_hip_forward_target": float(filter_debug.get("lift_hip_forward_target", 0.0)) if isinstance(filter_debug.get("lift_hip_forward_target", 0.0), (float, int)) else 0.0,
                    "lift_hip_forward_blend": float(filter_debug.get("lift_hip_forward_blend", 0.0)) if isinstance(filter_debug.get("lift_hip_forward_blend", 0.0), (float, int)) else 0.0,
                    "lift_knee_final_target": float(filter_debug.get("lift_knee_final_target", 0.0)) if isinstance(filter_debug.get("lift_knee_final_target", 0.0), (float, int)) else 0.0,
                    "lift_knee_raw_target": float(filter_debug.get("lift_knee_raw_target", 0.0)) if isinstance(filter_debug.get("lift_knee_raw_target", 0.0), (float, int)) else 0.0,
                    "lift_knee_blend": float(filter_debug.get("lift_knee_blend", 0.0)) if isinstance(filter_debug.get("lift_knee_blend", 0.0), (float, int)) else 0.0,
                    "lift_ankle_final_target": float(filter_debug.get("lift_ankle_final_target", 0.0)) if isinstance(filter_debug.get("lift_ankle_final_target", 0.0), (float, int)) else 0.0,
                    "lift_ankle_raw_target": float(filter_debug.get("lift_ankle_raw_target", 0.0)) if isinstance(filter_debug.get("lift_ankle_raw_target", 0.0), (float, int)) else 0.0,
                    "lift_ankle_blend": float(filter_debug.get("lift_ankle_blend", 0.0)) if isinstance(filter_debug.get("lift_ankle_blend", 0.0), (float, int)) else 0.0,
                }
                packet_count += 1

                row = {
                    "step": step,
                    "filter_enabled": int(candidate_selector is not None),
                    "filter_used": int(bool(filter_debug.get("used", False))),
                    "candidate_selected": filter_debug.get("selected", ""),
                    "candidate_group": filter_debug.get("selected_group", ""),
                    "candidate_score": filter_debug.get("score", ""),
                    "num_candidates": filter_debug.get("num_candidates", ""),
                    "risk_level": int(risk_state.level),
                    "risk_ema": float(risk_state.ema_error),
                    "detected_bad_leg": int(-1 if detected_bad_leg is None else detected_bad_leg),
                    "bad_leg": int(risk_state.bad_leg),
                    "effective_bad_leg": filter_debug.get("effective_bad_leg", risk_state.bad_leg),
                    "effective_risk_level": filter_debug.get("effective_risk_level", risk_state.level),
                    "forced_lift": int(bool(filter_debug.get("forced_lift", False))),
                    "forced_lift_reason": filter_debug.get("forced_lift_reason", ""),
                    "recovery_latch_active": int(bool(filter_debug.get("recovery_latch_active", False))),
                    "recovery_latch_leg": filter_debug.get("recovery_latch_leg", -1),
                    "recovery_latch_hold": filter_debug.get("recovery_latch_hold", 0),
                    "recovery_latch_triggered": int(bool(filter_debug.get("recovery_latch_triggered", False))),
                    "recovery_latch_accepted": int(bool(filter_debug.get("recovery_latch_accepted", False))),
                    "lift_hip_forward_target": filter_debug.get("lift_hip_forward_target", ""),
                    "lift_hip_forward_blend": filter_debug.get("lift_hip_forward_blend", ""),
                    "lift_knee_final_target": filter_debug.get("lift_knee_final_target", ""),
                    "lift_knee_raw_target": filter_debug.get("lift_knee_raw_target", ""),
                    "lift_knee_blend": filter_debug.get("lift_knee_blend", ""),
                    "lift_ankle_final_target": filter_debug.get("lift_ankle_final_target", ""),
                    "lift_ankle_raw_target": filter_debug.get("lift_ankle_raw_target", ""),
                    "lift_ankle_blend": filter_debug.get("lift_ankle_blend", ""),
                    "horizon": 4,
                    "server_ms": server_ms,
                    "action_min": float(action_raw.min()),
                    "action_max": float(action_raw.max()),
                    "action_mean": float(action_raw.mean()),
                    "error": filter_debug.get("error", ""),
                }
                debug_writer.writerow(row)
                event_key = (
                    int(risk_state.level),
                    bool(anomaly_debug.get("is_anomaly", False)),
                    int(risk_state.bad_leg),
                    str(row["candidate_selected"]),
                )
                event_active = (
                    risk_state.level > 0
                    or bool(anomaly_debug.get("is_anomaly", False))
                    or str(row["candidate_group"]) in {"lift", "ankle_protected", "scaled", "blend"}
                )
                should_print = (
                    packet_count <= 5
                    or (packet_count % max(1, int(log_every))) == 0
                    or (event_active and event_key != last_event_key)
                )
                if should_print:
                    last_event_key = event_key
                    debug_file.flush()
                    print(
                        "[PC] "
                        f"from={addr[0]}:{addr[1]} step={step} "
                        f"act[min={action_raw.min():.4f}, max={action_raw.max():.4f}, mean={action_raw.mean():.4f}] "
                        f"server={server_ms:.2f}ms "
                        f"filter_used={row['filter_used']} cand={row['candidate_selected']} "
                        f"group={row['candidate_group']} forced_lift={row['forced_lift']} score={row['candidate_score']} "
                        f"latch={row['recovery_latch_active']} latch_leg={row['recovery_latch_leg']} "
                        f"latch_hold={row['recovery_latch_hold']} "
                        f"risk={risk_state.level} "
                        f"contact_enabled={int(anomaly_debug['enabled'])} "
                        f"contact_anomaly={int(anomaly_debug['is_anomaly'])} "
                        f"contact_error={anomaly_debug['error']} "
                        f"contact_steps={anomaly_debug['steps']} "
                        f"det_bad_leg={-1 if detected_bad_leg is None else detected_bad_leg} "
                        f"bad_leg={risk_state.bad_leg} eff_bad_leg={row['effective_bad_leg']}"
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
    parser.add_argument("--use-stability-filter", dest="use_stability_filter", action="store_true", default=True)
    parser.add_argument("--disable-stability-filter", dest="use_stability_filter", action="store_false")
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
