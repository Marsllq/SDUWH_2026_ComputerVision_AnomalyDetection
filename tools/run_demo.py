#!/usr/bin/env python3
"""
Demo 视频生成：支持倍速播放，每帧都显示检测状态。
用法: python tools/run_demo.py taskA 205 50 2

使用 ROI 稳定状态分件，DINOv2 + PatchCore 进行异常检测，
最后输出带仪表盘的可视化视频。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.preprocessing import extract_localized_training_unit_patches, extract_training_frames, extract_training_unit_patches, get_video_info, rotate_rois
from src.preprocessing import crop_rois
from src.detection import AnomalyDetector
from src.locator import crop_dynamic_rois
from src.tracker import UnitTracker
from src.visualization import DashboardRenderer
import cv2
import numpy as np


def score_rois(detector, roi_patches):
    scores = []
    for i, patch in enumerate(roi_patches):
        score, is_ng = detector.score_frame([patch])
        scores.append((i, float(score), bool(is_ng)))
    return scores


def run_demo(task_name, start_s, duration_s, speed=1.0):
    cfg = load_config(task_name)
    video_path = cfg["video_path"]

    # ---- ROI 解析 ---------------------------------------------------------------
    if "rois" in cfg and cfg["rois"]:
        roi_raw = cfg["rois"]
        roi_cfg = [{"x": r[0], "y": r[1], "w": r[2], "h": r[3]} for r in roi_raw]
    elif "fixed_roi" in cfg and cfg["fixed_roi"]:
        roi_cfg = {"x": cfg["fixed_roi"][0], "y": cfg["fixed_roi"][1],
                   "w": cfg["fixed_roi"][2], "h": cfg["fixed_roi"][3]}
    else:
        raise ValueError("No ROI configured")

    rotate = cfg.get("rotate", 0)
    skip_s = cfg.get("skip_duration", 60)
    train_s = cfg.get("train_duration", 120)
    step = cfg.get("step_test", 3)
    info = get_video_info(video_path)
    if cfg.get("rotate_rois_with_frame", False):
        roi_list = roi_cfg if isinstance(roi_cfg, list) else [roi_cfg]
        roi_cfg = rotate_rois(roi_list, int(info["width"]), int(info["height"]), rotate)

    # ---- 训练 -------------------------------------------------------------------
    print(f"  [{task_name}] Training ...")
    use_dynamic = bool(cfg.get("dynamic_localization", False))
    if use_dynamic:
        train_patches = extract_localized_training_unit_patches(
            video_path, roi_cfg if isinstance(roi_cfg, list) else [roi_cfg], task_name, cfg,
            skip_s=skip_s, train_s=train_s,
            step=cfg.get("training_unit_step", 3), rotate=rotate,
            blur_threshold=cfg.get("blur_threshold", 80),
            motion_threshold=cfg.get("motion_threshold", 6),
            min_foreground_ratio=cfg.get("min_foreground_ratio", 0.015),
            foreground_threshold=cfg.get("foreground_threshold", 22),
            min_stable_frames=cfg.get("min_stable_frames", 3),
            end_gap_frames=cfg.get("end_gap_frames", 3),
            max_unit_frames=cfg.get("max_unit_frames", 180),
            bootstrap_min_saturation=cfg.get("bootstrap_min_saturation", 8.0),
            bootstrap_min_texture=cfg.get("bootstrap_min_texture", 20.0),
            presence_from_input=cfg.get("presence_from_input", False),
            unit_trim_ratio=cfg.get("unit_trim_ratio", 0.2))
    elif cfg.get("use_unit_training", True):
        train_patches = extract_training_unit_patches(
            video_path, roi_cfg, skip_s=skip_s, train_s=train_s,
            step=cfg.get("training_unit_step", 3), rotate=rotate,
            blur_threshold=cfg.get("blur_threshold", 80),
            motion_threshold=cfg.get("motion_threshold", 6),
            min_foreground_ratio=cfg.get("min_foreground_ratio", 0.015),
            foreground_threshold=cfg.get("foreground_threshold", 22),
            min_stable_frames=cfg.get("min_stable_frames", 3),
            end_gap_frames=cfg.get("end_gap_frames", 3),
            max_unit_frames=cfg.get("max_unit_frames", 180),
            bootstrap_min_saturation=cfg.get("bootstrap_min_saturation", 8.0),
            bootstrap_min_texture=cfg.get("bootstrap_min_texture", 20.0),
            presence_from_input=cfg.get("presence_from_input", False),
            unit_trim_ratio=cfg.get("unit_trim_ratio", 0.2))
    else:
        train_patches = extract_training_frames(
            video_path, roi_cfg, skip_s=skip_s, train_s=train_s,
            step=cfg.get("step_train", 30), rotate=rotate,
            blur_threshold=cfg.get("blur_threshold", 80),
            motion_threshold=cfg.get("motion_threshold", 6))
    print(f"  Patches: {len(train_patches)}")

    detector = AnomalyDetector(cfg)
    detector.train(train_patches)
    print(f"  Threshold: {detector.threshold:.4f}")

    # ---- 稳定工件分件 ----------------------------------------------------------
    tracker_rois = roi_cfg if isinstance(roi_cfg, list) else [roi_cfg]
    tracker = UnitTracker(
        tracker_rois,
        motion_threshold=cfg.get("motion_threshold", 6),
        blur_threshold=cfg.get("blur_threshold", 80),
        min_foreground_ratio=cfg.get("min_foreground_ratio", 0.015),
        foreground_threshold=cfg.get("foreground_threshold", 22),
        min_stable_frames=cfg.get("min_stable_frames", 3),
        end_gap_frames=cfg.get("end_gap_frames", 3),
        max_unit_frames=cfg.get("max_unit_frames", 180),
        bootstrap_min_saturation=cfg.get("bootstrap_min_saturation", 8.0),
        bootstrap_min_texture=cfg.get("bootstrap_min_texture", 20.0),
        presence_from_input=cfg.get("presence_from_input", False),
    )
    cfg["threshold"] = float(detector.threshold)
    cfg["rois"] = tracker_rois
    renderer = DashboardRenderer(cfg)

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", f"{task_name}_demo_result.mp4",
    )
    out_w, out_h = cfg.get("viz_width", 1920), cfg.get("viz_height", 1080)

    out_fps = info["fps"] / step * max(0.1, float(speed))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, (out_w, out_h))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    total_frames_to_read = int(duration_s * fps)

    frame_count = 0
    results = []
    last_unit_info = None
    last_roi_scores = []
    last_live_unit_id = None
    live_score_cache = None
    live_score_interval = max(1, int(cfg.get("live_score_interval", 5)))

    print(f"  Testing {start_s}s–{start_s + duration_s}s, "
          f"output at {out_fps:.1f} fps ({speed}x) ...")

    # ---- 检测循环 ---------------------------------------------------------------
    while frame_count < total_frames_to_read:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step != 0:
            frame_count += 1
            continue

        if rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        if use_dynamic:
            rois, display_rois = crop_dynamic_rois(frame, tracker_rois, task_name, cfg)
        else:
            rois = crop_rois(frame, roi_cfg)
            display_rois = tracker_rois

        cfg["rois"] = display_rois
        renderer.config["rois"] = display_rois

        tracker.update(frame_count, frame, rois)

        completed_this_frame = False
        while tracker.has_completed_unit():
            unit = tracker.pop_unit()
            completed_this_frame = True
            mean_s, max_s, is_ng = detector.score_unit(unit["frames"])
            status = "NG" if is_ng else "OK"
            last_r = unit["frames"][-1][2]
            roi_scores = score_rois(detector, last_r)
            last_unit_info = {
                "unit_id": unit["id"], "status": status,
                "mean_score": mean_s, "max_score": max_s,
                "frame_count": unit["duration"], "is_completed": True,
            }
            last_roi_scores = roi_scores
            results.append({"id": unit["id"], "score": max_s, "status": status})
            print(f"  Unit #{unit['id']}: {'❌ NG' if is_ng else '✅ OK'} "
                  f"(max={max_s:.4f})")

        current_ui = last_unit_info if last_unit_info else {
            "unit_id": 0, "status": "WAIT", "mean_score": 0, "max_score": 0,
            "frame_count": 0, "is_completed": False,
        }
        current_ui = dict(current_ui)
        metrics = tracker.last_metrics
        display_state = metrics.get("display_state", "MOVING")
        is_inspecting = display_state == "INSPECTING"
        current_ui["motion_gate"] = "STABLE" if is_inspecting else "MOVING"
        current_ui["blur_gate"] = "PASS" if metrics.get("blur", 0.0) >= cfg.get("blur_threshold", 80) else "LOW"
        current_ui["motion"] = metrics.get("motion", 0.0)
        current_ui["foreground_ratio"] = metrics.get("foreground_ratio", 0.0)
        if not is_inspecting:
            live_score_cache = None
            if current_ui.get("is_completed"):
                current_ui["completed_status"] = current_ui.get("status", "OK")
            current_ui["status"] = "MOVING"
        elif not completed_this_frame:
            current_unit_id = current_ui.get("unit_id", 0)
            if current_unit_id != last_live_unit_id:
                live_score_cache = None
                last_live_unit_id = current_unit_id
            if live_score_cache is None or frame_count % (step * live_score_interval) == 0:
                live_roi_scores = score_rois(detector, rois)
                live_score = max((s for _, s, _ in live_roi_scores), default=0.0)
                live_is_ng = live_score > float(detector.threshold) * float(cfg.get("unit_threshold_margin", 1.03))
                live_score_cache = (live_roi_scores, live_score, live_is_ng)
            live_roi_scores, live_score, live_is_ng = live_score_cache
            last_roi_scores = live_roi_scores
            current_ui["status"] = "NG" if live_is_ng else "OK"
            current_ui["is_completed"] = False
            current_ui["max_score"] = live_score
            current_ui["mean_score"] = live_score

        dash = renderer.render(frame, last_roi_scores, current_ui)
        if dash.shape[1] != out_w or dash.shape[0] != out_h:
            dash = cv2.resize(dash, (out_w, out_h))
        writer.write(cv2.cvtColor(dash, cv2.COLOR_RGB2BGR))

        frame_count += 1

    # ---- 尾部处理 ---------------------------------------------------------------
    remaining = tracker.flush()
    if remaining:
        mean_s, max_s, is_ng = detector.score_unit(remaining["frames"])
        last_f = remaining["frames"][-1][1]
        last_r = remaining["frames"][-1][2]
        roi_scores = score_rois(detector, last_r)
        ui = {
            "unit_id": remaining["id"], "status": "NG" if is_ng else "OK",
            "mean_score": mean_s, "max_score": max_s,
            "frame_count": remaining["duration"], "is_completed": True,
        }
        for _ in range(30):
            dash = renderer.render(last_f, roi_scores, ui)
            if dash.shape[1] != out_w or dash.shape[0] != out_h:
                dash = cv2.resize(dash, (out_w, out_h))
            writer.write(cv2.cvtColor(dash, cv2.COLOR_RGB2BGR))
        results.append({"id": remaining["id"], "score": max_s, "status": "NG" if is_ng else "OK"})
        print(f"  Unit #{remaining['id']}: {'NG' if is_ng else 'OK'} (max={max_s:.4f})")

    cap.release()
    writer.release()

    ok_count = sum(1 for r in results if r["status"] == "OK")
    ng_count = sum(1 for r in results if r["status"] == "NG")
    print(f"\n  结果: {len(results)} 件, OK={ok_count}, NG={ng_count}")
    print(f"  输出: {out_path} ({out_fps:.1f}fps, {duration_s}s source, {speed}x)")


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "taskA"
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 360
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 25
    speed = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    run_demo(task, start, dur, speed)
