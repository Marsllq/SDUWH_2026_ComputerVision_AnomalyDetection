#!/usr/bin/env python3
"""
快速推理脚本：跳过视频渲染，只做检测+评分，输出统计。
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config
from src.preprocessing import extract_training_frames, extract_training_unit_patches, extract_test_frames_gen, get_video_info, rotate_rois
from src.detection import AnomalyDetector
from src.tracker import UnitTracker
import numpy as np

def run_fast(task_name):
    cfg = load_config(task_name)
    video_path = cfg["video_path"]

    # Normalize ROI
    if "rois" in cfg and cfg["rois"]:
        roi_raw = cfg["rois"]
        roi_cfg = [{"x": r[0], "y": r[1], "w": r[2], "h": r[3]} for r in roi_raw]
    elif "fixed_roi" in cfg and cfg["fixed_roi"]:
        roi_cfg = {"x": cfg["fixed_roi"][0], "y": cfg["fixed_roi"][1],
                   "w": cfg["fixed_roi"][2], "h": cfg["fixed_roi"][3]}
    else:
        raise ValueError("No ROI config")

    rotate = cfg.get("rotate", 0)
    skip_s = cfg.get("skip_duration", 60)
    train_s = cfg.get("train_duration", 120)
    if cfg.get("rotate_rois_with_frame", False):
        info = get_video_info(video_path)
        roi_list = roi_cfg if isinstance(roi_cfg, list) else [roi_cfg]
        roi_cfg = rotate_rois(roi_list, int(info["width"]), int(info["height"]), rotate)

    print(f"  [{task_name}] Training...")
    t0 = time.time()
    if cfg.get("use_unit_training", True):
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
    print(f"  Patches: {len(train_patches)}, {time.time()-t0:.1f}s")

    detector = AnomalyDetector(cfg)
    t0 = time.time()
    detector.train(train_patches)
    print(f"  Train: {time.time()-t0:.1f}s, threshold={detector.threshold:.4f}")

    tracker = UnitTracker(
        roi_cfg,
        motion_threshold=cfg.get("motion_threshold", 6),
        blur_threshold=cfg.get("blur_threshold", 80),
        min_foreground_ratio=cfg.get("min_foreground_ratio", 0.015),
        foreground_threshold=cfg.get("foreground_threshold", 22),
        min_stable_frames=cfg.get("min_stable_frames", 3),
        end_gap_frames=cfg.get("end_gap_frames", 3),
        max_unit_frames=cfg.get("max_unit_frames", 180),
        presence_from_input=cfg.get("presence_from_input", False),
        presence_mode=cfg.get("presence_mode", "generic"),
        min_present_rois=cfg.get("min_present_rois", 1),
        no_part_max_present_rois=cfg.get("no_part_max_present_rois", 0),
        roi_roles=cfg.get("taskB_roi_roles", []),
        blue_presence_min=cfg.get("taskB_blue_presence_min", 0.18),
        white_presence_min=cfg.get("taskB_white_presence_min", 0.35),
    )
    test_start = skip_s + train_s

    results = []
    t0 = time.time()
    frame_count = 0

    for idx, frame, rois in extract_test_frames_gen(
        video_path, roi_cfg, skip_s=skip_s, step=cfg.get("step_test", 3),
        rotate=rotate, test_start_s=test_start):
        tracker.update(idx, frame, rois)
        frame_count += 1
        while tracker.has_completed_unit():
            unit = tracker.pop_unit()
            mean_s, max_s, is_ng = detector.score_unit(unit["frames"])
            results.append({
                "id": unit["id"], "frames": unit["duration"],
                "mean": round(float(mean_s), 4),
                "max": round(float(max_s), 4),
                "ng": is_ng,
                "start_idx": unit["start_idx"]
            })
        if frame_count % 500 == 0:
            print(f"  {frame_count} frames, {len(results)} units...")

    remaining = tracker.flush()
    if remaining:
        mean_s, max_s, is_ng = detector.score_unit(remaining["frames"])
        results.append({
            "id": remaining["id"], "frames": remaining["duration"],
            "mean": round(float(mean_s), 4),
            "max": round(float(max_s), 4),
            "ng": is_ng,
            "start_idx": remaining["start_idx"]
        })

    total_time = time.time() - t0
    ok_count = sum(1 for r in results if not r["ng"])
    ng_count = sum(1 for r in results if r["ng"])

    print(f"\n{'='*50}")
    print(f"  RESULTS — {task_name}")
    print(f"{'='*50}")
    print(f"  Total frames: {frame_count}")
    print(f"  Total units:  {len(results)}")
    print(f"  OK: {ok_count}  NG: {ng_count}")
    print(f"  Time: {total_time:.1f}s")
    print(f"  Threshold: {detector.threshold:.4f}")
    print(f"\n  Per-unit breakdown:")
    for r in results:
        icon = "❌ NG" if r["ng"] else "✅ OK"
        print(f"  Unit #{r['id']}: {icon}  max={r['max']:.4f}  mean={r['mean']:.4f}  frames={r['frames']}")

    return {"task": task_name, "total": len(results), "ok": ok_count, "ng": ng_count,
            "threshold": detector.threshold, "time": round(total_time, 1)}

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "taskA"
    print(json.dumps(run_fast(task), indent=2))
