#!/usr/bin/env python3
"""
工业装配件缺失检测 — 主流程编排器

用法:
    python src/main.py taskA
    python src/main.py taskB

Pipeline:
  1. 加载配置
  2. 解析ROI
  3. 提取训练帧 → 训练异常检测器
  4. 提取测试帧 → UnitTracker分组 → 逐件评分 → 仪表盘渲染 → 输出视频
  5. 打印统计摘要
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# 确保 src/ 在路径上（支持从任意目录执行）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config
from src.detection import AnomalyDetector
from src.preprocessing import (
    extract_test_frames_gen,
    extract_training_frames,
    extract_training_unit_patches,
    get_video_info,
    rotate_rois,
)
from src.tracker import UnitTracker
from src.visualization import DashboardRenderer

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------
ROIDict = Dict[str, int]                # {"x": int, "y": int, "w": int, "h": int}
ROIConfig = Union[ROIDict, List[ROIDict]]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _normalize_roi_config(cfg: dict) -> List[ROIDict]:
    """
    将 config.yaml 中的 ROI 配置统一为 list[dict] 格式。

    支持两种输入：
      - taskA: fixed_roi = [x, y, w, h]           → [{"x":x,"y":y,"w":w,"h":h}]
      - taskB: rois = [[x,y,w,h], [x,y,w,h], ...]  → [{"x":x,...}, ...]
    """
    if "rois" in cfg and cfg["rois"]:
        raw = cfg["rois"]                                    # list of lists
    elif "fixed_roi" in cfg and cfg["fixed_roi"]:
        raw = [cfg["fixed_roi"]]                             # single list → wrap
    else:
        raise ValueError("配置中未找到 ROI（需要 fixed_roi 或 rois）")

    return [{"x": int(r[0]), "y": int(r[1]), "w": int(r[2]), "h": int(r[3])}
            for r in raw]


def _score_rois(detector: AnomalyDetector, roi_patches: List[np.ndarray]) -> List[Tuple[int, float, bool]]:
    """Score each ROI once for visualization overlays."""
    roi_scores: List[Tuple[int, float, bool]] = []
    for i, patch in enumerate(roi_patches):
        score, is_ng = detector.score_frame([patch])
        roi_scores.append((i, score, is_ng))
    return roi_scores


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_task(cfg: dict) -> dict:
    """
    对单个 task 执行完整检测流水线。

    Args:
        cfg: load_config(task_name) 返回的合并配置字典。

    Returns:
        统计摘要字典。
    """
    task_name = cfg.get("task_name", "unknown")
    video_path: str = cfg["video_path"]

    print(f"\n{'=' * 50}")
    print(f"  Anomaly Detection — {task_name}")
    print(f"{'=' * 50}")

    # ---- 1. 视频信息 ----------------------------------------------------------
    info = get_video_info(video_path)
    print(f"  Video: {info['width']}×{info['height']}, "
          f"{info['fps']:.2f} fps, {info['duration_s']:.0f} s")

    # ---- 2. 解析 ROI ---------------------------------------------------------
    roi_config: List[ROIDict] = _normalize_roi_config(cfg)
    rotate: int = cfg.get("rotate", 0)
    if cfg.get("rotate_rois_with_frame", False):
        roi_config = rotate_rois(roi_config, int(info["width"]), int(info["height"]), rotate)
    roi_count = len(roi_config)
    print(f"  ROI count: {roi_count}, rotate: {rotate}°, rotate_rois={cfg.get('rotate_rois_with_frame', False)}")

    # ---- 3. 提取训练帧 --------------------------------------------------------
    print(f"\n  [Phase 1] Extracting training frames …")
    if cfg.get("use_unit_training", True):
        train_patches = extract_training_unit_patches(
            video_path,
            roi_config,
            skip_s=cfg.get("skip_duration", 60),
            train_s=cfg.get("train_duration", 120),
            step=cfg.get("training_unit_step", 3),
            rotate=rotate,
            blur_threshold=cfg.get("blur_threshold", 80),
            motion_threshold=cfg.get("motion_threshold", 6),
            min_foreground_ratio=cfg.get("min_foreground_ratio", 0.015),
            foreground_threshold=cfg.get("foreground_threshold", 22),
            min_stable_frames=cfg.get("min_stable_frames", 3),
            end_gap_frames=cfg.get("end_gap_frames", 3),
            max_unit_frames=cfg.get("max_unit_frames", 180),
            bootstrap_min_saturation=cfg.get("bootstrap_min_saturation", 8.0),
            bootstrap_min_texture=cfg.get("bootstrap_min_texture", 20.0),
            unit_trim_ratio=cfg.get("unit_trim_ratio", 0.2),
        )
    else:
        train_patches = extract_training_frames(
            video_path,
            roi_config,
            skip_s=cfg.get("skip_duration", 60),
            train_s=cfg.get("train_duration", 120),
            step=cfg.get("step_train", 30),
            rotate=rotate,
            blur_threshold=cfg.get("blur_threshold", 80),
            motion_threshold=cfg.get("motion_threshold", 6),
        )
    if not train_patches:
        raise RuntimeError("训练帧提取结果为空，请检查视频路径和参数")
    print(f"  Total training patches: {len(train_patches)}")

    # ---- 4. 训练异常检测器 ----------------------------------------------------
    print(f"\n  [Phase 2] Training anomaly detector …")
    detector = AnomalyDetector(cfg)
    t0 = time.time()
    detector.train(train_patches)
    train_time = time.time() - t0
    print(f"  Training: {train_time:.1f} s")
    print(f"  Threshold: {detector.threshold:.4f}")
    cfg["threshold"] = float(detector.threshold)

    # ---- 5. 运行检测 ----------------------------------------------------------
    print(f"\n  [Phase 3] Running detection …")

    tracker = UnitTracker(
        roi_config,
        blur_threshold=cfg.get("blur_threshold", 80),
        motion_threshold=cfg.get("motion_threshold", 25),
        min_foreground_ratio=cfg.get("min_foreground_ratio", 0.015),
        foreground_threshold=cfg.get("foreground_threshold", 22),
        min_stable_frames=cfg.get("min_stable_frames", 3),
        end_gap_frames=cfg.get("end_gap_frames", 3),
        max_unit_frames=cfg.get("max_unit_frames", 180),
        bootstrap_min_saturation=cfg.get("bootstrap_min_saturation", 8.0),
        bootstrap_min_texture=cfg.get("bootstrap_min_texture", 20.0),
    )

    # 将 ROI 配置写入 cfg，供 DashboardRenderer 绘制检测框
    cfg["rois"] = roi_config

    renderer = DashboardRenderer(cfg)

    # 输出视频
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    out_path = os.path.join(PROJECT_ROOT, "results", f"{task_name}_result.mp4")
    out_w = cfg.get("viz_width", 1920)
    out_h = cfg.get("viz_height", 1080)
    out_fps = cfg.get("fps_output", 15)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (out_w, out_h))

    frame_count = 0
    last_unit_info: Optional[dict] = None
    last_roi_scores: List[Tuple[int, float, bool]] = []
    last_live_unit_id = None
    live_score_cache = None
    live_score_interval = max(1, int(cfg.get("live_score_interval", 5)))

    for frame_idx, full_frame, roi_patches in extract_test_frames_gen(
        video_path,
        roi_config,
        skip_s=cfg.get("skip_duration", 60),
        step=cfg.get("step_test", 3),
        rotate=rotate,
        # 测试从训练结束后开始，不与训练段重叠
        test_start_s=cfg.get("skip_duration", 60) + cfg.get("train_duration", 120),
    ):
        tracker.update(frame_idx, full_frame, roi_patches)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames …")

        # 处理已完成的件
        completed_this_frame = False
        while tracker.has_completed_unit():
            unit = tracker.pop_unit()
            completed_this_frame = True

            # detector.score_unit 返回 (mean_score, max_score, is_ng)
            mean_score, max_score, is_ng = detector.score_unit(unit["frames"])
            status = "NG" if is_ng else "OK"

            unit_info: dict = {
                "unit_id": unit["id"],
                "status": status,
                "mean_score": mean_score,
                "max_score": max_score,
                "frame_count": unit["duration"],
                "is_completed": True,
                "motion_gate": "STABLE",
                "blur_gate": "PASS",
            }

            # 使用该件的最后一帧作为仪表盘背景
            last_frame = unit["frames"][-1][1]                    # BGR
            last_rois = unit["frames"][-1][2]                     # List[np.ndarray]

            # 逐 ROI 打分
            roi_scores = _score_rois(detector, last_rois)
            last_unit_info = unit_info
            last_roi_scores = roi_scores

            print(f"  Unit #{unit['id']}: {status} "
                  f"(max={max_score:.4f}, mean={mean_score:.4f}, "
                  f"frames={unit['duration']})")

        metrics = tracker.last_metrics
        live_info = last_unit_info if last_unit_info is not None else {
            "unit_id": "-",
            "status": "WAIT",
            "mean_score": 0.0,
            "max_score": 0.0,
            "frame_count": 0,
            "is_completed": False,
        }
        live_info = dict(live_info)
        display_state = metrics.get("display_state", "MOVING")
        is_inspecting = display_state == "INSPECTING"
        live_info["motion_gate"] = "STABLE" if is_inspecting else "MOVING"
        live_info["blur_gate"] = "PASS" if metrics.get("blur", 0.0) >= cfg.get("blur_threshold", 80) else "LOW"
        live_info["motion"] = metrics.get("motion", 0.0)
        live_info["foreground_ratio"] = metrics.get("foreground_ratio", 0.0)
        if not is_inspecting:
            live_score_cache = None
            if live_info.get("is_completed"):
                live_info["completed_status"] = live_info.get("status", "OK")
            live_info["status"] = "MOVING"
        elif not completed_this_frame:
            current_unit_id = live_info.get("unit_id", "-")
            if current_unit_id != last_live_unit_id:
                live_score_cache = None
                last_live_unit_id = current_unit_id
            if live_score_cache is None or frame_count % live_score_interval == 0:
                live_roi_scores = _score_rois(detector, roi_patches)
                live_score = max((s for _, s, _ in live_roi_scores), default=0.0)
                live_is_ng = live_score > float(detector.threshold) * float(cfg.get("unit_threshold_margin", 1.03))
                live_score_cache = (live_roi_scores, live_score, live_is_ng)
            live_roi_scores, live_score, live_is_ng = live_score_cache
            last_roi_scores = live_roi_scores
            live_info["status"] = "NG" if live_is_ng else "OK"
            live_info["is_completed"] = False
            live_info["max_score"] = live_score
            live_info["mean_score"] = live_score

        dashboard = renderer.render(full_frame, last_roi_scores, live_info)
        if dashboard.shape[1] != out_w or dashboard.shape[0] != out_h:
            dashboard = cv2.resize(dashboard, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR))

    # ---- 6. 清空尾部未完结的件 ------------------------------------------------
    remaining = tracker.flush()
    if remaining is not None:
        mean_score, max_score, is_ng = detector.score_unit(remaining["frames"])
        status = "NG" if is_ng else "OK"
        unit_info = {
            "unit_id": remaining["id"],
            "status": status,
            "mean_score": mean_score,
            "max_score": max_score,
            "frame_count": remaining["duration"],
            "is_completed": True,
        }

        last_frame = remaining["frames"][-1][1]
        last_rois = remaining["frames"][-1][2]
        roi_scores = _score_rois(detector, last_rois)

        dashboard = renderer.render(last_frame, roi_scores, unit_info)
        if dashboard.shape[1] != out_w or dashboard.shape[0] != out_h:
            dashboard = cv2.resize(dashboard, (out_w, out_h),
                                   interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR))

        print(f"  Unit #{remaining['id']}: {status} "
              f"(max={max_score:.4f}, frames={remaining['duration']})")

    writer.release()

    # ---- 7. 摘要 -------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print(f"  RESULTS — {task_name}")
    print(f"{'=' * 50}")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Total units detected:   {renderer.total_units}")
    print(f"  OK:  {renderer.ok_count}")
    print(f"  NG:  {renderer.ng_count}")
    print(f"  Training time: {train_time:.1f} s")
    print(f"  Output video:   {out_path}")

    return {
        "task": task_name,
        "total_units": renderer.total_units,
        "ok": renderer.ok_count,
        "ng": renderer.ng_count,
        "threshold": float(detector.threshold),
        "train_time": round(train_time, 1),
    }


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="工业装配件缺失检测 — Anomaly Detection Pipeline"
    )
    parser.add_argument(
        "task",
        choices=["taskA", "taskB"],
        help="任务名称",
    )
    args = parser.parse_args()

    cfg = load_config(args.task)
    results = run_task(cfg)

    print(f"\nSUMMARY: {json.dumps(results, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
