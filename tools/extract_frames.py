#!/usr/bin/env python3
"""
ROI参考帧提取工具
从流水线视频中提取清晰、静止的帧，供用户手动标定固定大区域ROI坐标。

用法:
    python tools/extract_frames.py taskA
    python tools/extract_frames.py taskB
    python tools/extract_frames.py taskA --skip 60 --search 240
"""
import os
import sys
import argparse
import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROI_DIR = os.path.join(PROJECT_ROOT, "roi_reference")

TASK_PATHS = {
    "taskA": os.path.join(PROJECT_ROOT, "dataset/taskA/taskA_data_video.mp4"),
    "taskB": os.path.join(PROJECT_ROOT, "dataset/taskB/taskB_data_video.mp4"),
}


def extract_frames(task_name: str, skip_duration: int = 60, search_duration: int = 240):
    """
    提取稳定帧的主函数

    Args:
        task_name: taskA 或 taskB
        skip_duration: 跳过开头秒数（摄像机抖动）
        search_duration: 搜索范围时长（秒）
    """
    video_path = TASK_PATHS.get(task_name)
    if not video_path:
        print(f"❌ 未知任务: {task_name}，可选: {list(TASK_PATHS.keys())}")
        sys.exit(1)

    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        sys.exit(1)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        sys.exit(1)

    # 读取元数据
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if fps > 0 else 0
    orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))

    print(f"\n{'='*60}")
    print(f"📹 视频信息 — {task_name}")
    print(f"{'='*60}")
    print(f"  分辨率: {width}×{height}")
    print(f"  帧率:   {fps:.2f} fps")
    print(f"  总帧数: {total_frames}")
    print(f"  总时长: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"  旋转元: {orientation}°" if orientation else "  旋转元: 无")
    print()

    # 计算跳过的帧数和搜索范围
    skip_frames = int(skip_duration * fps)
    search_frames = int(min(search_duration, total_duration - skip_duration - 10) * fps)
    end_frame = skip_frames + search_frames

    if skip_frames >= total_frames:
        print(f"❌ skip_duration({skip_duration}s) 超过视频总时长")
        cap.release()
        sys.exit(1)

    # 跳到跳过位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

    # 逐帧扫描并计算拉普拉斯方差 + 运动分数
    frames_data = []  # [(frame_idx, laplacian_var, motion_score, frame_bgr), ...]
    prev_gray = None

    sample_interval = int(fps)  # 每秒采样1帧
    total_samples = search_frames // sample_interval

    print(f"⏳ 正在扫描 {skip_duration}s → {skip_duration + search_duration}s ...")
    print(f"  采样间隔: {sample_interval}帧 (≈1fps)")
    print(f"  预计采样: {total_samples}帧")

    current_frame = skip_frames
    pbar = tqdm(total=total_samples, desc="扫描进度")

    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (current_frame - skip_frames) % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # 拉普拉斯方差（清晰度）
            laplacian_var = cv2.Laplacian(gray_blur, cv2.CV_64F).var()

            # 帧差法运动评分
            motion_score = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(gray_blur, prev_gray)
                motion_score = float(np.mean(diff))
            prev_gray = gray_blur.copy()

            frames_data.append((current_frame, laplacian_var, motion_score, frame))
            pbar.update(1)

        current_frame += 1

    cap.release()
    pbar.close()

    # 筛选：清晰(>20) 且 静止(<25)
    # 注：拉普拉斯方差阈值根据实际视频调整（该数据集约30-50）
    candidates = [(idx, lv, ms, f) for idx, lv, ms, f in frames_data
                  if lv > 20 and ms < 25]

    print(f"\n  总采样: {len(frames_data)}帧")
    print(f"  满足条件(清晰+静止): {len(candidates)}帧")

    if len(candidates) == 0:
        # 放宽条件
        print("  ⚠️  无帧满足严格条件，放宽筛选标准...")
        # 用所有采样帧，按清晰度排序
        candidates = sorted(frames_data, key=lambda x: x[1], reverse=True)
        candidates = [(idx, lv, ms, f) for idx, lv, ms, f in candidates[:10]]
        print(f"  改用清晰度最高的 {len(candidates)} 帧")

    # 按清晰度排序，取top 5
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_frames = candidates[:5]

    # 确保输出目录存在
    os.makedirs(ROI_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"💾 保存参考帧到: roi_reference/")
    print(f"{'='*60}")

    saved_paths = []
    for i, (frame_idx, lv, ms, frame) in enumerate(top_frames):
        timestamp = frame_idx / fps
        base_name = f"{task_name}_stable_frame_{i}"

        # 原始方向
        path_orig = os.path.join(ROI_DIR, f"{base_name}.png")
        cv2.imwrite(path_orig, frame)
        saved_paths.append(path_orig)

        # 水平翻转（媒体播放器可能自动翻转）
        flipped = cv2.flip(frame, 1)
        path_flip = os.path.join(ROI_DIR, f"{base_name}_FLIPPED.png")
        cv2.imwrite(path_flip, flipped)
        saved_paths.append(path_flip)

        print(f"  [{i}] 帧#{frame_idx} ({timestamp:.1f}s) | "
              f"清晰度={lv:.1f} | 运动={ms:.2f}")
        print(f"      原图: {path_orig}")
        print(f"      翻转: {path_flip}")

    # 打印操作指引
    print(f"\n{'='*60}")
    print(f"📋 下一步操作")
    print(f"{'='*60}")
    print(f"""
1. 打开 Finder: open {ROI_DIR}
2. 对比 {task_name}_stable_frame_0.png 和
   {task_name}_stable_frame_0_FLIPPED.png
   → 看看哪个与你在媒体播放器(QuickTime/VLC)中看到的方向一致
3. 用预览/画图工具打开正确方向的图片
4. 标定一个"固定大区域"ROI，记录坐标 (x, y, w, h)
   → 这个区域应足够大，能包含工件到达时的各种可能位置
5. 更新 config.yaml 中 {task_name} 的 fixed_roi 字段:
       {task_name}:
         video_path: "dataset/{task_name}/{task_name}_data_video.mp4"
         fixed_roi: [x, y, w, h]     ← 在这里填入你的坐标
         rotate: 0                    ← 需要旋转? 0/90/180/270
6. 完成后告诉我，我继续下一步

建议: 用剪刀工具或截图测量ROI坐标。图片尺寸为 {width}×{height}。
""")


def main():
    parser = argparse.ArgumentParser(description="提取视频稳定帧用于ROI标定")
    parser.add_argument("task", choices=["taskA", "taskB"], help="任务名称")
    parser.add_argument("--skip", type=int, default=60, help="跳过开头秒数 (默认: 60)")
    parser.add_argument("--search", type=int, default=240,
                        help="搜索范围秒数 (默认: 240)")
    args = parser.parse_args()

    extract_frames(args.task, args.skip, args.search)


if __name__ == "__main__":
    main()
