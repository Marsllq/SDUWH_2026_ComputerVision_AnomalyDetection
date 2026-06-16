"""
Preprocessing module: video info, frame extraction, ROI cropping, rotation.

All functions accept roi_cfg as either a single dict {"x":x,"y":y,"w":w,"h":h}
or a list of such dicts for multi-ROI (taskA vs taskB).
"""
from __future__ import annotations

import os
from typing import Dict, Generator, List, Tuple, Union

import cv2
import numpy as np

# ROI config: single dict or list of dicts
ROIConfig = Union[Dict[str, int], List[Dict[str, int]]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_rois(roi_cfg: ROIConfig) -> List[Dict[str, int]]:
    """Normalise roi_cfg to a list of dicts regardless of input shape."""
    if isinstance(roi_cfg, dict):
        return [roi_cfg]
    return list(roi_cfg)


def _roi_slice(roi: Dict[str, int], h: int, w: int) -> Tuple[slice, slice]:
    """Return (y-slice, x-slice) for a ROI, clipped to frame bounds."""
    x1 = max(0, roi["x"])
    y1 = max(0, roi["y"])
    x2 = min(w, x1 + roi["w"])
    y2 = min(h, y1 + roi["h"])
    return slice(y1, y2), slice(x1, x2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_video_info(video_path: str) -> Dict[str, float]:
    """
    Read video metadata without decoding frames.

    Args:
        video_path: Absolute or relative path to the video file.

    Returns:
        dict with keys: width, height, fps, total_frames, duration_s.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0.0

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": duration_s,
    }


def crop_rois(frame: np.ndarray, roi_cfg: ROIConfig) -> List[np.ndarray]:
    """
    Crop one or more ROI regions from a full frame.

    Args:
        frame: BGR image (H x W x 3).
        roi_cfg: Single ROI dict {"x": x, "y": y, "w": w, "h": h}
                 or a list of such dicts for multi-ROI.

    Returns:
        List of cropped BGR patches (one per ROI).
    """
    h, w = frame.shape[:2]
    rois = _normalize_rois(roi_cfg)
    patches: List[np.ndarray] = []

    for roi in rois:
        ys, xs = _roi_slice(roi, h, w)
        patch = frame[ys, xs]
        if patch.size == 0:
            # Out-of-bounds ROI; return a blank patch of the requested size.
            patch = np.zeros((roi["h"], roi["w"], 3), dtype=np.uint8)
        patches.append(patch)

    return patches


def rotate_rois(rois: List[Dict[str, int]], frame_w: int, frame_h: int, angle: int) -> List[Dict[str, int]]:
    """Rotate ROI rectangles around the full frame origin.

    This is used when the video frame is rotated before cropping.  The ROI
    values in config can remain in original-video coordinates, then be mapped
    into the rotated detection coordinate system here.
    """
    rotated: List[Dict[str, int]] = []
    for roi in rois:
        x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
        if angle == 180:
            rotated.append({"x": frame_w - x - w, "y": frame_h - y - h, "w": w, "h": h})
        elif angle == 90:
            rotated.append({"x": frame_h - y - h, "y": x, "w": h, "h": w})
        elif angle == 270:
            rotated.append({"x": y, "y": frame_w - x - w, "w": h, "h": w})
        else:
            rotated.append({"x": x, "y": y, "w": w, "h": h})
    return rotated


def extract_training_frames(
    video_path: str,
    roi_cfg: ROIConfig,
    skip_s: float = 60.0,
    train_s: float = 120.0,
    step: int = 30,
    rotate: int = 0,
    blur_threshold: float = 80.0,
    motion_threshold: float = 6.0,
    min_training_patches: int = 16,
) -> List[np.ndarray]:
    """
    Extract training patches from a video segment.

    Behaviour:
      - Skips the first *skip_s* seconds (camera stabilisation).
      - Samples every *step*-th frame for *train_s* seconds.
      - Applies an optional 180° rotation.
      - Rejects blurry or moving frames using ROI-level Laplacian/motion gates.
      - For each valid frame, crops every ROI and collects all patches into a
        single flat list.

    Args:
        video_path: Path to the video file.
        skip_s: Seconds to skip at the start.
        train_s: Duration of the training segment in seconds.
        step: Frame interval between samples.
        roi_cfg: Single ROI dict or list of ROI dicts.
        rotate: Rotation to apply (0 or 180).
        blur_threshold: Minimum Laplacian variance to keep a frame.

    Returns:
        Flat list of BGR ROI patches.  For multi-ROI the order is
        [frame0_roi0, frame0_roi1, ..., frame1_roi0, ...].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(skip_s * fps)
    end_frame = min(total_frames, int((skip_s + train_s) * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rois = _normalize_rois(roi_cfg)

    patches: List[np.ndarray] = []
    fallback_patches: List[np.ndarray] = []
    prev_gray_patches: List[np.ndarray] = []
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % step == 0:
            # Rotation
            if rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Crop every ROI for this frame
            roi_patches = crop_rois(frame, rois)
            roi_patches = [p for p in roi_patches if p.size > 0]
            if roi_patches:
                fallback_patches.extend(roi_patches)

                gray_patches = [
                    cv2.resize(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), (160, 160), interpolation=cv2.INTER_AREA)
                    for p in roi_patches
                ]
                blur = max(float(cv2.Laplacian(g, cv2.CV_64F).var()) for g in gray_patches)
                if prev_gray_patches and len(prev_gray_patches) == len(gray_patches):
                    motion = max(
                        float(np.mean(cv2.absdiff(g, prev)))
                        for g, prev in zip(gray_patches, prev_gray_patches)
                    )
                else:
                    motion = 0.0

                if blur >= blur_threshold and motion <= motion_threshold:
                    patches.extend(roi_patches)
                prev_gray_patches = gray_patches

        frame_idx += 1

    cap.release()
    if len(patches) < min_training_patches and fallback_patches:
        print(
            f"  Stable training patches only {len(patches)}; "
            f"falling back to sampled patches ({len(fallback_patches)})"
        )
        patches = fallback_patches

    print(f"  Training patches extracted: {len(patches)}")
    return patches


def extract_training_unit_patches(
    video_path: str,
    roi_cfg: ROIConfig,
    skip_s: float = 60.0,
    train_s: float = 120.0,
    step: int = 3,
    rotate: int = 0,
    blur_threshold: float = 15.0,
    motion_threshold: float = 6.0,
    min_foreground_ratio: float = 0.015,
    foreground_threshold: int = 22,
    min_stable_frames: int = 3,
    end_gap_frames: int = 3,
    max_unit_frames: int = 180,
    bootstrap_min_saturation: float = 8.0,
    bootstrap_min_texture: float = 20.0,
    presence_from_input: bool = False,
    unit_trim_ratio: float = 0.2,
    sample_step: int = 5,
    min_training_patches: int = 16,
) -> List[np.ndarray]:
    """Extract normal training patches from middle frames of stable units."""
    from src.tracker import UnitTracker

    rois = _normalize_rois(roi_cfg)
    tracker = UnitTracker(
        rois,
        motion_threshold=motion_threshold,
        blur_threshold=blur_threshold,
        min_foreground_ratio=min_foreground_ratio,
        foreground_threshold=foreground_threshold,
        min_stable_frames=min_stable_frames,
        end_gap_frames=end_gap_frames,
        max_unit_frames=max_unit_frames,
        bootstrap_min_saturation=bootstrap_min_saturation,
        bootstrap_min_texture=bootstrap_min_texture,
        presence_from_input=presence_from_input,
    )

    patches: List[np.ndarray] = []
    end_s = skip_s + train_s
    fps = max(1.0, get_video_info(video_path)["fps"])
    for frame_idx, frame, roi_patches in extract_test_frames_gen(
        video_path,
        rois,
        skip_s=skip_s,
        step=step,
        rotate=rotate,
        test_start_s=skip_s,
    ):
        if frame_idx / fps >= end_s:
            break
        tracker.update(frame_idx, frame, roi_patches)
        while tracker.has_completed_unit():
            unit = tracker.pop_unit()
            patches.extend(_middle_unit_patches(unit["frames"], unit_trim_ratio, sample_step))

    remaining = tracker.flush()
    if remaining is not None:
        patches.extend(_middle_unit_patches(remaining["frames"], unit_trim_ratio, sample_step))

    if len(patches) < min_training_patches:
        print(f"  Unit training patches only {len(patches)}; falling back to stable frame sampling")
        return extract_training_frames(
            video_path,
            rois,
            skip_s=skip_s,
            train_s=train_s,
            step=max(step, 30),
            rotate=rotate,
            blur_threshold=blur_threshold,
            motion_threshold=motion_threshold,
            min_training_patches=min_training_patches,
        )

    print(f"  Unit training patches extracted: {len(patches)}")
    return patches


def extract_localized_training_unit_patches(
    video_path: str,
    roi_cfg: ROIConfig,
    task_name: str,
    cfg: dict,
    skip_s: float = 60.0,
    train_s: float = 120.0,
    step: int = 3,
    rotate: int = 0,
    blur_threshold: float = 15.0,
    motion_threshold: float = 6.0,
    min_foreground_ratio: float = 0.015,
    foreground_threshold: int = 22,
    min_stable_frames: int = 3,
    end_gap_frames: int = 3,
    max_unit_frames: int = 180,
    bootstrap_min_saturation: float = 8.0,
    bootstrap_min_texture: float = 20.0,
    presence_from_input: bool = False,
    unit_trim_ratio: float = 0.2,
    sample_step: int = 5,
    min_training_patches: int = 16,
) -> List[np.ndarray]:
    """Extract training patches after task-specific localization."""
    from src.locator import crop_dynamic_rois
    from src.tracker import UnitTracker

    coarse_rois = _normalize_rois(roi_cfg)
    tracker = UnitTracker(
        coarse_rois,
        motion_threshold=motion_threshold,
        blur_threshold=blur_threshold,
        min_foreground_ratio=min_foreground_ratio,
        foreground_threshold=foreground_threshold,
        min_stable_frames=min_stable_frames,
        end_gap_frames=end_gap_frames,
        max_unit_frames=max_unit_frames,
        bootstrap_min_saturation=bootstrap_min_saturation,
        bootstrap_min_texture=bootstrap_min_texture,
        presence_from_input=presence_from_input,
    )

    patches: List[np.ndarray] = []
    fps = max(1.0, get_video_info(video_path)["fps"])
    end_s = skip_s + train_s
    for frame_idx, frame, _ in extract_test_frames_gen(
        video_path,
        coarse_rois,
        skip_s=skip_s,
        step=step,
        rotate=rotate,
        test_start_s=skip_s,
    ):
        if frame_idx / fps >= end_s:
            break
        localized_patches, localized_rois = crop_dynamic_rois(frame, coarse_rois, task_name, cfg)
        if not localized_patches:
            continue
        tracker.update(frame_idx, frame, localized_patches)
        while tracker.has_completed_unit():
            unit = tracker.pop_unit()
            patches.extend(_middle_unit_patches(unit["frames"], unit_trim_ratio, sample_step))

    remaining = tracker.flush()
    if remaining is not None:
        patches.extend(_middle_unit_patches(remaining["frames"], unit_trim_ratio, sample_step))

    if len(patches) < min_training_patches:
        print(f"  Localized training patches only {len(patches)}; falling back to coarse unit training")
        return extract_training_unit_patches(
            video_path,
            coarse_rois,
            skip_s=skip_s,
            train_s=train_s,
            step=step,
            rotate=rotate,
            blur_threshold=blur_threshold,
            motion_threshold=motion_threshold,
            min_foreground_ratio=min_foreground_ratio,
            foreground_threshold=foreground_threshold,
            min_stable_frames=min_stable_frames,
            end_gap_frames=end_gap_frames,
            max_unit_frames=max_unit_frames,
            bootstrap_min_saturation=bootstrap_min_saturation,
            bootstrap_min_texture=bootstrap_min_texture,
            presence_from_input=presence_from_input,
            unit_trim_ratio=unit_trim_ratio,
            sample_step=sample_step,
            min_training_patches=min_training_patches,
        )

    print(f"  Localized training patches extracted: {len(patches)}")
    return patches


def _middle_unit_patches(
    unit_frames: List[Tuple[int, np.ndarray, List[np.ndarray]]],
    trim_ratio: float,
    sample_step: int,
) -> List[np.ndarray]:
    n = len(unit_frames)
    trim = int(n * trim_ratio)
    if n - 2 * trim >= max(3, sample_step):
        core = unit_frames[trim:n - trim]
    else:
        core = unit_frames
    selected = core[::max(1, sample_step)] or core
    patches: List[np.ndarray] = []
    for _, _, roi_patches in selected:
        patches.extend([p for p in roi_patches if p.size > 0])
    return patches


def extract_test_frames_gen(
    video_path: str,
    roi_cfg: ROIConfig,
    skip_s: float = 60.0,
    step: int = 3,
    rotate: int = 0,
    test_start_s: float = None,
) -> Generator[Tuple[int, np.ndarray, List[np.ndarray]], None, None]:
    """
    Generator yielding test frames one-by-one to save memory.

    Starts after *skip_s* + *test_start_s* seconds (or skip_s if test_start_s is None)
    and samples every *step*-th frame for the remainder of the video.

    Args:
        test_start_s: Optional additional offset. When training uses [skip_s, skip_s+train_s],
                      set test_start_s = skip_s + train_s to avoid overlap.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If test_start_s is provided, start from there; otherwise use skip_s
    effective_skip = test_start_s if test_start_s is not None else skip_s
    start_frame = int(effective_skip * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rois = _normalize_rois(roi_cfg)

    frame_idx = start_frame

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % step == 0:
            # Rotation
            if rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Crop ROIs
            patches = crop_rois(frame, rois)

            yield frame_idx, frame, patches

        frame_idx += 1

    cap.release()
