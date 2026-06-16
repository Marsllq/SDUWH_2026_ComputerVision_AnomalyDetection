"""Target localization inside manually marked coarse ROIs."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


ROIDict = Dict[str, int]


def crop_dynamic_rois(frame: np.ndarray, rois: List[ROIDict], task_name: str, cfg: dict) -> Tuple[List[np.ndarray], List[ROIDict]]:
    """Locate task-specific target ROIs inside coarse manual ROIs."""
    if task_name == "taskA" and rois:
        located = locate_taskA_endface(frame, rois[0], cfg)
        if located is None:
            return [], []
        patch, roi = located
        return [patch], [roi]

    patches: List[np.ndarray] = []
    valid_rois: List[ROIDict] = []
    h, w = frame.shape[:2]
    for roi in rois:
        x, y, rw, rh = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + rw), min(h, y + rh)
        if x2 <= x1 or y2 <= y1:
            continue
        patches.append(frame[y1:y2, x1:x2])
        valid_rois.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})
    return patches, valid_rois


def locate_taskA_endface(frame: np.ndarray, coarse_roi: ROIDict, cfg: dict) -> Optional[Tuple[np.ndarray, ROIDict]]:
    """Find taskA circular end face inside a coarse ROI and return an aligned crop."""
    h, w = frame.shape[:2]
    x, y, rw, rh = int(coarse_roi["x"]), int(coarse_roi["y"]), int(coarse_roi["w"]), int(coarse_roi["h"])
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + rw), min(h, y + rh)
    if x2 <= x1 or y2 <= y1:
        return None

    patch = frame[y1:y2, x1:x2]
    center = _yellow_center(patch)
    if center is None:
        center = _circle_center(patch)
    if center is None:
        return None

    cx, cy, radius = center
    crop_size = int(cfg.get("taskA_dynamic_crop_size", max(180, min(rw, rh) * 0.72)))
    half = crop_size // 2
    gx = x1 + cx
    gy = y1 + cy
    dx1, dy1 = max(0, gx - half), max(0, gy - half)
    dx2, dy2 = min(w, gx + half), min(h, gy + half)
    if dx2 - dx1 < 80 or dy2 - dy1 < 80:
        return None

    dyn_roi = {"x": int(dx1), "y": int(dy1), "w": int(dx2 - dx1), "h": int(dy2 - dy1)}
    return frame[dy1:dy2, dx1:dx2], dyn_roi


def _yellow_center(patch: np.ndarray) -> Optional[Tuple[int, int, int]]:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([10, 35, 35], dtype=np.uint8), np.array([45, 255, 255], dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 80:
        return None
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    radius = int(max(35, np.sqrt(area / np.pi) * 1.8))
    return cx, cy, radius


def _circle_center(patch: np.ndarray) -> Optional[Tuple[int, int, int]]:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    min_dim = min(patch.shape[:2])
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(60, min_dim // 3),
        param1=80,
        param2=24,
        minRadius=max(35, int(min_dim * 0.14)),
        maxRadius=max(60, int(min_dim * 0.42)),
    )
    if circles is None:
        return None
    circles = np.round(circles[0]).astype(int)
    ph, pw = patch.shape[:2]
    cx0, cy0 = pw // 2, ph // 2
    best = min(circles, key=lambda c: (c[0] - cx0) ** 2 + (c[1] - cy0) ** 2)
    return int(best[0]), int(best[1]), int(best[2])
