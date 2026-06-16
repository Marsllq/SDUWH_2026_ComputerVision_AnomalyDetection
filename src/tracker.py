"""Stable-workpiece tracker for per-unit industrial inspection.

The tracker is deliberately generic: it does not depend on task-specific
colours or shapes.  It segments a video stream into product units by looking
inside the manually marked ROI(s) and combining three signals:

* low inter-frame motion: the workpiece is not moving;
* sufficient sharpness/texture: the ROI is usable for visual inspection;
* foreground occupancy against an adaptive idle background.

Only stable foreground runs are collected.  A completed run is one inspected
unit, so downstream anomaly detection is reported by piece rather than by
frame.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


FrameRecord = Tuple[int, np.ndarray, List[np.ndarray]]


class UnitTracker:
    """Group stable ROI frames into per-workpiece units."""

    def __init__(
        self,
        roi_config: Dict[str, int] | List[Dict[str, int]],
        motion_threshold: float = 6.0,
        blur_threshold: float = 15.0,
        min_foreground_ratio: float = 0.015,
        foreground_threshold: int = 22,
        min_stable_frames: int = 3,
        end_gap_frames: int = 3,
        max_unit_frames: int = 180,
        background_lr: float = 0.04,
        bootstrap_min_saturation: float = 8.0,
        bootstrap_min_texture: float = 20.0,
        **_: object,
    ) -> None:
        self._rois: List[Dict[str, int]] = [roi_config] if isinstance(roi_config, dict) else list(roi_config)
        self.motion_threshold = float(motion_threshold)
        self.blur_threshold = float(blur_threshold)
        self.min_foreground_ratio = float(min_foreground_ratio)
        self.foreground_threshold = int(foreground_threshold)
        self.min_stable_frames = int(min_stable_frames)
        self.end_gap_frames = int(end_gap_frames)
        self.max_unit_frames = int(max_unit_frames)
        self.background_lr = float(background_lr)
        self.bootstrap_min_saturation = float(bootstrap_min_saturation)
        self.bootstrap_min_texture = float(bootstrap_min_texture)

        self._state = "IDLE"  # IDLE | COLLECTING
        self._candidate: deque[FrameRecord] = deque(maxlen=max(1, self.min_stable_frames))
        self._current_unit: Optional[List[FrameRecord]] = None
        self._completed: List[dict] = []
        self._counter = 0
        self._gap_count = 0

        self._prev_gray: Optional[List[np.ndarray]] = None
        self._background: Optional[List[np.ndarray]] = None
        self._bootstrap_opening_unit = True
        self.last_metrics: Dict[str, float | bool | str] = {
            "motion": 0.0,
            "blur": 0.0,
            "foreground_ratio": 0.0,
            "stable": False,
            "present": False,
            "state": self._state,
            "display_state": "MOVING",
        }

    def update(self, frame_idx: int, full_frame: np.ndarray, roi_patches: List[np.ndarray]) -> None:
        """Process one sampled video frame."""
        if not roi_patches:
            return

        gray_patches = [self._to_gray(p) for p in roi_patches[: len(self._rois)] if p.size > 0]
        if not gray_patches:
            return

        motion, blur, foreground_ratio = self._measure(gray_patches)
        objectness = self._objectness(roi_patches[: len(self._rois)])
        stable = motion <= self.motion_threshold and blur >= self.blur_threshold
        present = foreground_ratio >= self.min_foreground_ratio
        bootstrap_present = self._bootstrap_opening_unit and stable and objectness
        if bootstrap_present:
            present = True

        self.last_metrics = {
            "motion": motion,
            "blur": blur,
            "foreground_ratio": foreground_ratio,
            "stable": stable,
            "present": present,
            "objectness": objectness,
            "state": self._state,
            "display_state": "MOVING",
        }

        record: FrameRecord = (frame_idx, full_frame, roi_patches)
        inspectable = stable and present

        if self._state == "IDLE":
            if inspectable:
                self._candidate.append(record)
                if len(self._candidate) >= self.min_stable_frames:
                    self._state = "COLLECTING"
                    self._current_unit = list(self._candidate)
                    self._candidate.clear()
                    self._gap_count = 0
                    self.last_metrics["display_state"] = "INSPECTING"
            else:
                self._candidate.clear()
                if stable and not present:
                    self._bootstrap_opening_unit = False
                    self._update_background(gray_patches)

        elif self._state == "COLLECTING":
            if inspectable:
                self.last_metrics["display_state"] = "INSPECTING"
                self._gap_count = 0
                if self._current_unit is not None:
                    self._current_unit.append(record)
                    if len(self._current_unit) >= self.max_unit_frames:
                        self._finalize_unit()
            else:
                self._gap_count += 1
                if self._gap_count >= self.end_gap_frames:
                    self._finalize_unit()
                    self._bootstrap_opening_unit = False
                    if stable and not present:
                        self._update_background(gray_patches)

        self._prev_gray = gray_patches
        self.last_metrics["state"] = self._state

    def has_completed_unit(self) -> bool:
        """Return True when at least one unit can be popped."""
        return bool(self._completed)

    def pop_unit(self) -> dict:
        """Return the oldest completed unit."""
        return self._completed.pop(0)

    def flush(self) -> Optional[dict]:
        """Finish the current unit at end-of-stream, if long enough."""
        if self._current_unit is not None and len(self._current_unit) >= self.min_stable_frames:
            unit = self._make_unit(self._current_unit)
            self._current_unit = None
            self._state = "IDLE"
            self._gap_count = 0
            self._bootstrap_opening_unit = False
            return unit

        self._current_unit = None
        self._candidate.clear()
        self._state = "IDLE"
        self._gap_count = 0
        self._bootstrap_opening_unit = False
        return None

    def _measure(self, gray_patches: List[np.ndarray]) -> Tuple[float, float, float]:
        if self._background is None:
            self._background = [g.astype(np.float32) for g in gray_patches]

        motions: List[float] = []
        if self._prev_gray is None or len(self._prev_gray) != len(gray_patches):
            motions = [0.0 for _ in gray_patches]
        else:
            for gray, prev in zip(gray_patches, self._prev_gray):
                motions.append(float(np.mean(cv2.absdiff(gray, prev))))

        blurs = [float(cv2.Laplacian(g, cv2.CV_64F).var()) for g in gray_patches]
        fg_ratios = []
        for gray, bg in zip(gray_patches, self._background):
            bg_u8 = np.clip(bg, 0, 255).astype(np.uint8)
            diff = cv2.absdiff(gray, bg_u8)
            fg_ratios.append(float(np.count_nonzero(diff > self.foreground_threshold)) / diff.size)

        return float(max(motions)), float(max(blurs)), float(max(fg_ratios))

    def _update_background(self, gray_patches: List[np.ndarray]) -> None:
        if self._background is None or len(self._background) != len(gray_patches):
            self._background = [g.astype(np.float32) for g in gray_patches]
            return
        for i, gray in enumerate(gray_patches):
            cv2.accumulateWeighted(gray.astype(np.float32), self._background[i], self.background_lr)

    def _objectness(self, roi_patches: List[np.ndarray]) -> bool:
        """Heuristic guard against treating an empty stable station as a part."""
        for patch in roi_patches:
            if patch.size == 0:
                continue
            h, w = patch.shape[:2]
            r = max(8, int(min(h, w) * 0.3))
            cy, cx = h // 2, w // 2
            sub = patch[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r)]
            if sub.size == 0:
                continue
            hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
            sat = float(np.mean(hsv[:, :, 1]))
            gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
            texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if sat >= self.bootstrap_min_saturation or texture >= self.bootstrap_min_texture:
                return True
        return False

    def _finalize_unit(self) -> None:
        if self._current_unit is not None and len(self._current_unit) >= self.min_stable_frames:
            self._completed.append(self._make_unit(self._current_unit))
        self._current_unit = None
        self._candidate.clear()
        self._state = "IDLE"
        self._gap_count = 0
        self._bootstrap_opening_unit = False

    def _make_unit(self, frames_info: List[FrameRecord]) -> dict:
        self._counter += 1
        return {
            "id": self._counter,
            "frames": frames_info,
            "duration": len(frames_info),
            "start_idx": frames_info[0][0],
            "end_idx": frames_info[-1][0],
        }

    @staticmethod
    def _to_gray(roi_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)
