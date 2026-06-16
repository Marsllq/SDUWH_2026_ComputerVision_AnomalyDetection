"""High-level DINOv2 + PatchCore anomaly detection pipeline."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from src.config import load_config
from src.feature_extractor import extract_patch_features, load_dinov2
from src.memory_bank import build_memory_bank, calibrate_threshold, score_frame as patchcore_score_frame
# Legacy imports used by test() / _load_training_patches_from_task_config() —
# these are imported lazily inside those methods since their signatures may
# drift from the current preprocessing / tracker modules.
# from src.preprocessing import extract_training_frames
# from src.tracker import group_into_units, print_units_summary


class AnomalyDetector:
    """
    High-level anomaly detection pipeline.

    Usage:
        detector = AnomalyDetector(config)
        detector.train(train_patches)  # Extract features + build memory bank + calibrate
        score = detector.score_frame(roi_patches)  # Single frame
    """

    def __init__(self, config: Union[Dict[str, Any], str]):
        # config fields: dinov2_model, input_size, device, coreset_ratio, knn_k, threshold_k, score_aggregation
        self.task_name: Optional[str] = config if isinstance(config, str) else config.get("task_name")
        self.config: Dict[str, Any] = load_config(config) if isinstance(config, str) else dict(config)
        self.cfg = self.config  # Backwards-compatible alias used by existing scripts.

        self.model = None  # lazy load
        self.memory_bank: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.train_scores: Optional[np.ndarray] = None
        self.taskA_yellow_threshold: Optional[float] = None

        # Backwards-compatible object exposing .threshold for src/main.py.
        self.memory = None

    def _get(self, key: str, default: Any = None) -> Any:
        """Read a detector config value with a default."""
        return self.config.get(key, default)

    def _ensure_model(self) -> None:
        """Lazy-load DINOv2 model."""
        if self.model is None:
            self.model = load_dinov2(
                model_name=self._get("dinov2_model", "vit_small_patch14_dinov2"),
                device=self._get("device", "cpu"),
            )

    def _extract_features(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Extract patch-token features for a non-empty list of BGR ROI images."""
        self._ensure_model()
        return extract_patch_features(self.model, list(images), device=self._get("device", "cpu"))

    def train(self, train_patches: Optional[List[np.ndarray]] = None) -> None:
        """
        train_patches: list of np.ndarray (BGR ROI patches)
        1. Load DINOv2 model if not loaded
        2. Extract patch features
        3. Build memory bank
        4. Calibrate threshold
        """
        if train_patches is None:
            train_patches = self._load_training_patches_from_task_config()
        if not train_patches:
            raise ValueError("train_patches must contain at least one normal BGR ROI patch")

        per_frame_features = self._extract_features(train_patches)
        self.memory_bank = build_memory_bank(
            per_frame_features,
            coreset_ratio=float(self._get("coreset_ratio", 0.10)),
            seed=42,
        )
        self.threshold, self.train_scores = calibrate_threshold(
            per_frame_features,
            self.memory_bank,
            knn_k=int(self._get("knn_k", 3)),
            k_coeff=float(self._get("threshold_k", 3.0)),
            aggregation=self._get("score_aggregation", "mean_top5"),
        )
        self.memory = _MemoryView(self.memory_bank, self.threshold, self.train_scores)

        if self.task_name == "taskA":
            ratios = np.array([_taskA_yellow_ratio(p) for p in train_patches if p is not None and p.size > 0], dtype=np.float32)
            if ratios.size:
                k = float(self._get("taskA_yellow_threshold_k", 3.0))
                learned = float(np.mean(ratios) - k * np.std(ratios))
                floor = float(self._get("taskA_yellow_min_ratio", 0.10))
                self.taskA_yellow_threshold = max(floor, learned)

    def score_frame(self, roi_patches: Union[np.ndarray, List[np.ndarray]]) -> Tuple[float, bool]:
        """
        Score ROI patches for one frame.
        For multi-ROI: extract features from each ROI, score each, take max score.
        Returns anomaly_score, is_ng (bool)
        """
        if self.memory_bank is None or self.threshold is None:
            raise RuntimeError("Detector is not trained; call train() first")

        rois = roi_patches if isinstance(roi_patches, list) else [roi_patches]
        if not rois:
            raise ValueError("roi_patches must contain at least one ROI image")

        # Multi-ROI testing: score each ROI independently and take max; any ROI
        # failing should make the frame NG.
        features = self._extract_features(rois)
        scores = [
            patchcore_score_frame(
                roi_features,
                self.memory_bank,
                knn_k=int(self._get("knn_k", 3)),
                aggregation=self._get("score_aggregation", "mean_top5"),
            )
            for roi_features in features
        ]
        anomaly_score = float(np.max(scores))
        is_ng = anomaly_score > self.threshold
        return anomaly_score, is_ng

    def score_unit(self, unit_frames: List[Tuple[int, np.ndarray, List[np.ndarray]]],
                   sample_step: int = 5) -> Tuple[float, float, bool]:
        """
        Score a complete product unit (multiple frames).
        - unit_frames: list of (frame_idx, full_frame, roi_patches_list)
        - Drop entering/leaving edge frames; they are the noisiest part of a unit.
        - Score every sample_step frames (consecutive frames are nearly identical).
        - Aggregate with a high percentile instead of max to avoid one-frame spikes.
        - Returns: mean_score, unit_score, is_ng (bool)
        """
        if not unit_frames:
            raise ValueError("unit_frames must contain at least one frame; short units are still valid NG signals")

        n = len(unit_frames)
        trim_ratio = float(self._get("unit_trim_ratio", 0.20))
        trim = int(n * trim_ratio)
        if n - 2 * trim >= max(3, sample_step):
            core_frames = unit_frames[trim:n - trim]
        else:
            core_frames = unit_frames

        sampled = core_frames[::sample_step]
        if not sampled:
            sampled = core_frames

        frame_scores = []
        for _, full_frame, roi_patches_list in sampled:
            score, _ = self.score_frame(roi_patches_list if roi_patches_list else full_frame)
            frame_scores.append(score)

        mean_score = float(np.mean(frame_scores))
        percentile = float(self._get("unit_score_percentile", 80.0))
        unit_score = float(np.percentile(frame_scores, percentile))
        margin = float(self._get("unit_threshold_margin", 1.03))
        return mean_score, unit_score, unit_score > float(self.threshold) * margin

    def test(self) -> List[dict]:
        """
        Legacy task-level test entrypoint.  No longer maintained; use
        src/main.py directly for the full pipeline.
        """
        raise NotImplementedError(
            "detector.test() is deprecated.  Use 'python src/main.py <task>' instead."
        )

    def _load_training_patches_from_task_config(self) -> List[np.ndarray]:
        """
        Legacy auto-loading of training patches from task config.
        No longer maintained — pass train_patches explicitly to train().
        """
        raise NotImplementedError(
            "Auto-loading from task config is deprecated. "
            "Pass train_patches to detector.train() explicitly."
        )


class _MemoryView:
    """Tiny compatibility object for old code that reads detector.memory.threshold."""

    def __init__(self, memory_bank: np.ndarray, threshold: float, train_scores: np.ndarray) -> None:
        self.memory_bank = memory_bank
        self.threshold = threshold
        self.train_scores = train_scores.tolist()


def _taskA_yellow_ratio(roi_bgr: np.ndarray) -> float:
    """Yellow exposed-filter ratio for taskA cap-removal validation."""
    h, w = roi_bgr.shape[:2]
    r = max(8, int(min(h, w) * 0.32))
    cy, cx = h // 2, w // 2
    sub = roi_bgr[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r)]
    if sub.size == 0:
        return 0.0
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([10, 40, 40], dtype=np.uint8), np.array([45, 255, 255], dtype=np.uint8))
    return float(np.count_nonzero(mask)) / float(mask.size)
