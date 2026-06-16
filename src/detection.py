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


class PerRoiAnomalyDetector:
    """Independent PatchCore detector per ROI for multi-position taskB."""

    def __init__(self, config: Dict[str, Any], roi_count: int):
        self.config = dict(config)
        self.roi_count = int(roi_count)
        self.detectors = [AnomalyDetector(dict(config)) for _ in range(self.roi_count)]
        self.roles = list(self.config.get("taskB_roi_roles", []))
        self.color_thresholds: List[float] = [0.0 for _ in range(self.roi_count)]
        self.threshold = 0.0
        self.last_unit_debug: List[Dict[str, Any]] = []

    def train(self, roi_train_patches: List[List[np.ndarray]]) -> None:
        if len(roi_train_patches) != self.roi_count:
            raise ValueError(f"Expected {self.roi_count} ROI training groups, got {len(roi_train_patches)}")
        thresholds = []
        for idx, patches in enumerate(roi_train_patches):
            if not patches:
                raise ValueError(f"ROI #{idx} has no training patches")
            self.detectors[idx].train(patches)
            thresholds.append(float(self.detectors[idx].threshold))
            role = self._role(idx)
            color_ratios = np.array([_taskB_role_ratio(p, role) for p in patches if p is not None and p.size > 0], dtype=np.float32)
            if color_ratios.size:
                k = float(self.config.get("taskB_color_threshold_k", self.config.get("taskB_blue_threshold_k", 2.5)))
                self.color_thresholds[idx] = max(self._role_floor(idx), float(np.mean(color_ratios) - k * np.std(color_ratios)))
        self.threshold = float(np.max(thresholds))

    def score_rois(self, roi_patches: List[np.ndarray]) -> List[Tuple[int, float, bool]]:
        scores: List[Tuple[int, float, bool]] = []
        margin = float(self.config.get("unit_threshold_margin", 1.03))
        for i, patch in enumerate(roi_patches[: self.roi_count]):
            score, _ = self.detectors[i].score_frame([patch])
            color_ratio = _taskB_role_ratio(patch, self._role(i))
            is_ng = score > float(self.detectors[i].threshold) * margin or color_ratio < self.color_thresholds[i]
            scores.append((i, float(score), bool(is_ng)))
        return scores

    def score_frame(self, roi_patches: Union[np.ndarray, List[np.ndarray]]) -> Tuple[float, bool]:
        rois = roi_patches if isinstance(roi_patches, list) else [roi_patches]
        scores = self.score_rois(rois)
        if not scores:
            return 0.0, False
        return float(max(s for _, s, _ in scores)), any(is_ng for _, _, is_ng in scores)

    def score_unit(self, unit_frames: List[Tuple[int, np.ndarray, List[np.ndarray]]], sample_step: int = 5) -> Tuple[float, float, bool]:
        if not unit_frames:
            raise ValueError("unit_frames must contain at least one frame")
        n = len(unit_frames)
        trim_ratio = float(self.config.get("unit_trim_ratio", 0.20))
        trim = int(n * trim_ratio)
        if n - 2 * trim >= max(3, sample_step):
            core_frames = unit_frames[trim:n - trim]
        else:
            core_frames = unit_frames
        sampled = core_frames[::sample_step] or core_frames

        per_roi_scores: List[List[float]] = [[] for _ in range(self.roi_count)]
        per_roi_color_ng: List[List[bool]] = [[] for _ in range(self.roi_count)]
        for _, _, roi_patches in sampled:
            for i, patch in enumerate(roi_patches[: self.roi_count]):
                score, _ = self.detectors[i].score_frame([patch])
                per_roi_scores[i].append(score)
                color_ratio = _taskB_role_ratio(patch, self._role(i))
                per_roi_color_ng[i].append(color_ratio < self.color_thresholds[i])

        percentile = float(self.config.get("unit_score_percentile", 80.0))
        margin = float(self.config.get("unit_threshold_margin", 1.03))
        presence_vote_ratio = float(self.config.get("taskB_presence_ng_vote_ratio", 0.30))
        roi_unit_scores = []
        roi_ng = []
        self.last_unit_debug = []
        for i, scores in enumerate(per_roi_scores):
            if not scores:
                continue
            unit_score = float(np.percentile(scores, percentile))
            roi_unit_scores.append(unit_score)
            patchcore_ng = unit_score > float(self.detectors[i].threshold) * margin
            presence_ratio = float(np.mean(per_roi_color_ng[i])) if per_roi_color_ng[i] else 0.0
            presence_ng = bool(per_roi_color_ng[i]) and presence_ratio >= presence_vote_ratio
            roi_ng.append(patchcore_ng or presence_ng)
            self.last_unit_debug.append({
                "roi": i,
                "role": self._role(i),
                "score": unit_score,
                "threshold": float(self.detectors[i].threshold) * margin,
                "presence_ratio": presence_ratio,
                "patchcore_ng": patchcore_ng,
                "presence_ng": presence_ng,
            })

        if not roi_unit_scores:
            return 0.0, 0.0, False
        mean_score = float(np.mean(roi_unit_scores))
        max_score = float(np.max(roi_unit_scores))
        return mean_score, max_score, any(roi_ng)

    def _role(self, idx: int) -> str:
        return self.roles[idx] if idx < len(self.roles) else "generic"

    def _role_floor(self, idx: int) -> float:
        role = self._role(idx)
        if role == "blue":
            return float(self.config.get("taskB_blue_presence_min", self.config.get("taskB_blue_min_ratio", 0.015)))
        if role == "white":
            return float(self.config.get("taskB_white_presence_min", 0.35))
        return 0.02


class TaskBCapTemplateDetector:
    """TaskB detector based on four cap-presence checks and template matching."""

    def __init__(self, config: Dict[str, Any], roi_count: int):
        self.config = dict(config)
        self.roi_count = int(roi_count)
        self.roles = list(self.config.get("taskB_roi_roles", []))
        self.template_size = tuple(self.config.get("taskB_template_size", [96, 64]))
        self.sim_thresholds: List[float] = [0.0 for _ in range(self.roi_count)]
        self.color_thresholds: List[float] = [0.0 for _ in range(self.roi_count)]
        self.templates: List[np.ndarray] = []
        self.threshold = 1.0

    def train(self, roi_train_patches: List[List[np.ndarray]]) -> None:
        if len(roi_train_patches) != self.roi_count:
            raise ValueError(f"Expected {self.roi_count} ROI groups, got {len(roi_train_patches)}")

        self.templates = []
        for i, patches in enumerate(roi_train_patches):
            if not patches:
                raise ValueError(f"ROI #{i} has no training patches")
            prepped = [_prep_template_patch(p, self.template_size) for p in patches if p is not None and p.size > 0]
            if not prepped:
                raise ValueError(f"ROI #{i} has no valid training patches")

            template = np.median(np.stack(prepped, axis=0), axis=0).astype(np.uint8)
            self.templates.append(template)

            sims = np.array([_template_similarity(p, template, self.template_size) for p in patches], dtype=np.float32)
            sim_k = float(self.config.get("taskB_template_threshold_k", 3.0))
            sim_floor = float(self.config.get("taskB_template_min_similarity", 0.45))
            self.sim_thresholds[i] = max(sim_floor, float(np.mean(sims) - sim_k * np.std(sims)))

            color_vals = np.array([_taskB_role_ratio(p, self._role(i)) for p in patches], dtype=np.float32)
            color_k = float(self.config.get("taskB_color_threshold_k", 2.5))
            role_floor = self._role_floor(i)
            self.color_thresholds[i] = max(role_floor, float(np.mean(color_vals) - color_k * np.std(color_vals)))

        self.threshold = float(max(1.0 - t for t in self.sim_thresholds))

    def score_rois(self, roi_patches: List[np.ndarray]) -> List[Tuple[int, float, bool]]:
        scores: List[Tuple[int, float, bool]] = []
        for i, patch in enumerate(roi_patches[: self.roi_count]):
            if i >= len(self.templates):
                continue
            sim = _template_similarity(patch, self.templates[i], self.template_size)
            color_ratio = _taskB_role_ratio(patch, self._role(i))
            is_ng = sim < self.sim_thresholds[i] or color_ratio < self.color_thresholds[i]
            score = max(1.0 - sim, max(0.0, self.color_thresholds[i] - color_ratio))
            scores.append((i, float(score), bool(is_ng)))
        return scores

    def score_frame(self, roi_patches: Union[np.ndarray, List[np.ndarray]]) -> Tuple[float, bool]:
        rois = roi_patches if isinstance(roi_patches, list) else [roi_patches]
        scores = self.score_rois(rois)
        if not scores:
            return 0.0, False
        return float(max(s for _, s, _ in scores)), any(is_ng for _, _, is_ng in scores)

    def score_unit(self, unit_frames: List[Tuple[int, np.ndarray, List[np.ndarray]]], sample_step: int = 5) -> Tuple[float, float, bool]:
        if not unit_frames:
            raise ValueError("unit_frames must contain at least one frame")
        n = len(unit_frames)
        trim_ratio = float(self.config.get("unit_trim_ratio", 0.20))
        trim = int(n * trim_ratio)
        core_frames = unit_frames[trim:n - trim] if n - 2 * trim >= max(3, sample_step) else unit_frames
        sampled = core_frames[::sample_step] or core_frames

        per_roi_scores: List[List[float]] = [[] for _ in range(self.roi_count)]
        per_roi_ng: List[List[bool]] = [[] for _ in range(self.roi_count)]
        for _, _, roi_patches in sampled:
            for i, score, is_ng in self.score_rois(roi_patches):
                per_roi_scores[i].append(score)
                per_roi_ng[i].append(is_ng)

        roi_unit_scores = []
        roi_votes = []
        vote_ratio = float(self.config.get("taskB_ng_vote_ratio", 0.35))
        percentile = float(self.config.get("unit_score_percentile", 95.0))
        for i in range(self.roi_count):
            if not per_roi_scores[i]:
                continue
            roi_unit_scores.append(float(np.percentile(per_roi_scores[i], percentile)))
            roi_votes.append(float(np.mean(per_roi_ng[i])) >= vote_ratio)

        if not roi_unit_scores:
            return 0.0, 0.0, False
        mean_score = float(np.mean(roi_unit_scores))
        max_score = float(np.max(roi_unit_scores))
        return mean_score, max_score, any(roi_votes)

    def _role(self, idx: int) -> str:
        return self.roles[idx] if idx < len(self.roles) else "generic"

    def _role_floor(self, idx: int) -> float:
        role = self._role(idx)
        if role == "blue":
            return float(self.config.get("taskB_blue_presence_min", 0.18))
        if role == "white":
            return float(self.config.get("taskB_white_presence_min", 0.35))
        return 0.02

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


def _taskB_blue_ratio(roi_bgr: np.ndarray) -> float:
    """Blue cap presence ratio for taskB key ROIs."""
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 45, 35], dtype=np.uint8), np.array([135, 255, 255], dtype=np.uint8))
    return float(np.count_nonzero(mask)) / float(mask.size)


def _taskB_role_ratio(roi_bgr: np.ndarray, role: str) -> float:
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    if role == "blue":
        mask = cv2.inRange(hsv, np.array([90, 45, 35], dtype=np.uint8), np.array([135, 255, 255], dtype=np.uint8))
    elif role == "white":
        mask = cv2.inRange(hsv, np.array([0, 0, 110], dtype=np.uint8), np.array([179, 95, 255], dtype=np.uint8))
    else:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 60, 255)
    return float(np.count_nonzero(mask)) / float(mask.size)


def _prep_template_patch(roi_bgr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(resized)


def _template_similarity(roi_bgr: np.ndarray, template: np.ndarray, size: Tuple[int, int]) -> float:
    if roi_bgr is None or roi_bgr.size == 0:
        return -1.0
    patch = _prep_template_patch(roi_bgr, size)
    result = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
    return float(result[0, 0])
