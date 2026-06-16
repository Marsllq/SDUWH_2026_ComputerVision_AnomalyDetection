"""PatchCore memory-bank construction, threshold calibration, and scoring."""

from typing import List, Optional, Tuple

import numpy as np


def _as_frame_array(features: np.ndarray) -> np.ndarray:
    """Validate one frame/ROI feature matrix with shape (patches, dim)."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected frame features with shape (P, D), got {features.shape}")
    return features


def build_memory_bank(all_patches: np.ndarray, coreset_ratio: float = 0.10, seed: int = 42) -> np.ndarray:
    """
    Build PatchCore memory bank with coreset subsampling.
    - all_patches: np.ndarray (N_total, 256, 384) — per-frame patch features
    - CRITICAL: keep per-frame structure for threshold calibration
      (DO NOT concatenate all patches into one big matrix, score each frame independently)
    - Coreset: random sample coreset_ratio of patches
    - Returns: memory_bank (M, 384)
    """
    all_patches = np.asarray(all_patches, dtype=np.float32)
    if all_patches.ndim != 3:
        raise ValueError(f"Expected all_patches with shape (N, P, D), got {all_patches.shape}")
    if not 0 < coreset_ratio <= 1:
        raise ValueError("coreset_ratio must be in (0, 1]")

    # Flatten only for coreset sampling. Keep the original (N, P, D) input for
    # threshold calibration via calibrate_threshold().
    flat_patches = all_patches.reshape(-1, all_patches.shape[-1])
    n_total = flat_patches.shape[0]
    n_sample = max(1, int(n_total * coreset_ratio))
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_sample, replace=False)
    return flat_patches[indices].astype(np.float32, copy=False)


def _aggregate_patch_distances(per_patch_distance: np.ndarray, aggregation: str) -> float:
    """Aggregate patch distances into one frame-level anomaly score."""
    if per_patch_distance.size == 0:
        raise ValueError("Cannot score empty patch feature set")
    if aggregation == "max":
        return float(np.max(per_patch_distance))
    if aggregation == "mean_top5":
        top_n = min(5, per_patch_distance.size)
        return float(np.mean(np.sort(per_patch_distance)[-top_n:]))
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def score_frame(features: np.ndarray, memory_bank: np.ndarray, knn_k: int = 3, aggregation: str = "mean_top5") -> float:
    """
    Score a single frame (256 patches) against memory bank.
    Returns anomaly score (float).
    """
    frame_features = _as_frame_array(features)
    memory_bank = _as_frame_array(memory_bank)
    if memory_bank.shape[0] == 0:
        raise ValueError("memory_bank must not be empty")

    # Features are L2-normalized, so dot product is cosine similarity.
    sim = frame_features @ memory_bank.T

    # Specified scoring uses the best match per patch. knn_k is kept for API and
    # config compatibility; nearest-neighbor PatchCore distance is 1 - max_sim.
    max_sim = np.max(sim, axis=1)
    per_patch_distance = 1.0 - max_sim
    return _aggregate_patch_distances(per_patch_distance, aggregation)


def calibrate_threshold(
    per_frame_features: np.ndarray,
    memory_bank: np.ndarray,
    knn_k: int = 3,
    k_coeff: float = 3.0,
    aggregation: str = "mean_top5",
) -> Tuple[float, np.ndarray]:
    """
    Compute per-frame anomaly scores on training data, then set threshold = mu + k*sigma.

    For each frame (256 patches):
      - Compute similarity: sim = frame_feat @ memory_bank.T  (256, M)
      - max_sim = np.max(sim, axis=1)  best match per patch
      - per_patch_distance = 1.0 - max_sim
      - If aggregation=="max": frame_score = np.max(per_patch_distance)
      - If aggregation=="mean_top5": frame_score = np.mean(np.sort(per_patch_distance)[-5:])

    threshold = np.mean(frame_scores) + k_coeff * np.std(frame_scores)

    Returns: threshold, frame_scores (for histogram/dashboard)
    """
    per_frame_features = np.asarray(per_frame_features, dtype=np.float32)
    if per_frame_features.ndim != 3:
        raise ValueError(f"Expected per_frame_features with shape (N, P, D), got {per_frame_features.shape}")
    if per_frame_features.shape[0] == 0:
        raise ValueError("per_frame_features must contain at least one frame")

    frame_scores = np.array(
        [score_frame(frame_feat, memory_bank, knn_k=knn_k, aggregation=aggregation) for frame_feat in per_frame_features],
        dtype=np.float32,
    )
    threshold = float(np.mean(frame_scores) + k_coeff * np.std(frame_scores))
    return threshold, frame_scores


class PatchCoreMemoryBank:
    """Compatibility wrapper around the functional PatchCore API."""

    def __init__(
        self,
        coreset_ratio: float = 0.10,
        knn_k: int = 3,
        threshold_k: float = 3.0,
        score_aggregation: str = "mean_top5",
    ) -> None:
        self.coreset_ratio = coreset_ratio
        self.knn_k = knn_k
        self.threshold_k = threshold_k
        self.score_aggregation = score_aggregation
        self.memory_bank: Optional[np.ndarray] = None
        self.threshold: float = 0.0
        self.train_scores: List[float] = []

    def build(self, per_frame_patches: List[np.ndarray]) -> None:
        """Build memory bank and calibrate threshold from per-frame features."""
        if not per_frame_patches:
            raise ValueError("per_frame_patches must contain at least one frame")
        frame_array = np.asarray(per_frame_patches, dtype=np.float32)
        self.memory_bank = build_memory_bank(frame_array, coreset_ratio=self.coreset_ratio, seed=42)
        self.threshold, scores = calibrate_threshold(
            frame_array,
            self.memory_bank,
            knn_k=self.knn_k,
            k_coeff=self.threshold_k,
            aggregation=self.score_aggregation,
        )
        self.train_scores = scores.tolist()

    def score_frame(self, patch_features: np.ndarray) -> Tuple[float, bool]:
        """Score one frame and return (score, is_ng)."""
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is not built; call build() first")
        score = score_frame(
            patch_features,
            self.memory_bank,
            knn_k=self.knn_k,
            aggregation=self.score_aggregation,
        )
        return score, score > self.threshold

    def score_unit(self, unit_patches: List[np.ndarray]) -> Tuple[float, bool]:
        """Score a unit using max frame score to preserve short NG peaks."""
        if not unit_patches:
            raise ValueError("unit_patches must contain at least one frame")
        scores = [self.score_frame(patches)[0] for patches in unit_patches]
        max_score = float(np.max(scores))
        return max_score, max_score > self.threshold
