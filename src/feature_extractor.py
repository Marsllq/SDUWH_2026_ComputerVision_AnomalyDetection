"""DINOv2 patch-token feature extraction for PatchCore anomaly detection."""

from typing import List, Optional

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F


DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DINO_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_device(device: str = "auto") -> str:
    """Resolve auto device selection for local acceleration."""
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _preprocess_image(image: np.ndarray, input_size: int = 224) -> np.ndarray:
    """Convert one HWC BGR image to normalized CHW RGB float32 tensor data."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC BGR image with 3 channels, got {image.shape}")

    rgb = cv2.cvtColor(cv2.resize(image, (input_size, input_size)), cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - DINO_MEAN) / DINO_STD
    return np.transpose(rgb, (2, 0, 1))


def load_dinov2(model_name: str = "vit_small_patch14_dinov2", device: str = "cpu"):
    """
    Load DINOv2 from timm.
    - Use model.forward_features(x) NOT model(x)
    - model(x) returns pooled features (1,384), NOT patch tokens
    - model.forward_features(x) returns (1, 257, 384) — CLS + 256 patch tokens
    - Always skip CLS token with output[:, 1:, :]
    - Keep fixed 224x224 input; dynamic position embedding interpolation is
      slower and can hit unsupported MPS ops on macOS.
    - Do NOT call model.reset_classifier(0)
    """
    device = resolve_device(device)
    model = timm.create_model(model_name, pretrained=True, img_size=224, dynamic_img_size=False)
    model.eval().to(device)
    return model


@torch.no_grad()
def extract_patch_features(model, images: List[np.ndarray], device: str = "cpu") -> np.ndarray:
    """
    Extract L2-normalized patch tokens from a list of images.
    - images: list of np.ndarray (HWC BGR)
    - Convert each to RGB, resize to 224x224, normalize with DINOv2 mean/std
    - Stack to batch: (B, 3, 224, 224)
    - Forward through model.forward_features()
    - Skip CLS token, take patch tokens only
    - L2 normalize each patch token
    - Returns np.ndarray of shape (B, 256, 384)

    DINOv2 normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    """
    if not images:
        raise ValueError("images must contain at least one BGR image")

    batch = np.stack([_preprocess_image(image, input_size=224) for image in images], axis=0)
    device = resolve_device(device)
    tensor = torch.from_numpy(batch).float().to(device)

    # CRITICAL: forward_features returns CLS + patch tokens; model(x) does not.
    output = model.forward_features(tensor)
    patch_tokens = output[:, 1:, :]
    patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)
    return patch_tokens.cpu().numpy()


class DINOv2Extractor:
    """Small compatibility wrapper around the functional DINOv2 extractor API."""

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2",
        device: Optional[str] = None,
        input_size: int = 224,
    ) -> None:
        if device is None:
            device = "auto"
        self.device = resolve_device(device)
        self.input_size = input_size
        self.model = load_dinov2(model_name=model_name, device=device)

    @torch.no_grad()
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract patch features for a batch of BGR images: (B, 256, D)."""
        if self.input_size == 224:
            return extract_patch_features(self.model, images, device=self.device)

        batch = np.stack([_preprocess_image(image, input_size=self.input_size) for image in images], axis=0)
        tensor = torch.from_numpy(batch).float().to(self.device)
        output = self.model.forward_features(tensor)
        patch_tokens = F.normalize(output[:, 1:, :], p=2, dim=-1)
        return patch_tokens.cpu().numpy()

    def extract_patch_features(self, image: np.ndarray) -> np.ndarray:
        """Extract patch features for one BGR image: (256, D)."""
        return self.extract_batch([image])[0]

    def extract_multi_roi_features(self, roi_images: List[np.ndarray]) -> np.ndarray:
        """Extract and concatenate patch features from multiple ROI images."""
        features = self.extract_batch(roi_images)
        return features.reshape(-1, features.shape[-1])
