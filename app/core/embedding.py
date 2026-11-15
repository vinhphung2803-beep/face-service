"""Trích xuất embedding khuôn mặt (Face Recognition). Cùng một lớp được dùng cho cả
đăng ký và nhận diện để đảm bảo cùng model và cùng tiền xử lý (căn chỉnh theo landmarks YuNet)."""
from __future__ import annotations

import hashlib
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from app.core.common import ModelLoadError
from app.core.detector import DetectedFace


def l2_normalize(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Embedding là vector không hoặc chứa NaN/vô cực")
    return (v / norm).astype(np.float32)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FaceEmbedder(ABC):
    """Giao diện để thay model. Đổi model => đổi model_version, index cũ bị từ chối."""

    model_version: str
    dimension: int

    @abstractmethod
    def embed(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Trả vector float32, độ dài `dimension`, đã chuẩn hóa L2."""


class SFaceEmbedder(FaceEmbedder):
    """SFace (OpenCV Zoo, Apache 2.0): mặt căn chỉnh 112x112 -> embedding 128 chiều."""

    def __init__(self, model_path: Path, model_version: str) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise ModelLoadError(f"Không tìm thấy model embedding: {path}")
        try:
            self._rec = cv2.FaceRecognizerSF.create(str(path), "")
            warm = self._rec.feature(np.zeros((112, 112, 3), dtype=np.uint8))  # khởi động + đo số chiều thật
        except cv2.error as exc:
            raise ModelLoadError(f"Không nạp được model embedding {path.name}: {exc}") from exc
        self.dimension = int(np.asarray(warm).size)
        self.model_version = model_version
        self.model_name = path.name
        self.model_sha256 = sha256_file(path)
        self._lock = threading.Lock()

    def embed(self, image_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        with self._lock:
            aligned = self._rec.alignCrop(image_bgr, face.raw)
            feat = self._rec.feature(aligned)
        vec = np.asarray(feat, dtype=np.float32).ravel()
        if vec.shape[0] != self.dimension:
            raise ValueError("Số chiều embedding không khớp với model đã nạp")
        return l2_normalize(vec)
