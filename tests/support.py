"""Tiện ích test. Dữ liệu ở đây là TỔNG HỢP (vector cơ sở, ảnh nhiễu), không phải khuôn mặt thật."""
from __future__ import annotations

import math

import numpy as np

from app.config import Settings
from app.core.detector import DetectedFace

SETTINGS = Settings()


def real_models_available() -> bool:
    return SETTINGS.detector_model_path.is_file() and SETTINGS.embedder_model_path.is_file()


def make_face(x=100.0, y=80.0, size=160.0, roll_deg=0.0, nose_dx=0.0, flip=False, score=0.95) -> DetectedFace:
    """Khuôn mặt giả với landmarks chính diện; có thể xoay (roll), lệch mũi (yaw), lộn ngược."""
    w = h = float(size)
    pts = np.array(
        [
            [x + 0.30 * w, y + 0.38 * h],  # mắt phải
            [x + 0.70 * w, y + 0.38 * h],  # mắt trái
            [x + (0.50 + nose_dx) * w, y + 0.58 * h],  # mũi
            [x + 0.35 * w, y + 0.78 * h],  # miệng phải
            [x + 0.65 * w, y + 0.78 * h],  # miệng trái
        ],
        dtype=np.float32,
    )
    c = np.array([x + w / 2, y + h / 2], dtype=np.float32)
    if flip:
        pts[:, 1] = 2 * c[1] - pts[:, 1]
    if roll_deg:
        a = math.radians(roll_deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=np.float32)
        pts = (pts - c) @ rot.T + c
    raw = np.concatenate([[x, y, w, h], pts.ravel(), [score]]).astype(np.float32)
    return DetectedFace.from_row(raw)


def textured_image(h=480, w=640, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def basis(i: int, dim: int = 128) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v
