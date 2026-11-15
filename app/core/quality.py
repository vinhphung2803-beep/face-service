"""Kiểm tra chất lượng khuôn mặt bằng các đại lượng đo được.

LƯU Ý TRUNG THỰC:
- Mọi ngưỡng là điểm khởi đầu, chưa hiệu chỉnh cho camera/môi trường cụ thể.
- Góc nghiêng ước lượng từ 5 landmarks (heuristic, không phải đo góc thật).
- "Che khuất" chỉ được kiểm tra gián tiếp (mặt bị cắt khỏi khung, landmarks vô lý).
  Khẩu trang/kính/tay che mặt KHÔNG được phát hiện đáng tin cậy ở bước này.
- qualityScore là chỉ số "độ dư so với ngưỡng" (0 = sát ngưỡng, 1 = dư nhiều), KHÔNG phải xác suất.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import Settings
from app.core.detector import DetectedFace

_CROP = 112


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    score: float | None
    failure_reason: str | None
    metrics: dict[str, float] = field(default_factory=dict)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def pose_from_landmarks(landmarks: np.ndarray) -> dict[str, float] | None:
    """Trả roll_deg, yaw_ratio, pitch_ratio; None nếu landmarks suy biến/vô lý (vd ảnh lộn ngược)."""
    re, le, nose, rm, lm = (np.asarray(p, dtype=np.float64) for p in landmarks)
    eye_vec = le - re
    eye_dist = float(np.linalg.norm(eye_vec))
    eye_mid = (re + le) / 2.0
    mouth_mid = (rm + lm) / 2.0
    axis = mouth_mid - eye_mid
    axis_len2 = float(axis @ axis)
    if eye_dist < 1e-6 or axis_len2 < 1e-6:
        return None
    if eye_vec[0] <= 0:  # mắt phải phải nằm bên trái khung hình
        return None
    if axis[1] <= 0:  # miệng phải nằm dưới mắt
        return None
    return {
        "roll_deg": math.degrees(math.atan2(eye_vec[1], eye_vec[0])),
        "yaw_ratio": float((nose[0] - eye_mid[0]) / eye_dist),
        "pitch_ratio": float(((nose - eye_mid) @ axis) / axis_len2),
    }


def assess_quality(image: np.ndarray, face: DetectedFace, s: Settings) -> QualityResult:
    h_img, w_img = image.shape[:2]
    metrics: dict[str, float] = {"face_size_px": face.size, "detector_score": face.score}
    reasons: list[str] = []

    if face.size < s.min_face_px:
        reasons.append("FACE_TOO_SMALL")

    x0, y0, x1, y1 = face.x, face.y, face.x + face.w, face.y + face.h
    overflow = max(0.0, -x0, -y0, x1 - w_img, y1 - h_img) / max(face.w, face.h, 1e-6)
    metrics["overflow_ratio"] = float(overflow)
    if overflow > s.max_face_overflow_ratio:
        reasons.append("FACE_PARTIALLY_OUTSIDE")

    lm = face.landmarks
    inside = bool(np.all((lm[:, 0] >= 0) & (lm[:, 0] <= w_img) & (lm[:, 1] >= 0) & (lm[:, 1] <= h_img)))
    pose = pose_from_landmarks(lm) if inside else None
    if pose is None:
        reasons.append("LANDMARKS_INCONSISTENT")
    else:
        metrics.update(pose)

    cx0, cy0 = max(0, int(x0)), max(0, int(y0))
    cx1, cy1 = min(w_img, int(math.ceil(x1))), min(h_img, int(math.ceil(y1)))
    if cx1 - cx0 < 2 or cy1 - cy0 < 2:
        reasons.append("FACE_PARTIALLY_OUTSIDE")
        return QualityResult(False, None, reasons[0], metrics)

    gray = cv2.cvtColor(image[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (_CROP, _CROP), interpolation=cv2.INTER_AREA)  # độ nét so sánh được giữa các cỡ mặt
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["brightness"] = brightness
    metrics["sharpness"] = sharpness

    if brightness < s.min_brightness:
        reasons.append("TOO_DARK")
    elif brightness > s.max_brightness:
        reasons.append("TOO_BRIGHT")
    if sharpness < s.min_sharpness:
        reasons.append("TOO_BLURRY")
    if pose is not None:
        if abs(pose["roll_deg"]) > s.max_roll_deg:
            reasons.append("POSE_ROLL")
        if abs(pose["yaw_ratio"]) > s.max_yaw_ratio:
            reasons.append("POSE_YAW")
        if not s.pitch_ratio_min <= pose["pitch_ratio"] <= s.pitch_ratio_max:
            reasons.append("POSE_PITCH")

    # Chỉ số headroom: min các thành phần, mỗi thành phần 0 (sát ngưỡng) .. 1 (dư nhiều).
    mid_b = (s.min_brightness + s.max_brightness) / 2.0
    half_b = (s.max_brightness - s.min_brightness) / 2.0
    parts = [
        _clip01((face.size - s.min_face_px) / s.min_face_px),
        _clip01((sharpness - s.min_sharpness) / s.min_sharpness),
        _clip01(1.0 - abs(brightness - mid_b) / half_b),
        _clip01(1.0 - overflow / max(s.max_face_overflow_ratio, 1e-6)),
    ]
    if pose is not None:
        mid_p = (s.pitch_ratio_min + s.pitch_ratio_max) / 2.0
        half_p = (s.pitch_ratio_max - s.pitch_ratio_min) / 2.0
        parts += [
            _clip01(1.0 - abs(pose["roll_deg"]) / s.max_roll_deg),
            _clip01(1.0 - abs(pose["yaw_ratio"]) / s.max_yaw_ratio),
            _clip01(1.0 - abs(pose["pitch_ratio"] - mid_p) / half_p),
        ]
    score = float(min(parts))
    return QualityResult(not reasons, score, reasons[0] if reasons else None, metrics)
