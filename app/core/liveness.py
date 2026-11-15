"""Presentation Attack Detection (chống ảnh in / màn hình điện thoại).

Tách biệt hoàn toàn với Face Detection và Face Recognition. Hiện CHƯA có model liveness
đáng tin cậy nên backend duy nhất là NOT_CONFIGURED: không bao giờ trả PASSED khi chưa kiểm tra.
Nhận diện đúng khuôn mặt KHÔNG đồng nghĩa với người thật đang đứng trước camera.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.core.common import LivenessStatus
from app.core.detector import DetectedFace


@dataclass(frozen=True)
class LivenessResult:
    status: LivenessStatus
    detail: str = ""

    @property
    def passed(self) -> bool | None:
        """True/False chỉ khi đã thực sự kiểm tra; None nếu chưa cấu hình."""
        if self.status is LivenessStatus.PASSED:
            return True
        if self.status is LivenessStatus.FAILED:
            return False
        return None


class LivenessChecker(ABC):
    @abstractmethod
    def check(self, image_bgr: np.ndarray, face: DetectedFace) -> LivenessResult: ...


class NotConfiguredLivenessChecker(LivenessChecker):
    def check(self, image_bgr: np.ndarray, face: DetectedFace) -> LivenessResult:
        return LivenessResult(LivenessStatus.NOT_CONFIGURED, "Chưa cấu hình model liveness")


def build_liveness_checker(backend: str) -> LivenessChecker:
    if backend in ("", "none"):
        return NotConfiguredLivenessChecker()
    raise ValueError(
        f"LIVENESS_BACKEND='{backend}' chưa được hỗ trợ. Hãy cài một backend liveness thật "
        "(implement LivenessChecker) hoặc đặt LIVENESS_BACKEND=none."
    )
