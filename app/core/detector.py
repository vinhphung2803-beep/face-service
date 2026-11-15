"""Face Detection (chỉ phát hiện mặt; KHÔNG nhận diện danh tính, KHÔNG chống giả mạo)."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from app.core.common import ModelLoadError


@dataclass(frozen=True, eq=False)
class DetectedFace:
    """Một khuôn mặt, tọa độ theo ảnh đã đưa vào detector (ảnh đã resize).

    landmarks (5x2): mắt phải, mắt trái, đỉnh mũi, khóe miệng phải, khóe miệng trái
    (theo người trong ảnh, nên "mắt phải" nằm bên TRÁI khung hình).
    raw: dòng 15 giá trị gốc của YuNet, SFace cần đúng dòng này để căn chỉnh khuôn mặt.
    """

    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: np.ndarray
    raw: np.ndarray

    @property
    def size(self) -> float:
        return float(min(self.w, self.h))

    @classmethod
    def from_row(cls, row: np.ndarray) -> "DetectedFace":
        row = np.asarray(row, dtype=np.float32).ravel()
        if row.shape[0] != 15:
            raise ValueError("Dòng detection YuNet phải có 15 giá trị")
        return cls(
            x=float(row[0]),
            y=float(row[1]),
            w=float(row[2]),
            h=float(row[3]),
            score=float(row[14]),
            landmarks=row[4:14].reshape(5, 2).copy(),
            raw=row.copy(),
        )


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Trả mọi khuôn mặt vượt ngưỡng score; danh sách rỗng nếu không có."""


class YuNetDetector(FaceDetector):
    def __init__(self, model_path: Path, score_threshold: float, nms_threshold: float, top_k: int) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise ModelLoadError(f"Không tìm thấy model detector: {path}")
        try:
            self._det = cv2.FaceDetectorYN.create(
                str(path), "", (320, 320), score_threshold, nms_threshold, top_k
            )
        except cv2.error as exc:
            raise ModelLoadError(f"Không nạp được model detector {path.name}: {exc}") from exc
        self._lock = threading.Lock()  # setInputSize + detect là trạng thái chung
        self.model_name = path.name

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.size == 0:
            return []
        h, w = image_bgr.shape[:2]
        with self._lock:
            self._det.setInputSize((w, h))
            _, faces = self._det.detect(image_bgr)
        if faces is None:
            return []
        return [DetectedFace.from_row(row) for row in faces]


class DetectionStatus(str, Enum):
    NO_FACE = "NO_FACE"
    SINGLE_FACE = "SINGLE_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"


@dataclass(frozen=True, eq=False)
class DetectionOutcome:
    status: DetectionStatus
    face: DetectedFace | None  # chỉ có khi SINGLE_FACE
    total_detected: int
    counted: int  # số khuôn mặt "đủ lớn" được tính khi đếm


def classify_detection(faces: list[DetectedFace], count_min_px: int) -> DetectionOutcome:
    """Quy tắc đếm mặt. KHÔNG bao giờ tự chọn một mặt khi có nhiều người.

    - Mặt nhỏ hơn count_min_px bị coi là người ở xa/nền, không tính khi đếm.
    - Không có mặt nào đủ lớn:
        * không phát hiện gì            -> NO_FACE
        * đúng 1 mặt (nhỏ)              -> SINGLE_FACE (bước chất lượng sẽ báo FACE_TOO_SMALL)
        * nhiều mặt nhỏ                 -> MULTIPLE_FACES
    - Đúng 1 mặt đủ lớn                 -> SINGLE_FACE
    - Từ 2 mặt đủ lớn trở lên           -> MULTIPLE_FACES
    """
    total = len(faces)
    if total == 0:
        return DetectionOutcome(DetectionStatus.NO_FACE, None, 0, 0)
    significant = [f for f in faces if f.size >= count_min_px]
    if len(significant) == 1:
        return DetectionOutcome(DetectionStatus.SINGLE_FACE, significant[0], total, 1)
    if len(significant) >= 2:
        return DetectionOutcome(DetectionStatus.MULTIPLE_FACES, None, total, len(significant))
    if total == 1:
        return DetectionOutcome(DetectionStatus.SINGLE_FACE, faces[0], 1, 0)
    return DetectionOutcome(DetectionStatus.MULTIPLE_FACES, None, total, 0)
