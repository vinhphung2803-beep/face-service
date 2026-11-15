"""Kiểu dữ liệu và lỗi dùng chung cho toàn bộ service."""
from __future__ import annotations

from enum import Enum


class RecognitionStatus(str, Enum):
    MATCHED = "MATCHED"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    LOW_QUALITY = "LOW_QUALITY"
    LIVENESS_FAILED = "LIVENESS_FAILED"


class LivenessStatus(str, Enum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    PASSED = "PASSED"
    FAILED = "FAILED"


class ModelLoadError(RuntimeError):
    """Không nạp được file model (thiếu file, hỏng, sai định dạng)."""


class DimensionMismatchError(ValueError):
    """Embedding không cùng số chiều với index/model."""


class InputValidationError(ValueError):
    """Đầu vào không hợp lệ. `code` là mã máy đọc được, `message` là thông báo cho người dùng."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
