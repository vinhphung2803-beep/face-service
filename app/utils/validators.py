"""Kiểm tra đầu vào: file ảnh upload, employeeCode, payload embedding."""
from __future__ import annotations

import math
import re
from typing import Sequence

import numpy as np

from app.config import Settings
from app.core.common import DimensionMismatchError, InputValidationError
from app.utils.image_utils import read_dimensions, sniff_image_type

EMPLOYEE_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")


def validate_upload(data: bytes, content_type: str | None, settings: Settings) -> str:
    """Kiểm tra file ảnh upload. Trả về MIME thực (theo magic bytes)."""
    if not data:
        raise InputValidationError("EMPTY_FILE", "File ảnh rỗng")
    if len(data) > settings.max_upload_bytes:
        raise InputValidationError(
            "FILE_TOO_LARGE", f"File vượt quá giới hạn {settings.max_upload_bytes // 1024} KB"
        )
    declared = (content_type or "").split(";")[0].strip().lower()
    if declared not in settings.allowed_mime_types:
        raise InputValidationError("UNSUPPORTED_MIME", "Chỉ chấp nhận ảnh JPEG hoặc PNG")
    mime = sniff_image_type(data)
    if mime is None or mime not in settings.allowed_mime_types:
        raise InputValidationError("UNSUPPORTED_MIME", "Nội dung file không phải ảnh JPEG/PNG hợp lệ")
    dims = read_dimensions(data, mime)
    if dims is None:
        raise InputValidationError("INVALID_IMAGE", "Header ảnh hỏng hoặc không đọc được kích thước")
    width, height = dims
    if width * height > settings.max_image_pixels:
        raise InputValidationError("IMAGE_TOO_LARGE", "Độ phân giải ảnh quá lớn")
    if min(width, height) < settings.min_input_side:
        raise InputValidationError(
            "IMAGE_TOO_SMALL", f"Ảnh quá nhỏ (cạnh ngắn tối thiểu {settings.min_input_side}px)"
        )
    return mime


def validate_employee_code(code: object) -> str:
    if not isinstance(code, str) or not EMPLOYEE_CODE_RE.match(code):
        raise InputValidationError(
            "EMPLOYEE_CODE_INVALID",
            "employeeCode chỉ gồm chữ, số, '_', '-', '.' và dài tối đa 32 ký tự",
        )
    return code


def validate_embedding_payload(
    values: Sequence[float],
    *,
    expected_dimension: int,
    expected_model_version: str,
    model_version: str,
) -> np.ndarray:
    """Kiểm tra một embedding do hệ thống nội bộ gửi lên. Trả vector float32 đã chuẩn hóa L2."""
    if model_version != expected_model_version:
        raise InputValidationError(
            "MODEL_VERSION_MISMATCH",
            f"modelVersion '{model_version}' không khớp với service hiện tại; cần đăng ký lại khuôn mặt",
        )
    try:
        vec = np.asarray(values, dtype=np.float32).ravel()
    except (TypeError, ValueError) as exc:
        raise InputValidationError("EMBEDDING_INVALID", "Embedding phải là danh sách số") from exc
    if vec.shape[0] != expected_dimension:
        raise InputValidationError(
            "EMBEDDING_DIMENSION_MISMATCH",
            f"Embedding phải có {expected_dimension} chiều, nhận được {vec.shape[0]}",
        )
    if not np.all(np.isfinite(vec)):
        raise InputValidationError("EMBEDDING_INVALID", "Embedding chứa NaN hoặc vô cực")
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm < 1e-6:
        raise InputValidationError("EMBEDDING_INVALID", "Embedding là vector không")
    return (vec / norm).astype(np.float32)


__all__ = [
    "validate_upload",
    "validate_employee_code",
    "validate_embedding_payload",
    "DimensionMismatchError",
]
