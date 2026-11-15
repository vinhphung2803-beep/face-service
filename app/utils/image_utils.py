"""Giải mã và tiền xử lý ảnh. Chỉ chấp nhận JPEG/PNG, kiểm tra kích thước từ header
TRƯỚC khi giải mã để tránh ảnh "bom pixel"."""
from __future__ import annotations

import cv2
import numpy as np

from app.core.common import InputValidationError

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def sniff_image_type(data: bytes) -> str | None:
    """Nhận dạng định dạng bằng magic bytes (không tin Content-Type do client gửi)."""
    if data.startswith(JPEG_MAGIC):
        return "image/jpeg"
    if data.startswith(PNG_MAGIC):
        return "image/png"
    return None


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:  # byte đệm
            i += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:  # marker không có độ dài
            i += 2
            continue
        if marker in (0xD9, 0xDA):  # hết ảnh / bắt đầu dữ liệu nén mà chưa thấy SOF
            return None
        if i + 4 > n:
            return None
        seg_len = int.from_bytes(data[i + 2 : i + 4], "big")
        if marker in sof_markers:
            if i + 9 > n:
                return None
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            return width, height
        if seg_len < 2:
            return None
        i += 2 + seg_len
    return None


def read_dimensions(data: bytes, mime: str) -> tuple[int, int] | None:
    """Trả (width, height) đọc từ header, hoặc None nếu header hỏng."""
    dims = _png_dimensions(data) if mime == "image/png" else _jpeg_dimensions(data)
    if dims is None or dims[0] <= 0 or dims[1] <= 0:
        return None
    return dims


def decode_image(data: bytes) -> np.ndarray:
    """Giải mã sang BGR uint8. OpenCV tự áp dụng hướng xoay EXIF (ảnh chụp từ điện thoại)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise InputValidationError("INVALID_IMAGE", "Không giải mã được ảnh (file hỏng hoặc sai định dạng)")
    return img


def resize_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Thu nhỏ nếu cạnh dài hơn max_side. Trả (ảnh, scale) với scale <= 1."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale
