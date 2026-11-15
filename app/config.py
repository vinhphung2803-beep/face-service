"""Cấu hình đọc từ biến môi trường / file .env. Không hardcode bí mật trong mã nguồn."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Đọc .env đơn giản (KEY=VALUE). Biến môi trường đã có sẵn luôn được ưu tiên."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _get(name: str, cast, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Biến môi trường {name} không hợp lệ: {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    # Model
    model_dir: Path = BASE_DIR / "models"
    detector_model_file: str = "face_detection_yunet_2023mar.onnx"
    embedder_model_file: str = "face_recognition_sface_2021dec.onnx"
    model_version: str = "yunet-2023mar+sface-2021dec"

    # Detection
    det_score_threshold: float = 0.7
    det_nms_threshold: float = 0.3
    det_top_k: int = 50
    max_image_side: int = 960
    count_face_min_px: int = 40

    # Giới hạn ảnh đầu vào
    max_upload_bytes: int = 5 * 1024 * 1024
    max_image_pixels: int = 25_000_000
    min_input_side: int = 100
    allowed_mime_types: tuple[str, ...] = ("image/jpeg", "image/png")

    # Chất lượng (heuristic, chưa hiệu chỉnh)
    min_face_px: int = 80
    max_face_overflow_ratio: float = 0.10
    min_brightness: float = 60.0
    max_brightness: float = 200.0
    min_sharpness: float = 40.0
    max_roll_deg: float = 25.0
    max_yaw_ratio: float = 0.35
    pitch_ratio_min: float = 0.30
    pitch_ratio_max: float = 0.75

    # Matching
    match_threshold: float = 0.363
    match_margin: float = 0.05
    match_aggregation: str = "max"
    match_top_k: int = 2

    # Liveness
    liveness_backend: str = "none"

    @property
    def detector_model_path(self) -> Path:
        return self.model_dir / self.detector_model_file

    @property
    def embedder_model_path(self) -> Path:
        return self.model_dir / self.embedder_model_file

    def validate(self) -> "Settings":
        if not -1.0 <= self.match_threshold <= 1.0:
            raise ValueError("MATCH_THRESHOLD phải nằm trong [-1, 1] (cosine similarity)")
        if not 0.0 <= self.match_margin <= 1.0:
            raise ValueError("MATCH_MARGIN phải nằm trong [0, 1]")
        if self.match_aggregation not in ("max", "topk_mean"):
            raise ValueError("MATCH_AGGREGATION phải là 'max' hoặc 'topk_mean'")
        if self.match_top_k < 1:
            raise ValueError("MATCH_TOP_K phải >= 1")
        if self.min_brightness >= self.max_brightness:
            raise ValueError("MIN_BRIGHTNESS phải nhỏ hơn MAX_BRIGHTNESS")
        if self.pitch_ratio_min >= self.pitch_ratio_max:
            raise ValueError("PITCH_RATIO_MIN phải nhỏ hơn PITCH_RATIO_MAX")
        if self.min_face_px < self.count_face_min_px:
            raise ValueError("MIN_FACE_PX phải >= COUNT_FACE_MIN_PX")
        if self.max_image_side < 64:
            raise ValueError("MAX_IMAGE_SIDE quá nhỏ")
        return self

    @classmethod
    def from_env(cls, dotenv: Path | None = None) -> "Settings":
        _load_dotenv(dotenv or BASE_DIR / ".env")
        model_dir = _get("FACE_MODEL_DIR", Path, Path("models"))
        if not model_dir.is_absolute():
            model_dir = BASE_DIR / model_dir
        d = cls()
        return cls(
            model_dir=model_dir,
            detector_model_file=_get("FACE_DETECTOR_MODEL", str, d.detector_model_file),
            embedder_model_file=_get("FACE_EMBEDDER_MODEL", str, d.embedder_model_file),
            model_version=_get("FACE_MODEL_VERSION", str, d.model_version),
            det_score_threshold=_get("DET_SCORE_THRESHOLD", float, d.det_score_threshold),
            det_nms_threshold=_get("DET_NMS_THRESHOLD", float, d.det_nms_threshold),
            det_top_k=_get("DET_TOP_K", int, d.det_top_k),
            max_image_side=_get("MAX_IMAGE_SIDE", int, d.max_image_side),
            count_face_min_px=_get("COUNT_FACE_MIN_PX", int, d.count_face_min_px),
            max_upload_bytes=_get("MAX_UPLOAD_BYTES", int, d.max_upload_bytes),
            max_image_pixels=_get("MAX_IMAGE_PIXELS", int, d.max_image_pixels),
            min_input_side=_get("MIN_INPUT_SIDE", int, d.min_input_side),
            min_face_px=_get("MIN_FACE_PX", int, d.min_face_px),
            max_face_overflow_ratio=_get("MAX_FACE_OVERFLOW_RATIO", float, d.max_face_overflow_ratio),
            min_brightness=_get("MIN_BRIGHTNESS", float, d.min_brightness),
            max_brightness=_get("MAX_BRIGHTNESS", float, d.max_brightness),
            min_sharpness=_get("MIN_SHARPNESS", float, d.min_sharpness),
            max_roll_deg=_get("MAX_ROLL_DEG", float, d.max_roll_deg),
            max_yaw_ratio=_get("MAX_YAW_RATIO", float, d.max_yaw_ratio),
            pitch_ratio_min=_get("PITCH_RATIO_MIN", float, d.pitch_ratio_min),
            pitch_ratio_max=_get("PITCH_RATIO_MAX", float, d.pitch_ratio_max),
            match_threshold=_get("MATCH_THRESHOLD", float, d.match_threshold),
            match_margin=_get("MATCH_MARGIN", float, d.match_margin),
            match_aggregation=_get("MATCH_AGGREGATION", str, d.match_aggregation),
            match_top_k=_get("MATCH_TOP_K", int, d.match_top_k),
            liveness_backend=_get("LIVENESS_BACKEND", str, d.liveness_backend).lower(),
        ).validate()
