"""Orchestration layer for the Face Service API.

Raw uploaded images are processed in memory and are never persisted.
Embeddings are biometric templates and must be protected as sensitive data.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.core.common import InputValidationError, LivenessStatus, RecognitionStatus
from app.core.detector import DetectionStatus, YuNetDetector, classify_detection
from app.core.embedding import SFaceEmbedder
from app.core.liveness import build_liveness_checker
from app.core.matcher import Gallery, MatcherConfig, match_embedding
from app.core.quality import assess_quality
from app.utils.image_utils import decode_image, resize_max_side, sniff_image_type
from app.utils.validators import validate_employee_code, validate_upload


STORE_SCHEMA = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class FaceEngine:
    """Owns loaded models, biometric gallery and recognition/enrollment operations."""

    def __init__(self, settings: Settings, store_path: Path | None = None):
        self.settings = settings
        self.store_path = store_path or (Path(__file__).resolve().parents[2] / "data" / "enrollments.json")
        self.detector = YuNetDetector(
            settings.detector_model_path, settings.det_score_threshold,
            settings.det_nms_threshold, settings.det_top_k
        )
        self.embedder = SFaceEmbedder(settings.embedder_model_path, settings.model_version)
        self.liveness = build_liveness_checker(settings.liveness_backend)
        self.matcher_config = MatcherConfig.from_settings(settings)
        self._lock = threading.RLock()
        self.store = self._load_or_create_store()
        self.gallery = self._build_gallery()

    def _new_store(self) -> dict[str, Any]:
        return {
            "schemaVersion": STORE_SCHEMA,
            "modelVersion": self.settings.model_version,
            "embedderSha256": self.embedder.model_sha256,
            "dimension": self.embedder.dimension,
            "employees": {},
        }

    def _load_or_create_store(self) -> dict[str, Any]:
        if not self.store_path.exists():
            return self._new_store()
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InputValidationError("STORE_INVALID", f"Không đọc được kho embedding: {exc}") from exc
        if (not isinstance(data, dict) or data.get("schemaVersion") != STORE_SCHEMA
                or not isinstance(data.get("employees"), dict)):
            raise InputValidationError("STORE_INVALID", "Kho embedding sai schema")
        if data.get("modelVersion") != self.settings.model_version:
            raise InputValidationError("MODEL_VERSION_MISMATCH", "Kho embedding được tạo bởi modelVersion khác")
        if data.get("embedderSha256") != self.embedder.model_sha256:
            raise InputValidationError("MODEL_FILE_MISMATCH", "File SFace đã thay đổi; cần đăng ký lại khuôn mặt")
        if data.get("dimension") != self.embedder.dimension:
            raise InputValidationError("EMBEDDING_DIMENSION_MISMATCH", "Số chiều embedding trong kho không khớp")
        for code, employee in data["employees"].items():
            validate_employee_code(code)
            if not isinstance(employee, dict) or not isinstance(employee.get("samples"), list):
                raise InputValidationError("STORE_INVALID", f"Dữ liệu embedding của {code} không hợp lệ")
            for sample in employee["samples"]:
                self._validate_vector(sample.get("embedding"))
        return data

    def _validate_vector(self, values) -> np.ndarray:
        try:
            vec = np.asarray(values, dtype=np.float32).ravel()
        except (TypeError, ValueError) as exc:
            raise InputValidationError("EMBEDDING_INVALID", "Embedding không hợp lệ") from exc
        if vec.size != self.embedder.dimension or not np.all(np.isfinite(vec)):
            raise InputValidationError("EMBEDDING_INVALID", "Embedding sai số chiều hoặc chứa giá trị không hữu hạn")
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            raise InputValidationError("EMBEDDING_INVALID", "Embedding là vector không")
        return (vec / norm).astype(np.float32)

    def _save_store(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            dir=str(self.store_path.parent), prefix=self.store_path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.store, handle, ensure_ascii=False, separators=(",", ":"))
            os.replace(temp_path, self.store_path)
        except BaseException:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _build_gallery(self) -> Gallery:
        samples = []
        for code, employee in self.store["employees"].items():
            for sample in employee["samples"]:
                samples.append((code, sample["embedding"]))
        return Gallery.from_samples(samples, self.embedder.dimension)

    def _prepare_image(self, data: bytes, content_type: str | None):
        mime = sniff_image_type(data)
        validate_upload(data, mime or content_type, self.settings)
        image = decode_image(data)
        image, _ = resize_max_side(image, self.settings.max_image_side)
        return image

    def _analyze(self, data: bytes, content_type: str | None):
        image = self._prepare_image(data, content_type)
        faces = self.detector.detect(image)
        outcome = classify_detection(faces, self.settings.count_face_min_px)
        if outcome.status is DetectionStatus.NO_FACE:
            return {"status": "NO_FACE", "detail": "Không phát hiện khuôn mặt", "imageSize": [image.shape[1], image.shape[0]]}
        if outcome.status is DetectionStatus.MULTIPLE_FACES:
            return {"status": "MULTIPLE_FACES", "detail": "Phát hiện nhiều khuôn mặt; hệ thống không tự chọn một người", "imageSize": [image.shape[1], image.shape[0]], "faceCount": outcome.total_detected}
        face = outcome.face
        quality = assess_quality(image, face, self.settings)
        if not quality.passed:
            return {
                "status": "LOW_QUALITY", "detail": quality.failure_reason,
                "quality": {"passed": False, "failureReason": quality.failure_reason, "metrics": quality.metrics},
                "imageSize": [image.shape[1], image.shape[0]],
            }
        embedding = self.embedder.embed(image, face)
        live = self.liveness.check(image, face)
        return {
            "status": "READY", "embedding": embedding,
            "livenessStatus": live.status.value, "livenessDetail": live.detail,
            "quality": {"passed": True, "score": quality.score, "metrics": quality.metrics},
            "imageSize": [image.shape[1], image.shape[0]],
        }

    def enroll(self, employee_code: str, data: bytes, content_type: str | None, display_name: str | None = None):
        code = validate_employee_code(employee_code)
        result = self._analyze(data, content_type)
        if result["status"] != "READY":
            return {k: v for k, v in result.items() if k != "embedding"}
        if result["livenessStatus"] == LivenessStatus.FAILED.value:
            return {"status": "LIVENESS_FAILED", "detail": result["livenessDetail"]}
        with self._lock:
            employee = self.store["employees"].setdefault(
                code, {"displayName": display_name, "samples": []}
            )
            if display_name:
                employee["displayName"] = display_name
            if len(employee["samples"]) >= 10:
                raise InputValidationError("MAX_SAMPLES_REACHED", "Mỗi nhân viên tối đa 10 mẫu embedding")
            employee["samples"].append({
                "embedding": result["embedding"].astype(float).tolist(),
                "createdAt": utc_now_iso(),
            })
            self._save_store()
            self.gallery = self._build_gallery()
            count = len(employee["samples"])
        return {
            "status": "ENROLLED", "employeeCode": code, "sampleCount": count,
            "modelVersion": self.settings.model_version,
            "livenessStatus": result["livenessStatus"],
            "livenessDetail": result["livenessDetail"],
            "quality": result["quality"],
        }

    def recognize(self, data: bytes, content_type: str | None):
        result = self._analyze(data, content_type)
        if result["status"] != "READY":
            return {k: v for k, v in result.items() if k != "embedding"}
        if result["livenessStatus"] == LivenessStatus.FAILED.value:
            return {"status": "LIVENESS_FAILED", "livenessStatus": result["livenessStatus"],
                    "attendanceEligible": False, "detail": result["livenessDetail"]}
        with self._lock:
            gallery = self.gallery
        match = match_embedding(result["embedding"], gallery, self.matcher_config)
        matched = match.status is RecognitionStatus.MATCHED
        live_passed = result["livenessStatus"] == LivenessStatus.PASSED.value
        return {
            "status": match.status.value,
            "employeeCode": match.employee_code,
            "similarity": None if match.best_score is None else float(np.clip(match.best_score, -1.0, 1.0)),
            "runnerUpSimilarity": None if match.runner_up_score is None else float(np.clip(match.runner_up_score, -1.0, 1.0)),
            "reason": match.reason,
            "modelVersion": self.settings.model_version,
            "livenessStatus": result["livenessStatus"],
            "livenessDetail": result["livenessDetail"],
            "attendanceEligible": bool(matched and live_passed),
            "quality": result["quality"],
        }

    def remove_employee(self, employee_code: str) -> bool:
        code = validate_employee_code(employee_code)
        with self._lock:
            if code not in self.store["employees"]:
                return False
            del self.store["employees"][code]
            self._save_store()
            self.gallery = self._build_gallery()
            return True

    def stats(self):
        with self._lock:
            return {
                "employeeCount": self.gallery.employee_count,
                "sampleCount": self.gallery.size,
                "modelVersion": self.settings.model_version,
                "embeddingDimension": self.embedder.dimension,
                "livenessBackend": self.settings.liveness_backend,
            }
