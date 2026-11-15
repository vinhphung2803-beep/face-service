"""Matching 1:N bằng cosine similarity trên embedding đã chuẩn hóa L2.

Điểm similarity KHÔNG phải xác suất đúng. Ngưỡng/margin mặc định chỉ là điểm khởi đầu
và phải hiệu chỉnh bằng dữ liệu thật (scripts/benchmark.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from app.config import Settings
from app.core.common import DimensionMismatchError, RecognitionStatus
from app.core.embedding import l2_normalize


@dataclass(frozen=True)
class MatcherConfig:
    threshold: float
    margin: float
    aggregation: str = "max"  # "max" | "topk_mean"
    top_k: int = 2
    top_n: int = 3  # số ứng viên giữ lại cho chẩn đoán/benchmark

    @classmethod
    def from_settings(cls, s: Settings) -> "MatcherConfig":
        return cls(s.match_threshold, s.match_margin, s.match_aggregation, s.match_top_k)


class Gallery:
    """Ảnh chụp bất biến (immutable snapshot) của index: các hàng được xếp liên tục theo nhân viên.
    Cập nhật index = tạo Gallery mới rồi hoán đổi tham chiếu, nên đọc luôn nhất quán, không cần khóa."""

    __slots__ = ("matrix", "codes", "starts", "dimension")

    def __init__(self, matrix: np.ndarray, codes: tuple[str, ...], starts: np.ndarray, dimension: int) -> None:
        self.matrix = matrix
        self.codes = codes
        self.starts = starts
        self.dimension = dimension

    @classmethod
    def empty(cls, dimension: int) -> "Gallery":
        m = np.zeros((0, dimension), dtype=np.float32)
        m.setflags(write=False)
        return cls(m, (), np.zeros(0, dtype=np.int64), dimension)

    @classmethod
    def from_samples(cls, samples: Iterable[tuple[str, Sequence[float]]], dimension: int) -> "Gallery":
        grouped: dict[str, list[np.ndarray]] = {}
        for code, emb in samples:
            v = np.asarray(emb, dtype=np.float32).ravel()
            if v.shape[0] != dimension:
                raise DimensionMismatchError(f"Embedding có {v.shape[0]} chiều, index cần {dimension}")
            grouped.setdefault(code, []).append(l2_normalize(v))
        if not grouped:
            return cls.empty(dimension)
        codes = tuple(grouped)
        rows: list[np.ndarray] = []
        starts: list[int] = []
        for code in codes:
            starts.append(len(rows))
            rows.extend(grouped[code])
        matrix = np.ascontiguousarray(np.vstack(rows), dtype=np.float32)
        matrix.setflags(write=False)
        return cls(matrix, codes, np.asarray(starts, dtype=np.int64), dimension)

    @property
    def size(self) -> int:
        """Số mẫu embedding."""
        return int(self.matrix.shape[0])

    @property
    def employee_count(self) -> int:
        return len(self.codes)


@dataclass(frozen=True)
class MatchResult:
    status: RecognitionStatus
    employee_code: Optional[str]  # CHỈ khác None khi MATCHED
    best_score: Optional[float]  # cosine similarity thô của ứng viên đầu (chẩn đoán)
    runner_up_score: Optional[float]
    top_candidates: tuple[tuple[str, float], ...]  # chỉ để chẩn đoán/benchmark, KHÔNG trả ra API công khai
    reason: str  # OK | BELOW_THRESHOLD | AMBIGUOUS_MARGIN | EMPTY_INDEX


def _aggregate(sims: np.ndarray, gallery: Gallery, cfg: MatcherConfig) -> np.ndarray:
    if cfg.aggregation == "max":
        return np.maximum.reduceat(sims, gallery.starts)
    ends = np.append(gallery.starts[1:], gallery.size)
    out = np.empty(gallery.employee_count, dtype=np.float32)
    for i, (a, b) in enumerate(zip(gallery.starts, ends)):
        seg = sims[a:b]
        k = min(cfg.top_k, seg.shape[0])
        out[i] = np.partition(seg, seg.shape[0] - k)[seg.shape[0] - k :].mean()
    return out


def match_embedding(query: Sequence[float], gallery: Gallery, cfg: MatcherConfig) -> MatchResult:
    q = l2_normalize(query)
    if q.shape[0] != gallery.dimension:
        raise DimensionMismatchError(f"Embedding truy vấn có {q.shape[0]} chiều, index cần {gallery.dimension}")
    if gallery.size == 0:
        return MatchResult(RecognitionStatus.UNKNOWN, None, None, None, (), "EMPTY_INDEX")

    scores = _aggregate(gallery.matrix @ q, gallery, cfg)
    order = np.argsort(-scores)
    top = tuple((gallery.codes[i], float(scores[i])) for i in order[: max(cfg.top_n, 2)])
    best_code, best = top[0]
    runner = top[1][1] if len(order) > 1 else None

    if best < cfg.threshold:
        return MatchResult(RecognitionStatus.UNKNOWN, None, best, runner, top, "BELOW_THRESHOLD")
    if runner is not None and (best - runner) < cfg.margin:
        return MatchResult(RecognitionStatus.AMBIGUOUS, None, best, runner, top, "AMBIGUOUS_MARGIN")
    return MatchResult(RecognitionStatus.MATCHED, best_code, best, runner, top, "OK")
