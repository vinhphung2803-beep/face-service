import unittest

import numpy as np

from app.core.common import DimensionMismatchError, RecognitionStatus as S
from app.core.matcher import Gallery, MatcherConfig, match_embedding
from tests.support import basis

CFG = MatcherConfig(threshold=0.363, margin=0.05, aggregation="max", top_k=2)


def unit(*pairs):
    """Tổ hợp tuyến tính của các vector cơ sở: unit((0,1.0),(1,0.3))."""
    v = np.zeros(128, dtype=np.float32)
    for i, w in pairs:
        v[i] += w
    return v / np.linalg.norm(v)


class MatcherTests(unittest.TestCase):
    """Dùng vector cơ sở trực giao (tổng hợp) để kết quả cosine là chính xác và lặp lại được.
    Không phản ánh độ chính xác trên khuôn mặt thật."""

    def gallery(self):
        return Gallery.from_samples([("NV001", basis(0)), ("NV002", basis(1)), ("NV003", basis(2))], 128)

    def test_registered_person_matched(self):
        r = match_embedding(unit((1, 1.0), (50, 0.3)), self.gallery(), CFG)
        self.assertEqual(r.status, S.MATCHED)
        self.assertEqual(r.employee_code, "NV002")
        self.assertGreater(r.best_score, 0.9)

    def test_unregistered_person_is_unknown_and_has_no_code(self):
        r = match_embedding(basis(100), self.gallery(), CFG)
        self.assertEqual(r.status, S.UNKNOWN)
        self.assertIsNone(r.employee_code)
        self.assertEqual(r.reason, "BELOW_THRESHOLD")

    def test_threshold_is_configurable(self):
        q = unit((0, 1.0), (100, 1.5))  # cosine với NV001 ~ 0.55
        loose = match_embedding(q, self.gallery(), MatcherConfig(0.4, 0.05))
        strict = match_embedding(q, self.gallery(), MatcherConfig(0.7, 0.05))
        self.assertEqual(loose.status, S.MATCHED)
        self.assertEqual(strict.status, S.UNKNOWN)
        self.assertIsNone(strict.employee_code)

    def test_two_similar_employees_are_ambiguous(self):
        r = match_embedding(unit((0, 1.0), (1, 1.0)), self.gallery(), CFG)
        self.assertEqual(r.status, S.AMBIGUOUS)
        self.assertIsNone(r.employee_code)
        self.assertEqual(r.reason, "AMBIGUOUS_MARGIN")

    def test_margin_is_configurable(self):
        q = unit((0, 1.0), (1, 0.9))  # chênh ~0.074
        self.assertEqual(match_embedding(q, self.gallery(), MatcherConfig(0.363, 0.05)).status, S.MATCHED)
        self.assertEqual(match_embedding(q, self.gallery(), MatcherConfig(0.363, 0.10)).status, S.AMBIGUOUS)

    def test_multiple_samples_per_employee_does_not_count_as_competitor(self):
        g = Gallery.from_samples([("NV001", unit((0, 1.0), (60, 0.2))), ("NV001", unit((0, 1.0), (61, 0.2))), ("NV002", basis(1))], 128)
        r = match_embedding(basis(0), g, CFG)
        self.assertEqual(r.status, S.MATCHED)
        self.assertEqual(r.employee_code, "NV001")
        self.assertEqual(g.size, 3)
        self.assertEqual(g.employee_count, 2)

    def test_aggregation_max_vs_topk_mean(self):
        g = Gallery.from_samples([("NV001", basis(0)), ("NV001", basis(10)), ("NV001", basis(11))], 128)
        q = basis(0)
        self.assertEqual(match_embedding(q, g, MatcherConfig(0.363, 0.05, "max")).status, S.MATCHED)
        # trung bình top-3 = (1 + 0 + 0) / 3 = 0.33 < 0.363
        self.assertEqual(match_embedding(q, g, MatcherConfig(0.363, 0.05, "topk_mean", 3)).status, S.UNKNOWN)

    def test_empty_gallery_returns_unknown_with_reason(self):
        r = match_embedding(basis(0), Gallery.empty(128), CFG)
        self.assertEqual(r.status, S.UNKNOWN)
        self.assertEqual(r.reason, "EMPTY_INDEX")
        self.assertIsNone(r.employee_code)

    def test_single_employee_gallery_has_no_margin_check(self):
        g = Gallery.from_samples([("NV001", basis(0))], 128)
        self.assertEqual(match_embedding(basis(0), g, CFG).status, S.MATCHED)

    def test_wrong_dimension_raises(self):
        with self.assertRaises(DimensionMismatchError):
            match_embedding(np.ones(64, dtype=np.float32), self.gallery(), CFG)
        with self.assertRaises(DimensionMismatchError):
            Gallery.from_samples([("NV001", np.ones(64))], 128)

    def test_invalid_query_raises(self):
        for bad in (np.zeros(128), np.full(128, np.nan)):
            with self.assertRaises(ValueError):
                match_embedding(bad, self.gallery(), CFG)

    def test_gallery_is_immutable_snapshot(self):
        g = self.gallery()
        with self.assertRaises(ValueError):
            g.matrix[0, 0] = 5.0


if __name__ == "__main__":
    unittest.main()
