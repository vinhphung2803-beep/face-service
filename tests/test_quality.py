import unittest

import cv2

from app.core.quality import assess_quality
from tests.support import SETTINGS, make_face, textured_image


class QualityTests(unittest.TestCase):
    """Dùng ảnh nhiễu tổng hợp + landmarks giả: kiểm tra logic đo/ngưỡng, KHÔNG phải chất lượng mặt thật."""

    def setUp(self):
        self.img = textured_image()

    def reason(self, img=None, **face_kw):
        return assess_quality(self.img if img is None else img, make_face(**face_kw), SETTINGS)

    def test_good_sample_passes(self):
        r = self.reason()
        self.assertTrue(r.passed, r)
        self.assertIsNone(r.failure_reason)
        self.assertGreater(r.score, 0.0)

    def test_face_too_small(self):
        self.assertEqual(self.reason(size=50).failure_reason, "FACE_TOO_SMALL")

    def test_face_cut_by_frame(self):
        self.assertEqual(self.reason(x=-70).failure_reason, "FACE_PARTIALLY_OUTSIDE")

    def test_dark_image(self):
        dark = (self.img * 0.1).astype("uint8")
        self.assertEqual(self.reason(img=dark).failure_reason, "TOO_DARK")

    def test_overexposed_image(self):
        bright = cv2.convertScaleAbs(self.img, alpha=0.2, beta=215)
        self.assertEqual(self.reason(img=bright).failure_reason, "TOO_BRIGHT")

    def test_blurry_image(self):
        blurry = cv2.GaussianBlur(self.img, (51, 51), 15)
        r = self.reason(img=blurry)
        self.assertEqual(r.failure_reason, "TOO_BLURRY")
        self.assertFalse(r.passed)

    def test_head_roll(self):
        self.assertEqual(self.reason(roll_deg=35).failure_reason, "POSE_ROLL")

    def test_head_yaw(self):
        self.assertEqual(self.reason(nose_dx=0.3).failure_reason, "POSE_YAW")

    def test_upside_down_landmarks(self):
        self.assertEqual(self.reason(flip=True).failure_reason, "LANDMARKS_INCONSISTENT")

    def test_score_is_not_a_probability_but_bounded_index(self):
        r = self.reason()
        self.assertGreaterEqual(r.score, 0.0)
        self.assertLessEqual(r.score, 1.0)


if __name__ == "__main__":
    unittest.main()
