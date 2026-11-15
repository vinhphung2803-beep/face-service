import unittest

import cv2
import numpy as np

from app.core.common import InputValidationError
from app.core.detector import DetectionStatus, YuNetDetector, classify_detection
from app.core.common import ModelLoadError
from app.utils.image_utils import decode_image, resize_max_side
from tests.support import SETTINGS, make_face, real_models_available


class ClassifyDetectionTests(unittest.TestCase):
    def test_no_face(self):
        out = classify_detection([], 40)
        self.assertEqual(out.status, DetectionStatus.NO_FACE)
        self.assertIsNone(out.face)

    def test_single_valid_face(self):
        f = make_face(size=150)
        out = classify_detection([f], 40)
        self.assertEqual(out.status, DetectionStatus.SINGLE_FACE)
        self.assertIs(out.face, f)

    def test_two_large_faces_never_pick_one(self):
        out = classify_detection([make_face(size=150), make_face(x=400, size=120)], 40)
        self.assertEqual(out.status, DetectionStatus.MULTIPLE_FACES)
        self.assertIsNone(out.face)

    def test_tiny_background_face_is_ignored_when_counting(self):
        big, tiny = make_face(size=150), make_face(x=500, size=20)
        out = classify_detection([big, tiny], 40)
        self.assertEqual(out.status, DetectionStatus.SINGLE_FACE)
        self.assertIs(out.face, big)
        self.assertEqual(out.total_detected, 2)

    def test_single_tiny_face_goes_on_to_quality_check(self):
        out = classify_detection([make_face(size=20)], 40)
        self.assertEqual(out.status, DetectionStatus.SINGLE_FACE)

    def test_several_tiny_faces_are_multiple(self):
        out = classify_detection([make_face(size=20), make_face(x=300, size=25)], 40)
        self.assertEqual(out.status, DetectionStatus.MULTIPLE_FACES)


class ImageHandlingTests(unittest.TestCase):
    def test_decode_garbage_raises_validation_error(self):
        with self.assertRaises(InputValidationError) as ctx:
            decode_image(b"not an image at all")
        self.assertEqual(ctx.exception.code, "INVALID_IMAGE")

    def test_resize_keeps_aspect_and_never_upscales(self):
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        small, scale = resize_max_side(img, 960)
        self.assertEqual(small.shape[:2], (480, 960))
        self.assertAlmostEqual(scale, 0.48, places=3)
        same, scale2 = resize_max_side(small, 960)
        self.assertIs(same, small)
        self.assertEqual(scale2, 1.0)


class YuNetDetectorTests(unittest.TestCase):
    def test_missing_model_raises_clear_error(self):
        with self.assertRaises(ModelLoadError):
            YuNetDetector("khong_ton_tai.onnx", 0.7, 0.3, 50)

    @unittest.skipUnless(real_models_available(), "Chưa tải model: chạy python scripts/download_models.py")
    def test_blank_image_returns_no_face_without_crashing(self):
        det = YuNetDetector(SETTINGS.detector_model_path, 0.7, 0.3, 50)
        for img in (np.zeros((480, 640, 3), np.uint8), np.full((480, 640, 3), 255, np.uint8)):
            self.assertEqual(det.detect(img), [])

    @unittest.skipUnless(real_models_available(), "Chưa tải model: chạy python scripts/download_models.py")
    def test_detector_handles_different_input_sizes(self):
        det = YuNetDetector(SETTINGS.detector_model_path, 0.7, 0.3, 50)
        for shape in ((240, 320, 3), (720, 1280, 3)):
            self.assertIsInstance(det.detect(np.zeros(shape, np.uint8)), list)


if __name__ == "__main__":
    unittest.main()
