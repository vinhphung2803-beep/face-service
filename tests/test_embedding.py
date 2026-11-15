import unittest

import numpy as np

from app.core.common import InputValidationError, ModelLoadError
from app.core.embedding import SFaceEmbedder, l2_normalize
from app.utils.validators import validate_embedding_payload
from tests.support import SETTINGS, real_models_available

VER = "yunet-2023mar+sface-2021dec"


class NormalizeTests(unittest.TestCase):
    def test_unit_norm_float32(self):
        v = l2_normalize([3.0, 4.0])
        self.assertEqual(v.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=6)

    def test_zero_and_nan_rejected(self):
        for bad in ([0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0]):
            with self.assertRaises(ValueError):
                l2_normalize(bad)


class PayloadValidationTests(unittest.TestCase):
    def call(self, values, version=VER, dim=128):
        return validate_embedding_payload(values, expected_dimension=dim, expected_model_version=VER, model_version=version)

    def test_valid_payload_is_normalized(self):
        v = self.call([2.0] * 128)
        self.assertEqual(v.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_wrong_dimension(self):
        with self.assertRaises(InputValidationError) as ctx:
            self.call([1.0] * 127)
        self.assertEqual(ctx.exception.code, "EMBEDDING_DIMENSION_MISMATCH")

    def test_model_version_mismatch(self):
        with self.assertRaises(InputValidationError) as ctx:
            self.call([1.0] * 128, version="other-model")
        self.assertEqual(ctx.exception.code, "MODEL_VERSION_MISMATCH")

    def test_nan_zero_and_non_numeric(self):
        for bad in ([float("nan")] * 128, [0.0] * 128, ["a"] * 128):
            with self.assertRaises(InputValidationError) as ctx:
                self.call(bad)
            self.assertEqual(ctx.exception.code, "EMBEDDING_INVALID")


class SFaceTests(unittest.TestCase):
    def test_missing_model_raises_clear_error(self):
        with self.assertRaises(ModelLoadError):
            SFaceEmbedder("khong_ton_tai.onnx", VER)

    @unittest.skipUnless(real_models_available(), "Chưa tải model: chạy python scripts/download_models.py")
    def test_real_model_dimension_and_determinism(self):
        from tests.support import make_face, textured_image

        emb = SFaceEmbedder(SETTINGS.embedder_model_path, VER)
        self.assertEqual(emb.dimension, 128)
        img, face = textured_image(), make_face()
        a, b = emb.embed(img, face), emb.embed(img, face)
        self.assertEqual(a.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=5)
        self.assertTrue(np.allclose(a, b, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
