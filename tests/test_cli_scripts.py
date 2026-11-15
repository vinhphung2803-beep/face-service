"""Test các script chạy thử (scripts/*.py) với MODEL THẬT (không mock).

Nhóm 1 - chạy được ngay: dùng ảnh nhiễu (chắc chắn không có mặt) để kiểm tra NO_FACE với detector thật,
         logic kho JSON, quy tắc mã nhân viên. Các test này KHÔNG chứng minh khả năng nhận diện.
Nhóm 2 - cần ảnh khuôn mặt thật do bạn cung cấp: đặt biến môi trường FACE_TEST_DIR trỏ tới thư mục:
            <FACE_TEST_DIR>/<MA_NV>/enroll*.jpg   ảnh đăng ký của từng nhân viên
            <FACE_TEST_DIR>/<MA_NV>/probe*.jpg    ảnh khác của đúng người đó, kỳ vọng nhận ra đúng mã
            <FACE_TEST_DIR>/_unknown/*.jpg        người CHƯA đăng ký, kỳ vọng UNKNOWN
            <FACE_TEST_DIR>/_multi/*.jpg          ảnh nhiều người, kỳ vọng MULTIPLE_FACES
         Kết quả chỉ có ý nghĩa thống kê khi bộ ảnh đủ lớn và giống điều kiện thực tế (camera, ánh sáng, góc).
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.core.common import InputValidationError
from app.core.matcher import MatcherConfig
from scripts import _face_cli as cli
from scripts import enroll_image, recognize_image, recognize_registered
from scripts.recognize_webcam import LiveRecognizer
from tests.support import SETTINGS, real_models_available, textured_image

NEEDS_MODELS = unittest.skipUnless(real_models_available(), "Chưa tải model: chạy python scripts/download_models.py")


def run(module, *argv) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = module.main([str(a) for a in argv])
    return rc, out.getvalue(), err.getvalue()


class HelperTests(unittest.TestCase):
    def test_ascii_text_strips_vietnamese_marks(self):
        self.assertEqual(cli.ascii_text("Nguyễn Đức Thắng"), "Nguyen Duc Thang")

    def test_reserved_codes_rejected(self):
        for code in ("UNKNOWN", "unknown", "NO_FACE", "MATCHED", "MULTIPLE_FACES"):
            with self.assertRaises(InputValidationError) as ctx:
                cli.validate_cli_employee_code(code)
            self.assertEqual(ctx.exception.code, "EMPLOYEE_CODE_RESERVED")
        with self.assertRaises(InputValidationError):
            cli.validate_cli_employee_code("a b")
        self.assertEqual(cli.validate_cli_employee_code("NV001"), "NV001")


@NEEDS_MODELS
class NoFaceAndStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.noise = cls.dir / "noise.png"
        cv2.imwrite(str(cls.noise), textured_image(480, 640, seed=3))
        cls.detector, cls.embedder = cli.build_models(SETTINGS)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_noise_is_no_face_with_real_detector(self):
        res = cli.analyze_single(textured_image(480, 640, seed=5), self.detector, self.embedder, SETTINGS)
        self.assertEqual(res.outcome.status.value, "NO_FACE")
        self.assertIsNone(res.embedding)

    def test_recognize_image_returns_1_on_no_face(self):
        rc, out, _ = run(recognize_image, self.noise, "--no-save")
        self.assertEqual(rc, 1)
        self.assertIn("NO_FACE", out)

    def test_enroll_rejects_no_face_and_does_not_create_store(self):
        store = self.dir / "s1.json"
        rc, _, err = run(enroll_image, "NV001", self.noise, "--store", store)
        self.assertEqual(rc, 1)
        self.assertIn("NO_FACE", err)
        self.assertFalse(store.exists())

    def test_enroll_rejects_reserved_code(self):
        rc, _, err = run(enroll_image, "UNKNOWN", self.noise, "--store", self.dir / "s2.json")
        self.assertEqual(rc, 2)
        self.assertIn("EMPLOYEE_CODE_RESERVED", err)

    def test_recognize_registered_missing_store(self):
        rc, _, err = run(recognize_registered, self.noise, "--store", self.dir / "none.json", "--no-save")
        self.assertEqual(rc, 2)
        self.assertIn("STORE_NOT_FOUND", err)

    def test_store_roundtrip_and_validation(self):
        store = cli.new_store(SETTINGS, self.embedder)
        rng = np.random.default_rng(0)
        for code in ("A1", "B2"):
            v = rng.standard_normal(128).astype(np.float32)
            v /= np.linalg.norm(v)
            store["employees"][code] = {"displayName": f"Tên {code}", "samples": [{"embedding": [float(x) for x in v]}]}
        path = self.dir / "store.json"
        cli.save_store(path, store)
        loaded = cli.load_store(path, SETTINGS, self.embedder)
        gallery, names = cli.store_to_gallery(loaded, self.embedder.dimension)
        self.assertEqual((gallery.employee_count, gallery.size), (2, 2))
        self.assertEqual(names["A1"], "Tên A1")
        self.assertEqual(list(self.dir.glob("*.tmp")), [])  # không để lại file tạm

        from dataclasses import replace
        with self.assertRaises(InputValidationError) as ctx:
            cli.load_store(path, replace(SETTINGS, model_version="khac"), self.embedder)
        self.assertEqual(ctx.exception.code, "MODEL_VERSION_MISMATCH")

        loaded["embedderSha256"] = "0" * 64
        cli.save_store(path, loaded)
        with self.assertRaises(InputValidationError) as ctx:
            cli.load_store(path, SETTINGS, self.embedder)
        self.assertEqual(ctx.exception.code, "MODEL_FILE_MISMATCH")

    def test_live_recognizer_reports_no_face_on_noise_frames(self):
        gallery, _ = cli.store_to_gallery(cli.new_store(SETTINGS, self.embedder), 128)
        live = LiveRecognizer(self.detector, self.embedder, gallery, SETTINGS, MatcherConfig.from_settings(SETTINGS))
        self.assertEqual(live.process(textured_image(480, 640, seed=9)).state, "NO_FACE")


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if folder.is_dir() else []


@NEEDS_MODELS
@unittest.skipUnless(os.environ.get("FACE_TEST_DIR"), "Đặt FACE_TEST_DIR để chạy test với ảnh khuôn mặt thật của bạn")
class RealPhotoSetTests(unittest.TestCase):
    """Chạy qua đúng các script CLI. Thất bại ở đây là thông tin về bộ ảnh/ngưỡng, hãy đọc kỹ từng ca."""

    def test_real_photo_set(self):
        root = Path(os.environ["FACE_TEST_DIR"])
        people = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))
        self.assertGreaterEqual(len(people), 2, "cần ít nhất 2 nhân viên để thử 1:N")
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "s.json"
            for person in people:
                enrolls = [p for p in _images(person) if p.stem.startswith("enroll")]
                self.assertTrue(enrolls, f"{person.name}: thiếu ảnh enroll*")
                for img in enrolls:
                    rc, _, err = run(enroll_image, person.name, img, "--store", store)
                    self.assertEqual(rc, 0, f"đăng ký {person.name}/{img.name} thất bại: {err}")
            cases = []
            for person in people:
                cases += [(img, person.name) for img in _images(person) if img.stem.startswith("probe")]
            cases += [(img, "UNKNOWN") for img in _images(root / "_unknown")]
            cases += [(img, "MULTIPLE_FACES") for img in _images(root / "_multi")]
            self.assertTrue(cases, "không có ảnh thử (probe*/_unknown/_multi)")
            wrong = []
            for img, expected in cases:
                rc, out, _ = run(recognize_registered, img, "--store", store, "--no-save")
                got = [l for l in out.splitlines() if l.startswith("RESULT: ")][-1][8:]
                if got != expected:
                    wrong.append(f"{img.parent.name}/{img.name}: kỳ vọng {expected}, nhận {got}")
            print(f"\n[FACE_TEST_DIR] {len(cases) - len(wrong)}/{len(cases)} ca đúng "
                  f"({len(people)} nhân viên). Chỉ là số liệu của bộ ảnh này, không phải độ chính xác tổng quát.")
            self.assertEqual(wrong, [], "\n" + "\n".join(wrong))


if __name__ == "__main__":
    unittest.main()
