import unittest

import cv2
import numpy as np

from app.core.common import InputValidationError
from app.utils.validators import validate_employee_code, validate_upload
from tests.support import SETTINGS, textured_image


def enc(ext, img):
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()


class UploadValidationTests(unittest.TestCase):
    def code(self, data, ctype):
        with self.assertRaises(InputValidationError) as ctx:
            validate_upload(data, ctype, SETTINGS)
        return ctx.exception.code

    def test_valid_jpeg_and_png(self):
        img = textured_image(300, 400)
        self.assertEqual(validate_upload(enc(".jpg", img), "image/jpeg", SETTINGS), "image/jpeg")
        self.assertEqual(validate_upload(enc(".png", img), "image/png", SETTINGS), "image/png")

    def test_empty_file(self):
        self.assertEqual(self.code(b"", "image/jpeg"), "EMPTY_FILE")

    def test_wrong_declared_mime(self):
        self.assertEqual(self.code(enc(".jpg", textured_image(300, 400)), "text/plain"), "UNSUPPORTED_MIME")

    def test_wrong_format_with_image_mime_is_rejected_by_magic_bytes(self):
        self.assertEqual(self.code(b"GIF89a" + b"\x00" * 200, "image/jpeg"), "UNSUPPORTED_MIME")
        self.assertEqual(self.code(b"hello world" * 50, "image/png"), "UNSUPPORTED_MIME")

    def test_corrupt_jpeg_header(self):
        self.assertEqual(self.code(b"\xff\xd8\xff" + bytes(range(100)), "image/jpeg"), "INVALID_IMAGE")

    def test_file_too_large(self):
        big = enc(".png", textured_image(300, 400)) + b"\x00" * SETTINGS.max_upload_bytes
        self.assertEqual(self.code(big, "image/png"), "FILE_TOO_LARGE")

    def test_image_too_small(self):
        self.assertEqual(self.code(enc(".png", textured_image(50, 50)), "image/png"), "IMAGE_TOO_SMALL")

    def test_pixel_bomb_rejected_from_header_before_decode(self):
        png = bytearray(enc(".png", textured_image(300, 400)))
        png[16:20] = (20000).to_bytes(4, "big")  # giả mạo width trong IHDR
        png[20:24] = (20000).to_bytes(4, "big")
        self.assertEqual(self.code(bytes(png), "image/png"), "IMAGE_TOO_LARGE")


class EmployeeCodeTests(unittest.TestCase):
    def test_valid(self):
        for c in ("NV001", "nv-002", "A.b_3"):
            self.assertEqual(validate_employee_code(c), c)

    def test_invalid(self):
        for c in ("", " NV1", "NV 1", "../etc", "x" * 33, None, 123, "NV;DROP"):
            with self.assertRaises(InputValidationError):
                validate_employee_code(c)


if __name__ == "__main__":
    unittest.main()
