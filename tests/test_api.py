import os
import unittest

from fastapi.testclient import TestClient

from app.main import create_app


class FakeSettings:
    max_upload_bytes = 1024


class FakeEngine:
    settings = FakeSettings()

    def recognize(self, data, content_type):
        return {"status": "UNKNOWN", "attendanceEligible": False}

    def enroll(self, employee_code, data, content_type, display_name=None):
        return {"status": "ENROLLED", "employeeCode": employee_code}

    def remove_employee(self, code):
        return code == "NV001"

    def stats(self):
        return {"employeeCount": 1, "sampleCount": 2}


class ApiTests(unittest.TestCase):
    def setUp(self):
        os.environ["FACE_API_KEY"] = "test-secret"
        self.client_context = TestClient(create_app(FakeEngine()))
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        os.environ.pop("FACE_API_KEY", None)

    def test_health_is_public(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_gallery_requires_api_key(self):
        self.assertEqual(self.client.get("/api/v1/gallery").status_code, 401)
        response = self.client.get("/api/v1/gallery", headers={"X-API-Key": "test-secret"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["employeeCount"], 1)

    def test_recognize_upload(self):
        response = self.client.post(
            "/api/v1/recognize",
            headers={"X-API-Key": "test-secret"},
            files={"file": ("sample.jpg", b"jpeg-data", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "UNKNOWN")

    def test_enroll_form(self):
        response = self.client.post(
            "/api/v1/enroll",
            headers={"X-API-Key": "test-secret"},
            data={"employeeCode": "NV001", "displayName": "Test"},
            files={"file": ("sample.jpg", b"jpeg-data", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ENROLLED")

    def test_delete_unknown_employee(self):
        response = self.client.delete(
            "/api/v1/employees/NV999", headers={"X-API-Key": "test-secret"}
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
