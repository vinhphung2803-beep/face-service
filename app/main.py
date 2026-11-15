"""FastAPI entry point for AttendQR Face Service."""
from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.core.common import InputValidationError, ModelLoadError
from app.services.face_engine import FaceEngine


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    configured = os.getenv("FACE_API_KEY", "").strip()
    if not configured:
        raise HTTPException(status_code=503, detail={
            "code": "API_KEY_NOT_CONFIGURED",
            "message": "Quản trị viên chưa cấu hình FACE_API_KEY",
        })
    if not x_api_key or not hmac.compare_digest(x_api_key, configured):
        raise HTTPException(status_code=401, detail={
            "code": "UNAUTHORIZED", "message": "API key không hợp lệ"
        })


async def read_limited_upload(file: UploadFile, max_bytes: int) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = await file.read(min(1024 * 1024, max_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise InputValidationError("FILE_TOO_LARGE", f"File vượt quá giới hạn {max_bytes // 1024} KB")
    return b"".join(chunks)


def create_app(engine: FaceEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is not None:
            app.state.face_engine = engine
        else:
            app.state.face_engine = FaceEngine(Settings.from_env())
        yield

    app = FastAPI(
        title="AttendQR Face Service",
        version="1.0.0",
        description=(
            "Dịch vụ phát hiện và nhận diện khuôn mặt 1:N. "
            "Spring Boot vẫn là hệ thống quyết định và ghi nhận chấm công."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(InputValidationError)
    async def input_validation_handler(request, exc: InputValidationError):
        return JSONResponse(status_code=400, content={
            "error": {"code": exc.code, "message": exc.message}
        })

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "attendqr-face-service"}

    @app.post("/api/v1/recognize", dependencies=[Depends(require_api_key)])
    async def recognize(file: UploadFile = File(...)):
        face_engine = app.state.face_engine
        data = await read_limited_upload(file, face_engine.settings.max_upload_bytes)
        return await run_in_threadpool(face_engine.recognize, data, file.content_type)

    @app.post("/api/v1/enroll", dependencies=[Depends(require_api_key)])
    async def enroll(
        employeeCode: str = Form(...),
        displayName: str | None = Form(default=None),
        file: UploadFile = File(...),
    ):
        face_engine = app.state.face_engine
        data = await read_limited_upload(file, face_engine.settings.max_upload_bytes)
        return await run_in_threadpool(face_engine.enroll, employeeCode, data, file.content_type, displayName)

    @app.delete("/api/v1/employees/{employee_code}", dependencies=[Depends(require_api_key)])
    async def remove_employee(employee_code: str):
        if not await run_in_threadpool(app.state.face_engine.remove_employee, employee_code):
            raise HTTPException(status_code=404, detail={
                "code": "EMPLOYEE_NOT_FOUND", "message": "Không tìm thấy nhân viên trong gallery"
            })
        return {"status": "DELETED", "employeeCode": employee_code}

    @app.get("/api/v1/gallery", dependencies=[Depends(require_api_key)])
    async def gallery_stats():
        return await run_in_threadpool(app.state.face_engine.stats)

    return app


app = create_app()
