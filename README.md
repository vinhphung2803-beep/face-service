# AttendQR Face Service — API phase

Dịch vụ Python độc lập dùng **YuNet + SFace** để phát hiện, kiểm tra chất lượng, trích xuất embedding 128 chiều và nhận diện khuôn mặt 1:N. Spring Boot vẫn là nơi xác thực nhân viên, kiểm tra ca làm và quyết định ghi nhận chấm công.

## 1. Kiến trúc

```text
Camera Kiosk (HTML/CSS/JS)
        |
        | ảnh chụp qua HTTPS / phiên kiosk đã xác thực
        v
Spring Boot AttendQR API
  - Xác thực kiosk / phiên làm việc
  - Gọi Face Service bằng API key server-side
        |
        | multipart/form-data: file
        v
Face Service (FastAPI, mạng nội bộ)
  - Validate JPEG/PNG
  - YuNet: phát hiện khuôn mặt
  - Quality: kiểm tra ảnh
  - SFace: embedding 128D
  - Matcher: so khớp 1:N
        |
        | employeeCode + status + similarity + livenessStatus
        v
Spring Boot AttendQR API
  - Kiểm tra nhân viên còn hoạt động
  - Kiểm tra lịch/ca làm và trạng thái chấm công
  - Chống ghi nhận trùng
  - Quyết định CHECK_IN / CHECK_OUT
  - Lưu PostgreSQL
```

Face Service **không tự ghi chấm công** và không kết nối trực tiếp PostgreSQL. Không dùng việc nhận diện thành công để tự động đổi CHECK_IN/CHECK_OUT theo kiểu luân phiên.

## 2. Cài đặt trên Windows

Khuyến nghị Python 3.11. Mở PowerShell tại thư mục `face-service`:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu thư mục `models` chưa có hai file ONNX, chạy:

```powershell
python scripts\download_models.py
```

Tạo file cấu hình:

```powershell
Copy-Item .env.example .env
notepad .env
```

Đổi `FACE_API_KEY` thành một chuỗi bí mật dài, ngẫu nhiên. Không chia sẻ API key và không commit file `.env`.

Chạy test:

```powershell
python -m unittest discover -s tests -t . -v
```

Khởi động API:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Mở tài liệu API tương tác tại `http://127.0.0.1:8000/docs`. Health check: `http://127.0.0.1:8000/health`.

## 3. Cấu hình `.env`

Các giá trị quan trọng:

| Biến | Ý nghĩa |
|---|---|
| `FACE_API_KEY` | Khóa xác thực bắt buộc cho mọi endpoint `/api/v1/*` |
| `FACE_MODEL_DIR` | Thư mục chứa model ONNX |
| `FACE_MODEL_VERSION` | Nhãn phiên bản embedding |
| `MATCH_THRESHOLD` | Ngưỡng cosine similarity ban đầu |
| `MATCH_MARGIN` | Chênh lệch tối thiểu giữa ứng viên đầu và thứ hai |
| `LIVENESS_BACKEND` | Hiện chỉ hỗ trợ `none` |

Ngưỡng matching mặc định chưa được hiệu chỉnh cho camera thực tế. Similarity không phải xác suất nhận diện đúng.

## 4. API

Tất cả endpoint `/api/v1/*` yêu cầu header. API key chỉ được lưu ở backend Spring Boot / biến môi trường Face Service; **không nhúng API key vào JavaScript hoặc trình duyệt kiosk**:

```http
X-API-Key: <FACE_API_KEY>
```

### Health check

`GET /health` — không yêu cầu API key.

### Đăng ký khuôn mặt

`POST /api/v1/enroll`

`multipart/form-data`:

| Field | Bắt buộc | Mô tả |
|---|---:|---|
| `employeeCode` | Có | Mã nhân viên |
| `file` | Có | Ảnh JPEG hoặc PNG, chỉ một khuôn mặt |
| `displayName` | Không | Tên hiển thị |

Ví dụ PowerShell:

```powershell
$headers = @{ "X-API-Key" = "THAY_BANG_API_KEY_CUA_BAN" }
$form = @{
    employeeCode = "NV001"
    displayName = "Nguyen Van A"
    file = Get-Item ".\portrait.jpg"
}
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/enroll" -Headers $headers -Form $form
```

Mỗi nhân viên được lưu tối đa 10 mẫu embedding. Gửi thêm ảnh với cùng `employeeCode` để bổ sung mẫu.

### Nhận diện

`POST /api/v1/recognize`

`multipart/form-data` với field `file`.

```powershell
$headers = @{ "X-API-Key" = "THAY_BANG_API_KEY_CUA_BAN" }
$form = @{ file = Get-Item ".\camera-frame.jpg" }
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/recognize" -Headers $headers -Form $form
```

Các trạng thái chính: `MATCHED`, `UNKNOWN`, `AMBIGUOUS`, `NO_FACE`, `MULTIPLE_FACES`, `LOW_QUALITY`, `LIVENESS_FAILED`.

Ví dụ response khi tìm thấy nhân viên:

```json
{
  "status": "MATCHED",
  "employeeCode": "NV001",
  "similarity": 0.82,
  "runnerUpSimilarity": 0.41,
  "reason": "OK",
  "modelVersion": "yunet-2023mar+sface-2021dec",
  "livenessStatus": "NOT_CONFIGURED",
  "attendanceEligible": false
}
```

**Lưu ý quan trọng:** code hiện tại chưa tích hợp model liveness. Khi `LIVENESS_BACKEND=none`, API có thể trả `MATCHED` nhưng luôn trả `attendanceEligible: false`. Không được coi nhận diện khuôn mặt là bằng chứng người thật đang đứng trước camera. Chỉ bật ghi nhận chấm công tự động sau khi tích hợp và kiểm thử chống giả mạo thực sự.

### Thống kê gallery

`GET /api/v1/gallery`

Trả số nhân viên, số mẫu, model version, số chiều embedding và backend liveness.

### Xóa dữ liệu khuôn mặt nhân viên

`DELETE /api/v1/employees/{employee_code}`

Xóa toàn bộ embedding của mã nhân viên khỏi gallery.

## 5. Lưu trữ và dữ liệu sinh trắc học

- Ảnh upload chỉ xử lý trong bộ nhớ; API không lưu ảnh gốc.
- Embedding được lưu tại `data/enrollments.json`; đây vẫn là dữ liệu sinh trắc học nhạy cảm.
- Thư mục `data/` đã được loại khỏi Git bằng `.gitignore`.
- Giới hạn quyền truy cập file/thư mục, mã hóa ổ đĩa hoặc volume, sao lưu có kiểm soát và xây dựng quy trình xóa dữ liệu theo yêu cầu.
- Chỉ đăng ký khuôn mặt khi có thông báo và căn cứ/đồng ý phù hợp; cung cấp phương án chấm công thủ công thay thế.
- Không chạy API công khai qua HTTP. Khi triển khai qua mạng, đặt sau HTTPS/reverse proxy, giới hạn IP và quản lý API key an toàn.

## 6. Tích hợp với Spring Boot

Kiosk gửi ảnh đến endpoint Spring Boot; Spring Boot gọi `POST /api/v1/recognize` bằng API key server-side và nhận `employeeCode`, `status`, `similarity`, `livenessStatus`, `attendanceEligible`.

Spring Boot phải tự:
1. Tra cứu `employeeCode` trong cơ sở dữ liệu và kiểm tra nhân viên hợp lệ.
2. Kiểm tra lịch làm việc, trạng thái chấm công và cửa sổ thời gian.
3. Chống gửi lặp bằng cooldown/idempotency và transaction.
4. Chỉ ghi nhận khi chính sách bảo mật cho phép; hiện tại `attendanceEligible` luôn false do liveness chưa cấu hình.
5. Ghi log kết quả cần thiết, không log embedding hoặc ảnh.

Nếu Spring Boot đang chạy trên Render/cloud còn Face Service chạy trên máy cá nhân, `localhost` của Render **không trỏ về máy cá nhân**. Hai dịch vụ cần có kết nối mạng riêng/địa chỉ HTTPS được bảo vệ.

## 7. Giới hạn hiện tại

- Ngưỡng quality/matching là giá trị khởi đầu, cần hiệu chỉnh trên camera, ánh sáng và tập người dùng thực tế.
- Test tổng hợp không chứng minh FAR/FRR trên người thật.
- Liveness chưa được triển khai (`NOT_CONFIGURED`).
- Gallery JSON là giải pháp prototype một instance; chưa phù hợp chạy nhiều replica hoặc môi trường production có yêu cầu HA.
