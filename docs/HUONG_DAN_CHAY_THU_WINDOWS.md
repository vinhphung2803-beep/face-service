# Hướng dẫn chạy thử nhận diện khuôn mặt trên Windows (PowerShell)

Dự án hiện có cả các script thử nghiệm trong `scripts\\` và API FastAPI (`app\\main.py`). Script dùng để kiểm tra
model/camera cục bộ; API dùng để tích hợp với AttendQR Spring Boot.

> **Dữ liệu sinh trắc học.** `data\enrollments.json` chứa embedding khuôn mặt. Chỉ dùng để thử, không commit
> (thư mục `data\` đã nằm trong `.gitignore`), không gửi cho người khác. Chỉ đăng ký ảnh của người đã đồng ý.

## 0. Chuẩn bị (một lần)

```powershell
cd face-service
py -3.11 -m venv .venv
# Nếu bị chặn chạy script: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_models.py

# Hiển thị tiếng Việt đúng trong PowerShell (mỗi cửa sổ mới):
$OutputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

SHA256 của hai model (script tải sẽ in lại, hãy đối chiếu):

| File | SHA256 |
|---|---|
| `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

Nếu tải tự động lỗi, tải thủ công từ <https://github.com/opencv/opencv_zoo> (thư mục `models\face_detection_yunet`
và `models\face_recognition_sface`) rồi chép vào `models\`.

**Chỉ khi dùng webcam hoặc `--show`** (cần cửa sổ, `opencv-python-headless` không có):

```powershell
pip uninstall -y opencv-python-headless
pip install -r requirements-desktop.txt
```

Ảnh đầu vào đi qua cùng các kiểm tra của service: chỉ JPEG/PNG, tối đa 5 MB (`MAX_UPLOAD_BYTES`), cạnh ngắn
tối thiểu 100 px. Ảnh điện thoại lớn hơn 5 MB sẽ bị từ chối; thu nhỏ ảnh hoặc tăng `MAX_UPLOAD_BYTES` trong `.env`.
Đường dẫn có dấu tiếng Việt hoặc khoảng trắng dùng được, nhớ đặt trong dấu nháy.

## 1. Phát hiện mặt + embedding 128 chiều: `recognize_image.py`

Không nhận diện danh tính, chỉ để xem model chạy: vẽ khung, in embedding.

```powershell
python scripts\recognize_image.py "C:\anh\nguoi1.jpg"
python scripts\recognize_image.py "C:\anh\nguoi1.jpg" --out C:\anh\ket_qua.jpg   # chọn nơi lưu ảnh
python scripts\recognize_image.py "C:\anh\nguoi1.jpg" --print-full                # in đủ 128 giá trị
python scripts\recognize_image.py "C:\anh\nguoi1.jpg" --json > embedding.json     # JSON đầy đủ
python scripts\recognize_image.py "C:\anh\nguoi1.jpg" --show                      # mở cửa sổ (cần opencv-python)
```

Mặc định ảnh có khung lưu ở `data\annotated\<tên>_detected.jpg`. Mã thoát: `0` có mặt, `1` không có mặt, `2` lỗi.
Khung xanh = đạt chất lượng, cam = chưa đạt (kèm lý do, ví dụ `TOO_BLURRY`).

## 2. Đăng ký thử nghiệm: `enroll_image.py`

```powershell
python scripts\enroll_image.py NV001 "C:\anh\nv001_1.jpg" --name "Nguyễn Văn A"
python scripts\enroll_image.py NV001 "C:\anh\nv001_2.jpg"            # thêm mẫu thứ 2 (góc/ánh sáng khác)
python scripts\enroll_image.py NV001 "C:\anh\nv001_moi.jpg" --replace  # thay toàn bộ mẫu cũ của NV001
python scripts\enroll_image.py NV002 "C:\anh\nv002.jpg" --store C:\tam\kho_khac.json
```

Kết quả lưu ở `data\enrollments.json` (ghi nguyên tử, mỗi mẫu kèm tên ảnh nguồn, thời điểm, điểm detector).
Script **từ chối** (mã thoát `1`) khi: không có mặt, có nhiều mặt, chất lượng chưa đạt (kèm gợi ý), hoặc khuôn mặt
đã giống một nhân viên khác (dùng `--allow-duplicate-face` nếu thật sự muốn). Nó cảnh báo nếu mẫu mới không giống
mẫu cũ nào của cùng mã (có thể nhầm người). Mã `UNKNOWN`, `NO_FACE`... bị cấm vì trùng nhãn kết quả.
Nên đăng ký 2–3 ảnh khác góc/ánh sáng cho mỗi người.

## 3. Nhận diện 1:N trên ảnh: `recognize_registered.py`

```powershell
python scripts\recognize_registered.py "C:\anh\can_nhan_dien.jpg"
python scripts\recognize_registered.py "C:\anh\can_nhan_dien.jpg" --show-candidates   # xem điểm các ứng viên
python scripts\recognize_registered.py "C:\anh\can_nhan_dien.jpg" --threshold 0.40 --margin 0.05
python scripts\recognize_registered.py "C:\anh\can_nhan_dien.jpg" --json
```

Dòng cuối luôn là `RESULT: <mã nhân viên>`, `RESULT: UNKNOWN`, hoặc một trạng thái khác:
`AMBIGUOUS` (hai người quá sát điểm, không kết luận), `NO_FACE`, `MULTIPLE_FACES` (không tự chọn ai),
`LOW_QUALITY`. Mã thoát: `0` nhận ra, `1` kết quả khác, `2` lỗi (kho sai phiên bản model, ảnh hỏng...).
Điểm là **cosine similarity**, không phải xác suất đúng. Ngưỡng mặc định `0.363` chỉ là điểm khởi đầu.

## 4. Webcam: `recognize_webcam.py`

```powershell
python scripts\recognize_webcam.py                    # camera 0
python scripts\recognize_webcam.py --camera 1         # camera khác
python scripts\recognize_webcam.py --width 1280 --height 720
python scripts\recognize_webcam.py --confirm-frames 5 --threshold 0.40
```

Phím `q` hoặc `ESC` để thoát. Hành vi:

- **Nhiều khuôn mặt trong khung → không nhận diện ai** (khung đỏ, cảnh báo cam), không tính embedding.
- Chất lượng chưa đạt → hiện lý do, không nhận diện.
- Chỉ hiện mã/tên khi cùng một kết quả lặp lại `--confirm-frames` khung liên tiếp (mặc định 3).
- Xanh = đã đăng ký (mã - tên, điểm cosine); đỏ = `UNKNOWN`; cam = `AMBIGUOUS`/chất lượng; vàng = đang xác nhận.
- Terminal chỉ in dòng mới khi kết quả ổn định thay đổi.
- Tên có dấu hiển thị trên cửa sổ sẽ bị bỏ dấu (OpenCV không vẽ được chữ có dấu); terminal vẫn in đủ dấu.

Chạy không cửa sổ / thử bằng video:

```powershell
python scripts\recognize_webcam.py --source "C:\anh\video_thu.mp4" --no-display --output-video C:\anh\ket_qua.avi
```

> **Chưa có chống giả mạo (liveness).** Ảnh hoặc video của người đã đăng ký đưa trước camera vẫn có thể được nhận.
> Nhận đúng khuôn mặt không có nghĩa là người thật đang đứng trước camera.

Nếu không mở được camera: đóng ứng dụng khác đang dùng camera (Zoom, Teams...), bật *Settings → Privacy & security
→ Camera → Let desktop apps access your camera*, rồi thử `--camera 1`.

## 5. Kiểm thử tự động

```powershell
python -m unittest discover -s tests -t . -v
```

Chạy với model thật nếu đã tải. Nhóm test ảnh nhiễu/kho JSON chạy ngay; chúng **không** chứng minh khả năng nhận diện.
Để thử với khuôn mặt thật của bạn, tạo thư mục:

```
C:\bo_anh_thu\
  NV001\enroll1.jpg  NV001\probe1.jpg  NV001\probe2.jpg   (enroll* = ảnh đăng ký, probe* = ảnh KHÁC của cùng người)
  NV002\enroll1.jpg  NV002\probe1.jpg
  _unknown\nguoi_la1.jpg                                    (người chưa đăng ký, kỳ vọng UNKNOWN)
  _multi\nhom1.jpg                                          (nhiều người, kỳ vọng MULTIPLE_FACES)
```

```powershell
$env:FACE_TEST_DIR = "C:\bo_anh_thu"
python -m unittest tests.test_cli_scripts.RealPhotoSetTests -v
```

## 6. Về độ chính xác

Chưa thể nói hệ thống "chính xác X%". Muốn có con số đáng tin cần bộ ảnh đủ lớn (nhiều người, nhiều ngày/ánh sáng/góc,
camera thật tại điểm chấm công, cả người lạ) rồi hiệu chỉnh ngưỡng bằng dữ liệu đó (Giai đoạn 4, `benchmark.py`).
Mọi ngưỡng chất lượng và matching hiện chỉ là điểm khởi đầu.


## 7. Chạy Face Service API

Cài thêm dependency API bằng:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Trong `.env`, đặt `FACE_API_KEY` thành secret ngẫu nhiên dài. Không chia sẻ secret, không commit `.env`.

Khởi động:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Nhận diện: `POST /api/v1/recognize` (multipart field `file`)
- Đăng ký: `POST /api/v1/enroll` (multipart fields `employeeCode`, `file`, tùy chọn `displayName`)
- Thống kê: `GET /api/v1/gallery`
- Xóa mẫu nhân viên: `DELETE /api/v1/employees/{employee_code}`

Mọi endpoint `/api/v1/*` cần header `X-API-Key`. Chi tiết request/response và lệnh PowerShell xem `README.md`.

**Liveness hiện vẫn chưa được tích hợp.** Khi backend là `none`, nhận diện có thể trả `MATCHED` nhưng
`attendanceEligible` vẫn là `false`; không cho Spring Boot tự động ghi chấm công dựa trên kết quả này.
