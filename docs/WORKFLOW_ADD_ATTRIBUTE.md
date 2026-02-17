# Hướng Dẫn: Thêm Thuộc Tính Mới Cho Ảnh (Workflow)

Tài liệu này hướng dẫn các bước cần thiết để thêm một thuộc tính mới (ví dụ: `sharpness`, `color_entropy`, v.v.) vào quy trình xử lý ảnh.

---

## 1. Backend (Python/FastAPI)

### Bước 1.1: Cập nhật Database Model

**File:** `api/models/database.py`
Thêm cột mới vào bảng `ImageMetadata`.

```python
class ImageMetadata(Base):
    # ... các cột cũ
    new_attribute = Column(Float)  # <-- Thêm dòng này
```

> ⚠️ **Lưu ý:** Nếu database đã có dữ liệu, bạn cần chạy lệnh SQL (`ALTER TABLE image_metadata ADD COLUMN new_attribute FLOAT;`) hoặc xóa file `.db` (nếu dùng SQLite dev) để tái tạo bảng.

### Bước 1.2: Cập nhật Pydantic Schema

**File:** `api/models/schemas.py`
Thêm trường mới vào `ImageResponse` để API trả về dữ liệu này cho Frontend.

```python
class ImageResponse(BaseModel):
    # ...
    new_attribute: Optional[float] = None  # <-- Thêm dòng này
```

### Bước 1.3: Cập nhật Logic Trích Xuất (Image Processor)

**File:** `api/services/image_processor.py`
Tính toán giá trị thuộc tính mới trong hàm `extract_features`.

```python
def extract_features(self, image_path: Path) -> Dict:
    # ... (code xử lý ảnh)
    
    # Tính toán thuộc tính mới
    new_value = self._calculate_something(img) 

    return {
        # ...
        'new_attribute': new_value,  # <-- Thêm vào dict trả về
    }
```

### Bước 1.4: Cập nhật Database Service

**File:** `api/services/database_service.py`
Cập nhật 2 hàm `create_image_metadata` và `update_image_metadata` để lưu dữ liệu vào database.

```python
# Trong create_image_metadata:
image_record = ImageMetadata(
    # ...
    new_attribute=features.get('new_attribute'), # <-- Map từ features dict
)

# Trong update_image_metadata:
image.new_attribute = features.get('new_attribute') # <-- Cập nhật field
```

---

## 2. Frontend (React)

### Bước 2.1: Hiển thị trên Giao diện

**File:** `frontend/src/components/import/ImportHistory.jsx`
Thêm phần hiển thị giá trị mới trong thẻ `Card`.

```jsx
<div className="flex flex-col">
    <span className="text-slate-500 text-xs mb-1">New Attribute</span>
    <span className="text-slate-200 font-medium">
        {item.new_attribute ? item.new_attribute.toFixed(2) : 'N/A'}
    </span>
</div>
```

---

## 3. Cập nhật dữ liệu cũ (Recompute)

Sau khi deploy code mới:

1. Vào giao diện **Import History**.
2. Nhấn nút **"Recompute All"** trên thanh tiêu đề.
3. Hệ thống sẽ tính toán lại `new_attribute` cho TẤT CẢ ảnh cũ và lưu vào database. Không cần upload lại! 🚀
