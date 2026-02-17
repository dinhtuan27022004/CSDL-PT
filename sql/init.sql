-- CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE image_metadata (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    
    -- Kích thước ảnh
    width INTEGER,
    height INTEGER,
    
    -- Các chỉ số thống kê (Basic Stats)
    brightness FLOAT,      -- Độ sáng (mean)
    contrast FLOAT,        -- Độ tương phản (std)
    saturation FLOAT,      -- Độ bão hòa (mean kênh S)
    edge_density FLOAT,    -- Mật độ cạnh (Canny)
    
    -- Màu sắc chủ đạo (Dominant Color)
    dominant_color_hex CHAR(7), -- Lưu mã màu dạng #RRGGBB
    
    -- Vector Embedding (ResNet50 thường trả về vector 2048 chiều)
--     embedding vector(2048),
    
    -- Dữ liệu đặc trưng nâng cao (Lưu dạng JSONB để tối ưu truy vấn)
    -- Bao gồm: Histogram, Color moments, GLCM, Gabor, Contour data
    features_json JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index cho tìm kiếm vector nhanh (Cosines Distance)
-- CREATE INDEX ON image_metadata USING hnsw (embedding vector_cosine_ops);