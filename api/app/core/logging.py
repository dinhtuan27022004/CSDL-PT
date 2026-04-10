import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Đảm bảo thư mục logs tồn tại
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Cấu hình định dạng log
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%H:%M:%S"

def setup_logging():

    # Lấy root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Tránh việc thêm nhiều handler nếu hàm này được gọi lại
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # 1. Console Handler (Output ra terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 2. Rotating File Handler (Lưu vào file logs/app.log)
    # Xóa file log cũ để làm sạch
    for log_file in LOG_DIR.glob("app.log*"):
        log_file.unlink(missing_ok=True)
    
    file_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("Logging system initialized successfully.")

def get_logger(name: str):
    """
    Hàm tiện ích để lấy logger theo tên module.
    """
    return logging.getLogger(name)
