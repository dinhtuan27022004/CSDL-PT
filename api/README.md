# Image Similarity Search API (OOP Architecture)

Modern, object-oriented architecture for image similarity search with database integration.

## Structure

```
api/
├── __init__.py
├── main.py                    # Entry point with FastAPI app
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration management
├── models/
│   ├── __init__.py
│   ├── database.py            # SQLAlchemy models
│   └── schemas.py             # Pydantic models
├── services/
│   ├── __init__.py
│   ├── image_processor.py     # Image processing logic
│   └── database_service.py    # Database operations
├── routes/
│   ├── __init__.py
│   ├── images.py              # Image-related endpoints
│   └── health.py              # Health check endpoints
└── utils/
    ├── __init__.py
    └── dependencies.py        # FastAPI dependencies
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Configure environment:

```bash
cp .env.example .env
# Edit .env with your database credentials
```

1. Run the server:

```bash
# From the api/ directory
python main.py

# Or using uvicorn directly
uvicorn api.main:app --reload
```

## Features

- **OOP Architecture**: Clean separation of concerns
- **Dependency Injection**: Easy testing and maintenance
- **Feature Extraction**: Automatic image analysis (brightness, contrast, saturation, edge density, dominant color)
- **Database Integration**: PostgreSQL with SQLAlchemy
- **API Documentation**: Auto-generated Swagger/OpenAPI docs

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /api/images/upload` - Upload images
- `GET /api/images` - List all images
- `GET /api/images/{id}` - Get specific image

## Documentation

Once running, visit:

- API Docs: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
