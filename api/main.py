import uvicorn
from app.core.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    
    print("\n" + "=" * 60)
    print("Image Similarity Search API (MODULAR ARCHITECTURE)")
    print("=" * 60)
    print(f"\nDatabase: {settings.database_url}")
    print(f"Uploads: {settings.uploads_dir}")
    print(f"\nAPI URL: http://localhost:{settings.api_port}")
    print(f"API Docs: http://localhost:{settings.api_port}/docs")
    print("\n" + "=" * 60 + "\n")
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        reload_excludes=["logs/*", "uploads/*", "visualizations/*", "*.log"]
    )
