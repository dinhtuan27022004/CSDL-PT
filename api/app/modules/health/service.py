from .schema import HealthResponse
from ...core.config import get_settings

class HealthService:
    def __init__(self):
        self.settings = get_settings()

    def get_health_status(self) -> HealthResponse:
        return HealthResponse(
            status="healthy",
            service=self.settings.api_title,
            version=self.settings.api_version
        )
