import logging
from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="InsureAssist AI",
    version="0.1.0",
    description="AI-powered insurance lead and policy assistant.",
    debug=settings.debug,
)

app.include_router(router, prefix=settings.api_prefix)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "InsureAssist AI API is running"}
