from fastapi import Header, HTTPException
from app.api.core.config import settings

def verify_api_key(x_api_key: str = Header(default=None)):
    """
    Simple API key check. Skips check if APP_API_KEY is not set (dev mode).
    """
    if not settings.app_api_key:
        return  # no key configured — auth disabled

    if x_api_key != settings.app_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")