# Compatibility shim — the application now lives in app/api/.
# uvicorn still references "app.main:app", so this re-exports it.
from app.api.app import app, config  # noqa: F401
