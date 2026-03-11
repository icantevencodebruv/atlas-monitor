import logging
import webbrowser

from fastapi import FastAPI

from app.api import context
from app.api import (
    routes_admin,
    routes_audio,
    routes_export,
    routes_record,
    routes_setup,
    routes_status,
    routes_ui,
)

logger = logging.getLogger(__name__)

# Re-export config so run.py can import it without touching context internals.
config = context.config

app = FastAPI()


@app.on_event("startup")
def _startup():
    context.worker.start()
    context.scheduler.start()
    context.retry_worker.start()
    try:
        if hasattr(context.backend, "precheck_offline_cache"):
            context.backend.precheck_offline_cache()
    except Exception as exc:
        context.backend_error = str(exc)
        logger.error("ASR backend precheck failed: %s", context.backend_error)
    if context.config.app.open_browser:
        url = f"http://{context.config.app.host}:{context.config.app.port}"
        webbrowser.open(url)


@app.on_event("shutdown")
def _shutdown():
    context.scheduler.shutdown()
    context.worker.shutdown()
    context.retry_worker.shutdown()
    context.recorder.shutdown()
    context.db.close()


app.include_router(routes_ui.router)
app.include_router(routes_status.router)
app.include_router(routes_record.router)
app.include_router(routes_export.router)
app.include_router(routes_audio.router)
app.include_router(routes_setup.router)
app.include_router(routes_admin.router)
