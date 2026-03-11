from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter()

_UI_DIR = Path(__file__).parent.parent / "ui"


@router.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((_UI_DIR / "home.html").read_text(encoding="utf-8"))


@router.get("/setup", response_class=HTMLResponse)
def setup_page():
    return HTMLResponse((_UI_DIR / "setup.html").read_text(encoding="utf-8"))


@router.get("/mic")
def mic_redirect():
    return RedirectResponse(url="/setup", status_code=301)


@router.get("/admin", response_class=HTMLResponse)
def admin_page():
    return HTMLResponse((_UI_DIR / "admin.html").read_text(encoding="utf-8"))
