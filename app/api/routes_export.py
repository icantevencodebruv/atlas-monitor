import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api import context
from app.services.exporter import build_export, compute_range

router = APIRouter()


class ExportRequest(BaseModel):
    range: str
    start: str | None = None
    end: str | None = None


@router.post("/export")
def export_range(req: ExportRequest):
    now = datetime.now(timezone.utc)
    session_row = context.db.get_latest_session()
    if req.range == "custom" and (not req.start or not req.end):
        raise HTTPException(
            status_code=400, detail="Custom range requires start and end timestamps."
        )
    try:
        start_ts, end_ts = compute_range(
            req.range, now, session_row, req.start, req.end, context.config.work_hours.timezone
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date/time: {exc}") from exc
    if start_ts >= end_ts:
        raise HTTPException(status_code=400, detail="Start must be before end.")
    export_id = build_export(context.db, req.range, start_ts, end_ts, context.config.storage.exports_dir)
    return {"id": export_id}


@router.get("/exports/recent")
def exports_recent(limit: int = 8):
    rows = context.db.list_exports(limit)
    payload = []
    for row in rows:
        file_path = row["file_path"]
        size_bytes = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        payload.append(
            {
                "id": row["id"],
                "created_ts": row["created_ts"],
                "range_label": row["range_label"],
                "start_ts": row["start_ts"],
                "end_ts": row["end_ts"],
                "file_path": row["file_path"],
                "size_bytes": size_bytes,
            }
        )
    return {"rows": payload}


@router.get("/download/{export_id}")
def download(export_id: int):
    export = context.db.get_export(export_id)
    if not export:
        raise HTTPException(status_code=404, detail="Export not found")
    return FileResponse(export["file_path"], filename=os.path.basename(export["file_path"]))
