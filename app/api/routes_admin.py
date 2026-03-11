import os
import zipfile
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api import context

router = APIRouter()


@router.get("/admin/failed")
def admin_failed():
    rows = context.db.list_failed_segments()
    payload = []
    for row in rows:
        start = datetime.fromisoformat(row["start_ts"])
        end = datetime.fromisoformat(row["end_ts"])
        duration = (end - start).total_seconds()
        payload.append(
            {
                "id": row["id"],
                "start_ts": row["start_ts"],
                "end_ts": row["end_ts"],
                "duration_sec": duration,
                "attempts": row["attempts"] or 0,
                "error": row["error"] or "",
            }
        )
    return {"rows": payload}


@router.post("/admin/segment/{segment_id}/retry")
def admin_retry(segment_id: int):
    segment = context.db.get_segment(segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    context.db.update_segment_status(segment_id, "pending", None)
    context.segment_queue.put(segment_id)
    return {"status": "queued"}


@router.post("/admin/segment/{segment_id}/delete")
def admin_delete(segment_id: int):
    segment = context.db.get_segment(segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    file_path = segment["file_path"]
    if os.path.exists(file_path):
        os.remove(file_path)
    context.db.delete_segment(segment_id)
    return {"status": "deleted"}


@router.get("/admin/segment/{segment_id}/export")
def admin_export(segment_id: int):
    segment = context.db.get_segment(segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    if segment["status"] != "failed":
        raise HTTPException(status_code=400, detail="Export only allowed for failed segments")
    file_path = segment["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    zip_name = f"segment_{segment_id}.zip"
    zip_path = os.path.join(context.config.storage.exports_dir, zip_name)
    os.makedirs(context.config.storage.exports_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname=os.path.basename(file_path))
    return FileResponse(zip_path, filename=zip_name)
