import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AppState:
    recording: bool = False
    manual_override: Optional[bool] = None
    current_session_id: Optional[int] = None
    speaker_lock: str = "auto"
    lock: threading.Lock = field(default_factory=threading.Lock)
