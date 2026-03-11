import os

import uvicorn

from app.config import load_config

if __name__ == "__main__":
    config = load_config(os.environ.get("APP_CONFIG", "config.yaml"))
    uvicorn.run("app.main:app", host=config.app.host, port=config.app.port, log_level="info")
