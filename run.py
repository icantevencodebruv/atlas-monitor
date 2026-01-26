import uvicorn

from app.main import config


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=config.app.host, port=config.app.port, log_level="info")
