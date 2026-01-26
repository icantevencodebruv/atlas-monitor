import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

try:
    import yaml
except Exception:
    yaml = None


def load_config(path: Path):
    if not path.exists():
        return {}
    if yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def download_models(models_dir: Path, config: dict) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    whisper_cfg = config.get("asr", {}).get("whisper_cpp", {})
    model_path = whisper_cfg.get("model_path", "./models/ggml-large-v3.bin")
    model_name = Path(model_path).name
    url = whisper_cfg.get(
        "model_url",
        f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{model_name}",
    )
    dest = models_dir / model_name
    if not dest.exists():
        print(f"Downloading {url} -> {dest}")
        urlretrieve(url, dest)


def copy_source(root: Path, dest: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".venv",
        "__pycache__",
        "data",
        "offline_bundle",
        "wheelhouse",
        ".git",
    )
    shutil.copytree(root, dest, dirs_exist_ok=True, ignore=ignore)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="offline_bundle")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--requirements", default="requirements.txt")
    parser.add_argument("--requirements-dev", default="requirements-dev.txt")
    parser.add_argument("--include-dev", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    bundle_root = output
    wheelhouse = bundle_root / "wheelhouse"
    models_dir = bundle_root / "models"

    bundle_root.mkdir(parents=True, exist_ok=True)
    wheelhouse.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "pip", "download", "-r", str(root / args.requirements), "-d", str(wheelhouse)]
    if args.include_dev and (root / args.requirements_dev).exists():
        cmd.extend(["-r", str(root / args.requirements_dev)])
    subprocess.run(cmd, check=True)

    config = load_config(root / args.config)
    download_models(models_dir, config)

    copy_source(root, bundle_root / "app_source")
    print(f"Offline bundle created at {bundle_root}")


if __name__ == "__main__":
    main()
