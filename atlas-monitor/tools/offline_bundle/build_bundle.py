import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

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


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_manifest_entries(manifest: list, path: Path, root: Path) -> None:
    if path.is_file():
        manifest.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        )
        return
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        manifest.append(
            {
                "path": str(file_path.relative_to(root)),
                "sha256": _sha256_file(file_path),
                "size": file_path.stat().st_size,
            }
        )


def _copy_any(src: Path, dest: Path) -> None:
    if src.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return
    shutil.copytree(src, dest, dirs_exist_ok=True)


def _download_to(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)


def _download_and_extract(url: str, dest_dir: Path) -> None:
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    archive = dest_dir.parent / (dest_dir.name + ".zip")
    before = {p.name for p in dest_dir.parent.iterdir()}
    _download_to(url, archive)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest_dir.parent)
    archive.unlink(missing_ok=True)
    if dest_dir.exists():
        return
    after = [p for p in dest_dir.parent.iterdir() if p.is_dir() and p.name not in before]
    if len(after) == 1:
        after[0].rename(dest_dir)


def _bundle_asset(
    root: Path,
    models_dir: Path,
    manifest: list,
    logical_group: str,
    path_value: str,
    url_value: str = "",
    allow_archive: bool = False,
) -> None:
    if not path_value:
        return
    src = _resolve_path(root, path_value)
    normalized = path_value.replace("\\", "/")
    if normalized.startswith("./models/"):
        dest = models_dir / normalized[len("./models/") :]
    elif normalized.startswith("models/"):
        dest = models_dir / normalized[len("models/") :]
    else:
        dest = models_dir / logical_group / Path(path_value).name
    if src.exists():
        print(f"Bundling local asset {src} -> {dest}")
        _copy_any(src, dest)
        _append_manifest_entries(manifest, dest, models_dir)
        return
    if not url_value:
        print(f"WARN: missing asset and no URL configured: {path_value}")
        return
    if allow_archive and str(url_value).lower().endswith(".zip"):
        _download_and_extract(url_value, dest)
    else:
        _download_to(url_value, dest)
    _append_manifest_entries(manifest, dest, models_dir)


def _write_manifest(models_dir: Path, entries: list) -> None:
    payload = {
        "version": 1,
        "files": sorted(entries, key=lambda item: item["path"]),
    }
    manifest_path = models_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote model manifest: {manifest_path}")


def download_models(root: Path, models_dir: Path, config: dict) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list = []
    asr_cfg = config.get("asr", {})

    whisper_cfg = asr_cfg.get("whisper_cpp", {})
    whisper_path = whisper_cfg.get("model_path", "./models/ggml-large-v3.bin")
    whisper_name = Path(whisper_path).name
    whisper_url = whisper_cfg.get(
        "model_url",
        f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{whisper_name}",
    )
    _bundle_asset(
        root=root,
        models_dir=models_dir,
        manifest=manifest_entries,
        logical_group="whisper_cpp",
        path_value=whisper_path,
        url_value=whisper_url,
        allow_archive=False,
    )

    pipeline_cfg = asr_cfg.get("pipeline_local", {})
    _bundle_asset(
        root=root,
        models_dir=models_dir,
        manifest=manifest_entries,
        logical_group="pipeline_local",
        path_value=pipeline_cfg.get("silero_model_path", ""),
        url_value=pipeline_cfg.get("silero_model_url", ""),
    )
    _bundle_asset(
        root=root,
        models_dir=models_dir,
        manifest=manifest_entries,
        logical_group="pipeline_local",
        path_value=pipeline_cfg.get("fasttext_model_path", ""),
        url_value=pipeline_cfg.get(
            "fasttext_model_url",
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
        ),
    )
    _bundle_asset(
        root=root,
        models_dir=models_dir,
        manifest=manifest_entries,
        logical_group="pipeline_local",
        path_value=pipeline_cfg.get("vosk_model_en_path", ""),
        url_value=pipeline_cfg.get("vosk_model_en_url", ""),
        allow_archive=True,
    )
    _bundle_asset(
        root=root,
        models_dir=models_dir,
        manifest=manifest_entries,
        logical_group="pipeline_local",
        path_value=pipeline_cfg.get("vosk_model_de_path", ""),
        url_value=pipeline_cfg.get("vosk_model_de_url", ""),
        allow_archive=True,
    )
    _bundle_asset(
        root=root,
        models_dir=models_dir,
        manifest=manifest_entries,
        logical_group="pipeline_local",
        path_value=pipeline_cfg.get("wav2vec2_model_en_path", ""),
        url_value=pipeline_cfg.get("wav2vec2_model_en_url", ""),
    )
    _bundle_asset(
        root=root,
        models_dir=models_dir,
        manifest=manifest_entries,
        logical_group="pipeline_local",
        path_value=pipeline_cfg.get("wav2vec2_model_de_path", ""),
        url_value=pipeline_cfg.get("wav2vec2_model_de_url", ""),
    )
    _bundle_asset(
        root=root,
        models_dir=models_dir,
        manifest=manifest_entries,
        logical_group="pipeline_local",
        path_value=pipeline_cfg.get("pyannote_model_path", ""),
        url_value=pipeline_cfg.get("pyannote_model_url", ""),
    )
    _bundle_asset(
        root=root,
        models_dir=models_dir,
        manifest=manifest_entries,
        logical_group="pipeline_local",
        path_value=pipeline_cfg.get("hf_cache_dir", ""),
    )

    _write_manifest(models_dir, manifest_entries)


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
    download_models(root, models_dir, config)

    copy_source(root, bundle_root / "app_source")
    print(f"Offline bundle created at {bundle_root}")


if __name__ == "__main__":
    main()
