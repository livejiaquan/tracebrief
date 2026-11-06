#!/usr/bin/env python3
"""Start TraceBrief on Windows 11 or Ubuntu: find the three environments, set the model paths, run the server.

Usage: python scripts/run_tracebrief.py [--config tracebrief.json] [--install DIR] [--set NAME=VALUE ...]
                                        [--host 127.0.0.1] [--port 8765] [--dry-run]

DIR (default: this checkout) holds .venv-app, .venv-vision and .venv-llm, each with bin/python on
POSIX or Scripts\\python.exe on Windows, and the models where the download scripts put them
(models/grounding-dino-tiny, models/qwen, models/qwen2-vl). A JSON config file can name another
install folder and set any environment variable; --set overrides both:

    {"install": "../clean-install/tracebrief", "port": 8786,
     "env": {"TRACEBRIEF_VLM_MODEL": "../vlm/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
             "TRACEBRIEF_FFMPEG": "C:/ffmpeg/bin/ffmpeg.exe"}}

Relative paths in the file are read from the file's folder. The server and every model process
get PYTHONUTF8=1 and offline Hugging Face settings. Standard library only; any Python 3.10+ runs it.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Variables holding a file or folder; relative values in a config file are taken from its folder.
PATHS = {"TRACEBRIEF_VISION_PYTHON", "TRACEBRIEF_LLM_PYTHON", "TRACEBRIEF_VLM_PYTHON", "TRACEBRIEF_MODEL_DIR",
         "TRACEBRIEF_LLM_MODEL", "TRACEBRIEF_VLM_MODEL", "TRACEBRIEF_VLM_PROJECTOR", "TRACEBRIEF_COMPACT_MODEL",
         "TRACEBRIEF_HAILO_MANIFEST", "TRACEBRIEF_RUNTIME"}
EXECUTABLES = {"TRACEBRIEF_FFMPEG", "TRACEBRIEF_FFPROBE"}  # a bare name is looked up on PATH, a path is a path
ENVIRONMENTS = {"TRACEBRIEF_VISION_PYTHON": "vision", "TRACEBRIEF_LLM_PYTHON": "llm"}
MODELS = {"TRACEBRIEF_MODEL_DIR": "models/grounding-dino-tiny",
          "TRACEBRIEF_LLM_MODEL": "models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf",
          "TRACEBRIEF_VLM_MODEL": "models/qwen2-vl/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
          "TRACEBRIEF_VLM_PROJECTOR": "models/qwen2-vl/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Start the local TraceBrief server with its model environments")
    parser.add_argument("--config", type=Path, help="JSON file with install, host, port and env")
    parser.add_argument("--install", type=Path, help="folder holding .venv-app, .venv-vision, .venv-llm and models/")
    parser.add_argument("--python", type=Path, help="interpreter for the server (default: the .venv-app one)")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="set an environment variable")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dry-run", action="store_true", help="print the command and environment, start nothing")
    return parser.parse_args(argv)


def venv_python(install: Path, name: str, windows: bool = os.name == "nt") -> Path:
    venv = Path(install) / f".venv-{name}"
    return venv / "Scripts" / "python.exe" if windows else venv / "bin" / "python"


def from_folder(value: str, folder: Path, bare_name_ok: bool = False) -> str:
    """value read from folder when relative; a bare name such as "ffmpeg" is kept when bare_name_ok."""
    if not value or Path(value).is_absolute() or (bare_name_ok and "/" not in value and "\\" not in value):
        return value
    return str((folder / value).resolve())


def resolve(key: str, value: str, folder: Path) -> str:
    if key in PATHS:
        return from_folder(value, folder)
    if key in EXECUTABLES:
        return from_folder(value, folder, bare_name_ok=True)
    return value


def plan(args, base=None, windows: bool = os.name == "nt"):
    """The server command, its environment and notes on anything missing; starts nothing."""
    base = dict(os.environ if base is None else base)
    config, folder = {}, ROOT
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        folder = Path(args.config).resolve().parent
    install = Path(args.install or from_folder(str(config.get("install") or ROOT), folder)).resolve()
    settings = {key: str(value) for key, value in (config.get("env") or {}).items()}
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise SystemExit(f"--set needs NAME=VALUE, not {item!r}")
        settings[key] = value
    # The config file and --set override inherited variables; the install folder fills in the rest.
    env = {**base, **{key: resolve(key, value, folder) for key, value in settings.items()}}
    notes = []
    for key, name in ENVIRONMENTS.items():
        if key not in env:
            python = venv_python(install, name, windows)
            if python.is_file():
                env[key] = str(python)
            else:
                notes.append(f"{python} not found; {key} falls back to the server's interpreter.")
    for key, relative in MODELS.items():
        if key not in env and (install / relative).exists():
            env[key] = str(install / relative)
    for key in ("TRACEBRIEF_LLM_MODEL", "TRACEBRIEF_VLM_MODEL"):
        if key not in env:
            notes.append(f"{key} is not set and {install / MODELS[key]} is missing; that step uses its fallback.")
    for name in ("ffmpeg", "ffprobe"):
        key = f"TRACEBRIEF_{name.upper()}"
        if not shutil.which(env.get(key) or name, path=env.get("PATH")):
            notes.append(f"{name} not found; install it or set {key} to the executable.")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src")] + ([base["PYTHONPATH"]] if base.get("PYTHONPATH") else []))
    if args.python:
        python = str(args.python)
    elif config.get("python"):
        python = from_folder(str(config["python"]), folder, bare_name_ok=True)
    else:
        app = venv_python(install, "app", windows)
        python = str(app) if app.is_file() else sys.executable
    host = args.host or config.get("host") or "127.0.0.1"
    port = args.port or config.get("port") or int(base.get("TRACEBRIEF_PORT", 8765))
    return [str(python), "-m", "tracebrief", "--host", str(host), "--port", str(port)], env, notes


def main(argv=None):
    args = parse_args(argv)
    command, env, notes = plan(args)
    for note in notes:
        print(f"run_tracebrief: {note}", file=sys.stderr)
    if args.dry_run:
        shown = {key: env[key] for key in sorted(env) if key.startswith(("TRACEBRIEF_", "PYTHON", "HF_", "TRANSFORMERS_"))}
        print(json.dumps({"command": command, "env": shown}, indent=2))
        return 0
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    try:
        return process.wait()
    except KeyboardInterrupt:  # the console sends Ctrl+C to the server too; give it time to stop
        try:
            return process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            return process.wait()


if __name__ == "__main__":
    sys.exit(main())
