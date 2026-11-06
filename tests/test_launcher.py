"""scripts/run_tracebrief.py: environment discovery on POSIX and Windows layouts, config files, overrides."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def launcher():
    spec = importlib.util.spec_from_file_location("run_tracebrief", ROOT / "scripts" / "run_tracebrief.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(tmp_path, windows):
    """An install folder with the three environments and the downloaded models."""
    folder = tmp_path / "install"
    for name in ("app", "vision", "llm"):
        python = folder / f".venv-{name}" / ("Scripts/python.exe" if windows else "bin/python")
        python.parent.mkdir(parents=True)
        python.write_bytes(b"")
    for relative in ("models/grounding-dino-tiny", "models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                     "models/qwen2-vl/Qwen2-VL-2B-Instruct-Q4_K_M.gguf", "models/qwen2-vl/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"):
        path = folder / relative
        if relative.endswith("tiny"):
            path.mkdir(parents=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
    return folder


@pytest.mark.parametrize("windows, python", [(False, Path("bin") / "python"), (True, Path("Scripts") / "python.exe")])
def test_finds_the_three_environments_and_the_models(launcher, tmp_path, windows, python):
    folder = install(tmp_path, windows)
    command, env, notes = launcher.plan(launcher.parse_args(["--install", str(folder), "--port", "8786"]), base={"PATH": ""}, windows=windows)
    assert command == [str(folder.resolve() / ".venv-app" / python), "-m", "tracebrief", "--host", "127.0.0.1", "--port", "8786"]
    assert env["TRACEBRIEF_VISION_PYTHON"] == str(folder.resolve() / ".venv-vision" / python)
    assert env["TRACEBRIEF_LLM_PYTHON"] == str(folder.resolve() / ".venv-llm" / python)
    assert env["TRACEBRIEF_VLM_PROJECTOR"].endswith("mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf")
    assert env["PYTHONUTF8"] == "1" and env["HF_HUB_OFFLINE"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT / "src")
    assert any("ffmpeg not found" in note for note in notes)  # PATH is empty here


def test_config_file_paths_are_read_from_its_folder_and_set_overrides(launcher, tmp_path):
    folder = install(tmp_path, windows=False)
    config = tmp_path / "configs" / "tracebrief.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"install": "../install", "port": 9000, "host": "0.0.0.0", "env": {
        "TRACEBRIEF_VLM_MODEL": "../vlm/model.gguf", "TRACEBRIEF_FFMPEG": "ffmpeg", "TRACEBRIEF_FFPROBE": "../bin/ffprobe",
        "TRACEBRIEF_BACKEND": "cpu"}}), encoding="utf-8")
    args = launcher.parse_args(["--config", str(config), "--set", "TRACEBRIEF_BACKEND=hailo", "--set", "TRACEBRIEF_VLM_THREADS=6"])
    command, env, _ = launcher.plan(args, base={"TRACEBRIEF_VLM_MODEL": "/inherited.gguf"}, windows=False)
    assert command[-4:] == ["--host", "0.0.0.0", "--port", "9000"]
    assert env["TRACEBRIEF_VLM_MODEL"] == str((tmp_path / "vlm" / "model.gguf").resolve())  # the file beats the inherited value
    assert env["TRACEBRIEF_FFMPEG"] == "ffmpeg"  # a bare name stays a PATH lookup
    assert env["TRACEBRIEF_FFPROBE"] == str((tmp_path / "bin" / "ffprobe").resolve())
    assert env["TRACEBRIEF_BACKEND"] == "hailo" and env["TRACEBRIEF_VLM_THREADS"] == "6"
    assert env["TRACEBRIEF_LLM_PYTHON"] == str(folder.resolve() / ".venv-llm" / "bin" / "python")


def test_inherited_variables_are_kept_when_nothing_overrides_them(launcher, tmp_path):
    install(tmp_path, windows=False)
    args = launcher.parse_args(["--install", str(tmp_path / "install")])
    _, env, _ = launcher.plan(args, base={"TRACEBRIEF_LLM_MODEL": "/models/own.gguf", "TRACEBRIEF_PORT": "8800"}, windows=False)
    assert env["TRACEBRIEF_LLM_MODEL"] == "/models/own.gguf"


def test_missing_environments_fall_back_with_a_note(launcher, tmp_path):
    args = launcher.parse_args(["--install", str(tmp_path / "empty")])
    command, env, notes = launcher.plan(args, base={}, windows=False)
    assert "TRACEBRIEF_VISION_PYTHON" not in env and any(".venv-vision" in note for note in notes)
    assert command[0] == launcher.sys.executable and command[-1] == "8765"


def test_bad_set_is_refused(launcher):
    with pytest.raises(SystemExit):
        launcher.plan(launcher.parse_args(["--set", "NO_EQUALS_SIGN"]), base={})


def test_dry_run_prints_the_plan_and_starts_nothing(launcher, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("dry run started the server"))
    assert launcher.main(["--install", str(install(tmp_path, os.name == "nt")), "--dry-run"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["command"][1:3] == ["-m", "tracebrief"] and shown["env"]["PYTHONUTF8"] == "1"
