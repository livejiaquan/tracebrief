"""scripts/check_setup.py checks the backend that TRACEBRIEF_BACKEND selects."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(tmp_path, **env):
    environ = {key: value for key, value in os.environ.items() if not key.startswith("TRACEBRIEF_")}
    environ.update(env, PYTHONUTF8="1")
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_setup.py")], capture_output=True, text=True,
                            encoding="utf-8", env=environ, cwd=tmp_path, timeout=120)
    report = json.loads(result.stdout)
    return result.returncode, report, {row["check"]: row for row in report["checks"]}


def manifest(tmp_path, vlm_sha=None):
    hefs = {name: tmp_path / f"{name}.hef" for name in ("detector", "vlm")}
    for name, file in hefs.items():
        file.write_bytes(name.encode())
    entries = {name: {"file": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()} for name, file in hefs.items()}
    if vlm_sha:
        entries["vlm"]["sha256"] = vlm_sha
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"hailort": "5.3", "models": entries}), encoding="utf-8")
    return path


def test_cpu_backend_checks_the_cpu_models(tmp_path):
    code, report, rows = run(tmp_path)
    assert report["backend"] == "cpu"
    assert {"vision imports", "generation model", "vision model"} <= set(rows)
    assert "HEFs" not in rows


def test_hailo_backend_checks_hailort_and_every_hef_hash(tmp_path):
    code, report, rows = run(tmp_path, TRACEBRIEF_BACKEND="hailo", TRACEBRIEF_HAILO_MANIFEST=str(manifest(tmp_path)))
    assert report["backend"] == "hailo" and "the device is not opened" in report["scope"]
    assert "vision imports" not in rows and "generation model" not in rows
    assert rows["HEFs"]["ready"], rows["HEFs"]["detail"]
    try:
        import hailo_platform  # noqa: F401
    except ImportError:
        assert not rows["HailoRT"]["ready"] and code == 1


def test_hailo_backend_reports_a_hash_mismatch(tmp_path):
    code, report, rows = run(tmp_path, TRACEBRIEF_BACKEND="hailo", TRACEBRIEF_HAILO_MANIFEST=str(manifest(tmp_path, "00" * 32)))
    assert code == 1 and not rows["HEFs"]["ready"]
    assert "mismatch" in rows["HEFs"]["detail"]


def test_repository_manifest_pins_the_public_vlm_hef():
    models = json.loads((ROOT / "config" / "hailo-manifest.json").read_text(encoding="utf-8"))["models"]
    assert models["vlm"]["sha256"] == "7b2a8ee15269172a6b5e63690ac88d3e59ff9eeaacb3d99d6e1049bc59fbc407"
    assert models["vlm"]["source"].startswith("https://dev-public.hailo.ai/v5.3.0/")
    assert models["detector"]["sha256"] is None  # compiled locally; a rebuild gives a different hash
