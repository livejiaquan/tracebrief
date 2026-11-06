"""Server robustness: atomic job writes, errors, served types, time limits, progress and configurable ffmpeg."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import pytest

from tracebrief import media, server

ROOT = Path(__file__).resolve().parents[1]
SOURCE = {"name": "Fixture", "source_label": "Synthetic fixture", "video_url": "/media/samples/fixture"}


def handler(path="/"):
    value = object.__new__(server.Handler)
    value.path, value.command, value.headers = path, "GET", {}
    value.responses, value.sent = [], []
    value.json_response = lambda body, status=200: value.responses.append((status, body))
    value.send_response = lambda status: value.sent.append(("status", status))
    value.send_header = lambda name, content: value.sent.append((name, content))
    value.end_headers = lambda: None
    value.wfile = io.BytesIO()
    return value


def test_job_writes_retry_while_windows_refuses_the_replace(tmp_path, monkeypatch):
    target = tmp_path / "job.json"
    real, failures = server.os.replace, []

    def replace(source, destination):
        if len(failures) < 2:
            failures.append(destination)
            raise PermissionError("in use by a reader")
        real(source, destination)
    monkeypatch.setattr(server.os, "replace", replace)
    monkeypatch.setattr(server.time, "sleep", lambda seconds: None)
    server.write_json(target, {"status": "running", "note": "中文"})
    assert server.load_json(target) == {"status": "running", "note": "中文"} and len(failures) == 2
    assert [p.name for p in tmp_path.iterdir()] == ["job.json"]  # no temporary file left behind


def test_a_replace_that_keeps_failing_raises_and_leaves_the_old_file(tmp_path, monkeypatch):
    target = tmp_path / "job.json"
    server.write_json(target, {"status": "queued"})

    def replace(source, destination):
        raise PermissionError("in use")
    monkeypatch.setattr(server.os, "replace", replace)
    monkeypatch.setattr(server.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        server.write_json(target, {"status": "running"})
    assert server.load_json(target) == {"status": "queued"} and [p.name for p in tmp_path.iterdir()] == ["job.json"]


def test_job_reads_retry_while_windows_refuses_access(tmp_path, monkeypatch):
    target = tmp_path / "job.json"
    target.write_text('{"status": "running"}', encoding="utf-8")
    real, failures = Path.read_text, []

    def read_text(self, *args, **kwargs):
        if not failures:
            failures.append(self)
            raise PermissionError("being replaced")
        return real(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(server.time, "sleep", lambda seconds: None)
    assert server.load_json(target) == {"status": "running"} and failures == [target]


def test_unexpected_get_error_is_a_500_not_a_dropped_connection(monkeypatch):
    def broken():
        raise RuntimeError("disk vanished")
    monkeypatch.setattr(server, "backend", broken)
    request = handler("/api/health")
    request.do_GET()
    assert request.responses[0][0] == 500 and "RuntimeError" in request.responses[0][1]["error"]


def test_unexpected_post_and_patch_errors_are_500s(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(server, "samples", broken)
    request = handler("/api/jobs")
    request.read_json = lambda: {"sample_id": "warehouse-1"}
    request.do_POST()
    assert request.responses[0][0] == 500
    monkeypatch.setattr(server, "read_job", broken)
    request = handler("/api/jobs/ffffffffffffffff/events/event-001")
    request.read_json = lambda: {"notes": "x"}
    request.do_PATCH()
    assert request.responses[0][0] == 500


@pytest.mark.parametrize("name, expected", [("app.js", "text/javascript; charset=utf-8"), ("styles.css", "text/css; charset=utf-8"),
                                            ("index.html", "text/html; charset=utf-8"), ("clip.mp4", "video/mp4"),
                                            ("frame.jpg", "image/jpeg"), ("data.json", "application/json; charset=utf-8")])
def test_served_types_do_not_depend_on_the_registry(tmp_path, monkeypatch, name, expected):
    path = tmp_path / name
    path.write_bytes(b"x")
    monkeypatch.setattr(server.mimetypes, "guess_type", lambda name: ("text/plain", None))  # a misconfigured registry
    request = handler()
    request.send_file(path)
    assert ("Content-Type", expected) in request.sent and ("X-Content-Type-Options", "nosniff") in request.sent


def test_source_hash_reuses_the_detectors_streamed_hash(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture video")
    digest = hashlib.sha256(b"fixture video").hexdigest()
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("the video was read into memory"))
    assert server.source_sha256({"video": {"path": str(video), "sha256": "a" * 64}}, video) == "a" * 64
    # Missing or for another file: hashed here, in chunks.
    assert server.source_sha256({"video": {"path": str(video)}}, video) == digest
    assert server.source_sha256({"video": {"path": "other.mp4", "sha256": "a" * 64}}, video) == digest


def test_time_limit_scales_with_duration_between_floor_and_cap():
    compact, dino = {"backend": "compact-onnx", "sample_fps": 1}, {"sample_fps": 1}
    assert server.vision_timeout(server.vision_estimate(dino, 10)) == 1800  # a Stage I clip keeps the 30 minutes
    assert server.vision_timeout(server.vision_estimate(compact, None)) == 1800
    hour = server.vision_timeout(server.vision_estimate(compact, 3600))
    day = server.vision_timeout(server.vision_estimate(compact, 24 * 3600))
    assert 1800 <= hour < day <= server.VISION_TIMEOUT_CAP
    assert server.vision_timeout(server.vision_estimate(dino, 3600)) == server.VISION_TIMEOUT_CAP


def test_progress_keeps_moving_past_the_estimate_and_stays_in_its_stage():
    values = [server.stage_progress(10, 69, t, 60) for t in (0, 30, 60, 120, 300, 600)]
    assert values == sorted(values) and len(set(values)) == len(values)
    assert values[0] == 10 and values[2] == pytest.approx(10 + 0.8 * 59, abs=0.1)
    assert 65 < values[-1] <= 69  # the old loop stopped at 65
    assert server.stage_progress(10, 69, 10 ** 6, 60) <= 69


def test_ticking_moves_progress_during_a_blocking_step(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUNTIME", tmp_path)
    job_id = "abcdefabcdefabcd"
    server.write_json(server.job_path(job_id), {"id": job_id, "status": "running"})
    seen = []
    real = server.update_job
    monkeypatch.setattr(server, "update_job", lambda job, **updates: seen.append(updates) or real(job, **updates))
    with server.ticking(job_id, 72, 80, 2, "Visual check of candidate 1 of 1"):
        time.sleep(2.5)
    assert seen[0] == {"progress": 72, "message": "Visual check of candidate 1 of 1"}
    assert len(seen) >= 3 and 72 < seen[-1]["progress"] < 80 and "s elapsed" in seen[-1]["message"]
    count = len(seen)
    time.sleep(1.2)
    assert len(seen) == count  # stopped with the step


def queue_cpu_job(server_module, tmp_path, monkeypatch, raw):
    monkeypatch.setattr(server_module, "RUNTIME", tmp_path / "runtime")
    monkeypatch.delenv("TRACEBRIEF_BACKEND", raising=False)

    class Process:
        pid, returncode = 1, 0

        def __init__(self, command, **kwargs):
            server_module.write_json(Path(command[3]), raw)

        def poll(self):
            return 0
    monkeypatch.setattr(server_module.subprocess, "Popen", Process)
    job_id = "abcdefabcdefabcd"
    server_module.write_json(server_module.job_path(job_id), {"id": job_id, "status": "queued"})
    return job_id


def test_cpu_job_records_the_scaled_limit_and_the_streamed_hash(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"; video.write_bytes(b"fixture")
    hours = 4 * 3600
    raw = {"video": {"duration_s": hours, "path": str(video), "sha256": "b" * 64}, "model": {"id": "fixture"}, "settings": {}, "events": []}
    job_id = queue_cpu_job(server, tmp_path, monkeypatch, raw)
    server.process_job(job_id, video, {**SOURCE, "duration": hours}, {"backend": "compact-onnx", "sample_fps": 1})
    job = server.read_job(job_id)
    assert job["status"] == "completed" and job["result"]["source_sha256"] == "b" * 64
    limit = server.load_json(server.job_path(job_id).parent / "process.json")["timeout_seconds"]
    assert limit == server.vision_timeout(server.vision_estimate({"backend": "compact-onnx", "sample_fps": 1}, hours)) > 1800


def test_upload_records_its_duration_for_the_time_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUNTIME", tmp_path)
    monkeypatch.setattr(server.media, "duration", lambda path: 42.5)
    request = handler("/api/uploads")
    request.command = "POST"
    request.headers = {"Content-Length": "7", "X-Filename": "clip.mp4"}
    request.rfile = io.BytesIO(b"fixture")
    request.do_POST()
    status, body = request.responses[0]
    assert status == 201
    meta = server.load_json(tmp_path / "uploads" / f"{body['id']}.json")
    assert meta["duration"] == 42.5


def test_ffmpeg_and_ffprobe_can_be_named_explicitly(tmp_path, monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[-1].endswith(".jpg"):  # the frame ffmpeg would write
            Path(command[-1]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(command, 0, stdout="12.500000\n", stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("TRACEBRIEF_FFMPEG", str(tmp_path / "ffmpeg-7" / "ffmpeg.exe"))
    monkeypatch.setenv("TRACEBRIEF_FFPROBE", str(tmp_path / "ffmpeg-7" / "ffprobe.exe"))
    media.crop_at(tmp_path / "clip.mp4", 2.0, [0, 0, 10, 10], tmp_path / "crop.jpg", 25.0)
    server.frame_image(tmp_path / "clip.mp4", tmp_path / "frame.jpg", 2.0, 25.0)
    assert media.duration(tmp_path / "clip.mp4") == 12.5
    assert [c[0] for c in commands] == [str(tmp_path / "ffmpeg-7" / "ffmpeg.exe")] * 2 + [str(tmp_path / "ffmpeg-7" / "ffprobe.exe")]


def test_ffmpeg_defaults_to_the_path_lookup(monkeypatch):
    monkeypatch.delenv("TRACEBRIEF_FFMPEG", raising=False)
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/opt/bin/{name}")
    assert media.ffmpeg() == "/opt/bin/ffmpeg"
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    assert media.ffmpeg() == "ffmpeg"


@pytest.mark.parametrize("stdout, returncode", [("N/A\n", 0), ("", 1), ("0\n", 0)])
def test_unreadable_duration_is_none(monkeypatch, stdout, returncode):
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=""))
    assert media.duration(Path("clip.mp4")) is None


def test_missing_ffprobe_is_none(monkeypatch):
    def run(command, **kwargs):
        raise FileNotFoundError(command[0])
    monkeypatch.setattr(subprocess, "run", run)
    assert media.duration(Path("clip.mp4")) is None


def test_health_reports_the_resident_workers(monkeypatch):
    monkeypatch.delenv("TRACEBRIEF_BACKEND", raising=False)
    request = handler("/api/health")
    request.do_GET()
    body = request.responses[0][1]
    assert body["backend"] == "cpu" and body["resident_workers"] in (True, False) and isinstance(body["workers"], dict)
