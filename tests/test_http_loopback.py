"""The HTTP server on a real loopback socket: uploads, byte ranges, path traversal and requests from other sites.

The other server tests call Handler methods directly; these go through ThreadingHTTPServer and http.client, so
headers, Content-Length, partial bodies and the socket timeout are what a browser would meet.
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import threading

import pytest

from tracebrief import media, server, workers

VIDEO = bytes(range(256)) * 4  # 1 KiB that ffprobe would never read; media.duration is replaced below


@pytest.fixture
def local(tmp_path, monkeypatch):
    """A TraceBrief server on 127.0.0.1 with its runtime in tmp_path; yields a function that sends one request."""
    monkeypatch.setattr(server, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(server.media, "duration", lambda path: 12.0)
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]

    def request(method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()
    request.port = port
    try:
        yield request
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)


def uploads(tmp_path):
    folder = tmp_path / "runtime" / "uploads"
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def upload(request, body=VIDEO, **headers):
    return request("POST", "/api/uploads", body, {"X-Filename": "clip.mp4", "Content-Type": "video/mp4", **headers})


def test_an_upload_is_served_back_whole_and_in_byte_ranges(local, tmp_path):
    status, _, body = upload(local)
    assert status == 201
    upload_id = json.loads(body)["id"]
    assert uploads(tmp_path) == sorted([f"{upload_id}.json", f"{upload_id}.mp4"])  # no .part left
    path = f"/media/uploads/{upload_id}.mp4"
    status, headers, data = local("GET", path)
    assert status == 200 and data == VIDEO and headers["Content-Type"] == "video/mp4" and headers["Accept-Ranges"] == "bytes"
    status, headers, data = local("GET", path, headers={"Range": "bytes=10-19"})
    assert status == 206 and data == VIDEO[10:20] and headers["Content-Range"] == f"bytes 10-19/{len(VIDEO)}"
    status, headers, data = local("GET", path, headers={"Range": "bytes=-16"})  # the last 16 bytes
    assert status == 206 and data == VIDEO[-16:]
    status, headers, data = local("GET", path, headers={"Range": "bytes=1000-5000"})  # clipped to the end
    assert status == 206 and data == VIDEO[1000:] and headers["Content-Length"] == str(len(VIDEO) - 1000)
    status, headers, _ = local("GET", path, headers={"Range": f"bytes={len(VIDEO)}-"})
    assert status == 416 and headers["Content-Range"] == f"bytes */{len(VIDEO)}"
    status, _, _ = local("GET", path, headers={"Range": "bytes=0-1,4-5"})  # multiple ranges are not served
    assert status == 400


@pytest.mark.parametrize("path", [
    "/media/uploads/../../pyproject.toml", "/media/uploads/..%2F..%2Fpyproject.toml", "/static/../pyproject.toml",
    "/..%2Fpyproject.toml", "/media/jobs/0123456789abcdef/frames/..%2F..%2Fjob.json", "/media/samples/..%2F..%2Fpyproject.toml",
])
def test_paths_outside_the_served_folders_are_not_found(local, path):
    status, _, body = local("GET", path)
    assert status == 404 and b"[project]" not in body


def test_an_upload_that_ends_early_leaves_nothing_behind(local, tmp_path):
    with socket.create_connection(("127.0.0.1", local.port), timeout=10) as raw:
        raw.sendall(b"POST /api/uploads HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Filename: clip.mp4\r\n"
                    b"Content-Type: video/mp4\r\nContent-Length: 4096\r\n\r\n" + VIDEO[:100])
        raw.shutdown(socket.SHUT_WR)
        answer = b""
        while chunk := raw.recv(65536):  # the headers and the JSON body can arrive in separate reads
            answer += chunk
    assert answer.startswith(b"HTTP/1.0 400") and b"Upload ended early" in answer
    assert uploads(tmp_path) == []


def test_a_file_that_is_not_a_readable_video_is_refused_and_removed(local, tmp_path, monkeypatch):
    monkeypatch.setattr(server.media, "duration", lambda path: None)
    status, _, body = upload(local)
    assert status == 400 and "could not be read as a video" in json.loads(body)["error"]
    assert uploads(tmp_path) == []


def test_requests_another_site_makes_the_browser_send_are_refused(local, tmp_path):
    # A page elsewhere can POST a text/plain body without a preflight; its Origin gives it away.
    status, _, body = upload(local, Origin="http://example.com")
    assert status == 403 and uploads(tmp_path) == []
    status, _, _ = local("POST", "/api/jobs", json.dumps({"sample_id": "warehouse-1"}),
                         {"Content-Type": "text/plain", "Origin": "http://example.com"})
    assert status == 403
    status, _, _ = local("POST", "/api/jobs", "{}", {"Content-Type": "application/json", "Origin": "null"})
    assert status == 403
    # DNS rebinding: a name that resolves to this machine still arrives with that name as Host.
    status, _, _ = local("PATCH", "/api/jobs/0123456789abcdef/events/event-001", "{}",
                         {"Content-Type": "application/json", "Host": f"attacker.example:{local.port}"})
    assert status == 403
    # Without Origin (a script), the body must still be declared JSON.
    status, _, body = local("POST", "/api/jobs", json.dumps({"sample_id": "warehouse-1"}), {"Content-Type": "text/plain"})
    assert status == 415 and "application/json" in json.loads(body)["error"]
    assert not (tmp_path / "runtime" / "jobs").exists()  # nothing was queued


def test_the_page_itself_may_post(local, tmp_path, monkeypatch):
    origin = f"http://127.0.0.1:{local.port}"
    status, _, _ = upload(local, Origin=origin)
    assert status == 201
    status, _, body = local("POST", "/api/jobs", json.dumps({"upload_id": "0" * 16}),
                            {"Content-Type": "application/json; charset=utf-8", "Origin": origin})
    assert status == 404  # past the origin and type checks: that upload does not exist
    monkeypatch.setenv("TRACEBRIEF_ALLOWED_HOSTS", "tracebrief.local")
    status, _, _ = upload(local, Host=f"tracebrief.local:{local.port}", Origin=f"http://tracebrief.local:{local.port}")
    assert status == 201


def test_candidate_ids_past_999_reach_the_review_routes(local):
    status, _, body = local("PATCH", "/api/jobs/0123456789abcdef/events/event-1000", "{}", {"Content-Type": "application/json"})
    assert status == 404 and json.loads(body)["error"] == "Review not found"  # the route matched; the job is absent
    status, _, body = local("PATCH", "/api/jobs/0123456789abcdef/events/event-12", "{}", {"Content-Type": "application/json"})
    assert status == 404 and json.loads(body)["error"] == "Not found"


def test_a_stalled_client_is_disconnected(local, monkeypatch):
    monkeypatch.setattr(server.Handler, "timeout", 0.5)
    with socket.create_connection(("127.0.0.1", local.port), timeout=10) as raw:
        raw.sendall(b"POST /api/uploads HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 100\r\n\r\n")
        assert raw.recv(65536) == b""  # closed by the server, without an answer, once the body stops coming


def test_a_corrupt_job_record_does_not_stop_the_server_from_starting(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(server, "RUNTIME", tmp_path)
    broken, running = tmp_path / "jobs" / "a", tmp_path / "jobs" / "b"
    for folder in (broken, running):
        folder.mkdir(parents=True)
    (broken / "job.json").write_text("{not json", encoding="utf-8")
    server.write_json(running / "job.json", {"id": "b", "status": "running"})
    server.recover_interrupted()
    assert server.load_json(running / "job.json")["status"] == "failed"
    assert "skipped" in capsys.readouterr().err


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX shell script as the interpreter")
def test_the_detector_process_is_stopped_when_the_job_fails_around_it(tmp_path, monkeypatch):
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\nexec sleep 60\n", encoding="utf-8")
    interpreter.chmod(0o755)
    monkeypatch.setenv("TRACEBRIEF_VISION_PYTHON", str(interpreter))
    monkeypatch.setattr(server, "RUNTIME", tmp_path / "runtime")
    started = []
    real = server.subprocess.Popen

    def popen(*args, **kwargs):
        started.append(real(*args, **kwargs))
        return started[-1]
    monkeypatch.setattr(server.subprocess, "Popen", popen)
    with pytest.raises(FileNotFoundError):  # the job record is gone, so the first progress write fails
        server.run_vision_process("0123456789abcdef", tmp_path, "vision_worker.py", tmp_path / "request.json",
                                  tmp_path / "raw.json", 10, lambda elapsed: "Detecting")
    assert started and started[0].poll() is not None  # terminated and reaped, not left running


def test_probe_results_are_kept_for_recent_videos_only(monkeypatch):
    cache = media.Recent(2)
    cache["a"], cache["b"] = 1, 2
    assert cache["a"] == 1  # used, so "b" is now the oldest
    cache["c"] = 3
    assert list(cache) == ["a", "c"] and "b" not in cache
    assert isinstance(media.STREAMS, media.Recent) and media.STREAMS.size >= 3


def test_a_worker_only_queues_protocol_lines():
    read, write = os.pipe()
    lines = workers.queue.Queue(maxsize=workers.LINES_KEPT)

    def write_all():  # from a thread: 13 KB is more than a Windows pipe holds before someone reads it
        with os.fdopen(write, "wb") as out:
            out.write(b"llama_model_loader: noise\n" * 500 + b'{"ready": true}\n')
    writer = threading.Thread(target=write_all)
    writer.start()
    workers.Worker._read(os.fdopen(read, "rb"), lines)
    writer.join(10)
    assert lines.get_nowait() == b'{"ready": true}\n' and lines.get_nowait() is None
