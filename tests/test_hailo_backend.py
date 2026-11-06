"""UGen300 path driven by a fake hailo_platform and synthetic buffers; no model or device is run.

What a real device must still confirm is marked ugen300 (skipped without one) or listed in
docs/HARDWARE_PLAN.md. A recorded device buffer can be dropped in as tests/fixtures/hailo/nms_by_class.npz.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from tracebrief import look_prompt
from tracebrief.hailo_backend import DeviceError, HailoRuntime, parse_nms, to_source_boxes, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "hailo" / "nms_by_class.npz"
SOURCE = {"name": "Fixture", "source_label": "Synthetic fixture", "video_url": "/media/samples/fixture"}
PERSON, FORKLIFT = (100, 50, 140, 200), (150, 60, 260, 200)  # x0, y0, x1, y1 in a 320x240 frame


def normalized(box, score, width=320, height=240):
    """A source box as the HEF reports it: [y0, x0, y1, x1, score] on the 416 letterbox, scaled to 0-1."""
    r = min(416 / height, 416 / width) / 416
    return [box[1] * r, box[0] * r, box[3] * r, box[2] * r, score]


def compact_settings():
    """The Fast search settings, the profile the detector HEF is compiled for."""
    profiles = json.loads((ROOT / "config" / "search_profiles.json").read_text(encoding="utf-8"))["profiles"]
    fast = next(p for p in profiles if p["id"] == "fast")
    return json.loads((ROOT / "config" / fast["settings_file"]).read_text(encoding="utf-8"))


def hailo_server(monkeypatch, tmp_path, manifest):
    from tracebrief import server
    monkeypatch.setenv("TRACEBRIEF_BACKEND", "hailo")
    monkeypatch.setenv("TRACEBRIEF_HAILO_MANIFEST", str(manifest))
    monkeypatch.setattr(server, "DEVICE", {"runtime": None, "error": None})
    monkeypatch.setattr(server, "RUNTIME", tmp_path / "runtime")
    return server


def queued_job(server, job_id="abcdefabcdefabcd"):
    server.write_json(server.job_path(job_id), {"id": job_id, "status": "queued"})
    return job_id


def forbid_cpu_models(server, monkeypatch):
    """Any CPU model starts through a subprocess (vision worker, llama.cpp look and draft)."""
    def forbidden(*args, **kwargs):
        raise AssertionError("a CPU model was started in hailo mode")
    for name in ("Popen", "run"):
        monkeypatch.setattr(server.subprocess, name, forbidden)
    monkeypatch.setattr(server, "visual_check", forbidden)
    monkeypatch.setattr(server, "generate", forbidden)


def synthetic_video(path, seconds=3, fps=10, size=(320, 240)):
    np = pytest.importorskip("numpy"); cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(seconds * fps):
        writer.write(np.full((size[1], size[0], 3), 40 + i, dtype=np.uint8))
    writer.release()
    return path


def two_moment_video(path, fps=10, seconds=10, size=(320, 240)):
    """Bright first and last three seconds, dark between: with pair_when_bright, two candidates (0-2 s and 7-9 s)."""
    np = pytest.importorskip("numpy"); cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(seconds * fps):
        bright = i < 3 * fps or i >= (seconds - 3) * fps
        writer.write(np.full((size[1], size[0], 3), 200 if bright else 40, dtype=np.uint8))
    writer.release()
    return path


def pair_when_bright(frame):
    return [[normalized(PERSON, .9)], [normalized(FORKLIFT, .8)]] if frame[0, 0, 0] > 120 else [[], []]


def host_media(server, monkeypatch):
    """Stand-ins for the host-side ffmpeg steps: the evidence frame and the visual-check crop."""
    Image = pytest.importorskip("PIL.Image")
    from tracebrief import media

    def frame(video, output, seconds, fps=None):
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24)).save(output)

    def crop(video, seconds, region, output, fps=None):
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (region[2], region[3]), (90, 90, 90)).save(output)
        return output
    monkeypatch.setattr(server, "frame_image", frame)
    monkeypatch.setattr(media, "crop_at", crop)


def encode_by_class(per_class, max_boxes=4):
    from conftest import encode_by_class as encode
    return encode(per_class, max_boxes)


# NMS output layouts and coordinate mapping

def test_nms_by_class_list_raw_buffer_and_batch_wrapper_parse_alike():
    np = pytest.importorskip("numpy")
    per_class = [[[0.1, 0.2, 0.3, 0.4, 0.9], [0.5, 0.5, 0.6, 0.6, 0.5]], [[0.2, 0.1, 0.4, 0.3, 0.8]]]
    hailo_format = [np.asarray(b, dtype=np.float32) for b in per_class]
    raw = encode_by_class(per_class)  # padded to max_boxes, as HailoRT allocates it
    for layout in (hailo_format, raw, [hailo_format], per_class):
        parsed = parse_nms(layout)
        assert [len(b) for b in parsed] == [2, 1]
        np.testing.assert_allclose(parsed[0], hailo_format[0], atol=1e-7)
        np.testing.assert_allclose(parsed[1], hailo_format[1], atol=1e-7)


def test_nms_by_score_objects_and_packed_bytes_parse_alike():
    np = pytest.importorskip("numpy")
    detections = [SimpleNamespace(y_min=.2, x_min=.1, y_max=.4, x_max=.3, score=.8, class_id=1),
                  SimpleNamespace(y_min=.1, x_min=.2, y_max=.3, x_max=.4, score=.7, class_id=0)]
    record = np.dtype([("y_min", "<f4"), ("x_min", "<f4"), ("y_max", "<f4"), ("x_max", "<f4"), ("score", "<f4"), ("class_id", "<u2")])
    assert record.itemsize == 22  # hailo_detection_t is packed (hailort.h, #pragma pack(1))
    rows = np.array([(d.y_min, d.x_min, d.y_max, d.x_max, d.score, d.class_id) for d in detections], dtype=record)
    raw = np.frombuffer(len(detections).to_bytes(2, "little") + rows.tobytes() + bytes(40), dtype=np.uint8)
    for layout in (detections, raw):
        person, forklift = parse_nms(layout)
        np.testing.assert_allclose(person, [[.1, .2, .3, .4, .7]], atol=1e-6)
        np.testing.assert_allclose(forklift, [[.2, .1, .4, .3, .8]], atol=1e-6)
    assert [len(b) for b in parse_nms([])] == [0, 0]  # by-score frame without detections


@pytest.mark.parametrize("layout, message", [
    ([[[0.1, 0.2, 0.3, 0.4, 0.9]]], "Expected 2"),
    ([[[0.1, 0.2, 0.3, 0.4]], []], "box length"),
    ([SimpleNamespace(y_min=0, x_min=0, y_max=1, x_max=1, score=.9, class_id=5)], "class_id"),
])
def test_malformed_nms_output_is_rejected(layout, message):
    pytest.importorskip("numpy")
    with pytest.raises(ValueError, match=message):
        parse_nms(layout)


def test_truncated_raw_nms_buffer_is_rejected():
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="shorter"):
        parse_nms(np.array([3, 0.1, 0.2, 0.3, 0.4, 0.9], dtype=np.float32))


def test_letterbox_matches_the_onnx_preprocessing():
    np = pytest.importorskip("numpy"); pytest.importorskip("cv2")
    from tracebrief.compact_vision import preprocess
    from tracebrief.hailo_backend import letterbox
    image = np.random.default_rng(3).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    canvas, ratio = letterbox(image)
    tensor, onnx_ratio = preprocess(image, 416)
    assert canvas.dtype == np.uint8 and canvas.shape == (416, 416, 3) and ratio == onnx_ratio
    assert np.array_equal(canvas, tensor[0].transpose(1, 2, 0).astype(np.uint8))


def test_vlm_frame_is_a_writeable_rgb_array_in_the_model_shape():
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("numpy")
    from tracebrief.hailo_backend import vlm_frame
    frame = vlm_frame(Image.new("L", (90, 40)), (336, 336, 3))
    assert frame.shape == (336, 336, 3) and str(frame.dtype) == "uint8"
    assert frame.flags.writeable and frame.flags.c_contiguous  # VLM.generate rejects read-only arrays on the device


@pytest.mark.parametrize("width, height", [(1920, 1080), (320, 240), (720, 1280)])
def test_device_boxes_map_back_through_the_letterbox(width, height):
    person, forklift = (0.31 * width, 0.2 * height, 0.38 * width, 0.9 * height), (0.4 * width, 0.3 * height, 0.99 * width, height)
    # A box wholly in the letterbox fill (below a landscape frame, right of a portrait one) maps outside and is dropped.
    padding = [0.95, 0.1, 0.99, 0.3, 0.95] if width > height else [0.1, 0.95, 0.3, 0.99, 0.95]
    boxes = to_source_boxes([[normalized(person, .9, width, height), padding], [normalized(forklift, .6, width, height)]], width, height)
    assert [b["class_name"] for b in boxes] == ["person", "forklift"]
    for found, expected in zip(boxes, (person, forklift)):
        assert found["bbox_xyxy"] == pytest.approx([round(v, 2) for v in expected], abs=0.011)


def test_boxes_below_the_score_threshold_are_dropped():
    assert to_source_boxes([[normalized(PERSON, .39)], []], 320, 240) == []


@pytest.mark.skipif(not FIXTURE.is_file(), reason="record on a UGen300 with scripts/hailo/bench_ugen300.py --record-fixture tests/fixtures/hailo/nms_by_class.npz")
def test_recorded_device_buffer_parses_like_hailort():
    check_fixture(FIXTURE)


def check_fixture(path):
    """Our raw-buffer parser must agree with HailoRT's own get_buffer() conversion of the same frame."""
    np = pytest.importorskip("numpy")
    data = np.load(path)
    rows = np.concatenate([np.column_stack([np.asarray(b, dtype=float).reshape(-1, 5), np.full(len(b), c)])
                           for c, b in enumerate(parse_nms(data["raw"]))])
    np.testing.assert_allclose(rows, data["converted"], atol=1e-6)


# One VDevice, created once, shared, released

def test_runtime_opens_one_vdevice_shared_by_detector_and_vlm(fake_hailo, manifest, tmp_path):
    np = pytest.importorskip("numpy")
    fake = fake_hailo(queue=4)
    runtime = HailoRuntime(manifest, tmp_path, confidence=.4, nms_iou=.45, batch_size=8)
    assert len(fake.vdevices) == 1
    vdevice = fake.vdevices[0]
    assert (vdevice.params.scheduling_algorithm, vdevice.params.group_id) == ("ROUND_ROBIN", "SHARED")
    assert [m.vdevice for m in fake.models] == [vdevice] and [v.vdevice for v in fake.vlms] == [vdevice]
    model = fake.models[0]
    # Formats and NMS limits are set before configure().
    assert model.formats_at_configure == ("UINT8", "NHWC", "FLOAT32", "HAILO_NMS_BY_CLASS",
                                          {"score_threshold": .4, "iou_threshold": .45, "max_proposals_per_class": 100})
    assert model.batch_size == 8 and runtime.detector.group == 4  # capped by the async queue size
    configured = model.configured
    buffers = [id(b.output().buffer) for b in configured.bindings]
    assert len(buffers) == 8  # two groups of four, allocated at load
    frame = np.zeros((416, 416, 3), np.uint8)
    runtime.detector.detect_batch([(frame, 640, 480)] * 10)
    runtime.detector.detect(frame, 640, 480)
    assert [id(b.output().buffer) for b in configured.bindings] == buffers  # reused, never reallocated
    assert configured.submissions == [4, 4, 2, 1]
    runtime.close()
    assert fake.vlms[0].released and configured.shut and vdevice.released
    assert fake.vlm_release_order == [False]  # VLM released while the VDevice was still open
    runtime.close()


def test_detector_batches_return_results_in_frame_order(fake_hailo, manifest, tmp_path):
    np = pytest.importorskip("numpy")
    fake = fake_hailo(queue=2)
    fake.boxes = lambda frame: [[normalized(PERSON, float(frame[0, 0, 0]) / 100)], []]
    runtime = HailoRuntime(manifest, tmp_path, batch_size=2)
    frames = [(np.full((416, 416, 3), score, np.uint8), 320, 240) for score in range(41, 48)]
    results = runtime.detector.detect_batch(frames)
    assert [r[0]["confidence"] for r in results] == [s / 100 for s in range(41, 48)]
    runtime.close()


def test_detector_close_waits_for_the_device_to_settle_after_the_last_job(fake_hailo, tmp_path, monkeypatch):
    """UGen300 (USB, HailoRT 5.3.2): shutdown() right after batched jobs can time out and reset the device."""
    np = pytest.importorskip("numpy")
    import time
    from tracebrief import hailo_backend as hb
    fake = fake_hailo(queue=8)
    hef = tmp_path / "detector.hef"; hef.write_bytes(b"hef")
    events = []
    frames = [(np.zeros((416, 416, 3), np.uint8), 320, 240)] * 16

    def detector(settle_s=None):
        det = hb.HailoDetector(hb.shared_vdevice(fake), hef, batch_size=8, settle_s=settle_s)
        shutdown = det.configured.shutdown
        det.configured.shutdown = lambda: (events.append(time.perf_counter()), shutdown())
        return det

    monkeypatch.setattr(hb, "SETTLE_S", 1.0)
    assert detector().settle_s == 1.0  # the default applies to the device path
    det = detector(settle_s=0.3)
    det.detect_batch(frames)
    last = det.last_job
    det.close()
    assert events[-1] - last >= 0.3 and det.configured is None
    det = detector(settle_s=0.3)
    det.detect_batch(frames)
    time.sleep(0.35)
    tick = time.monotonic(); det.close()
    assert time.monotonic() - tick < 0.2  # already settled: no extra wait
    det = detector(settle_s=0.3)
    tick = time.monotonic(); det.close()
    assert time.monotonic() - tick < 0.2  # nothing ran: shut down at once


def test_server_loads_the_runtime_once_and_releases_it(fake_hailo, manifest, tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    fake = fake_hailo()
    server = hailo_server(monkeypatch, tmp_path, manifest)
    assert server.device_runtime() is server.device_runtime()
    assert len(fake.vdevices) == 1 and len(fake.models) == 1 and len(fake.vlms) == 1
    server.close_device()
    assert fake.vdevices[0].released and server.DEVICE["runtime"] is None


# No silent fallback

def test_hailo_mode_without_a_device_fails_the_job_and_starts_no_cpu_model(fake_hailo, manifest, tmp_path, monkeypatch):
    fake = fake_hailo(devices=())
    server = hailo_server(monkeypatch, tmp_path, manifest)
    forbid_cpu_models(server, monkeypatch)
    job_id = queued_job(server)
    server.process_job(job_id, tmp_path / "clip.mp4", SOURCE, compact_settings())
    job = server.read_job(job_id)
    assert job["status"] == "failed"
    assert "No Hailo device was found" in job["error"] and "set TRACEBRIEF_BACKEND=cpu" in job["error"]
    assert not fake.vdevices


def test_hailo_mode_without_hailort_fails_the_job_clearly(manifest, tmp_path, monkeypatch):
    server = hailo_server(monkeypatch, tmp_path, manifest)
    monkeypatch.setitem(sys.modules, "hailo_platform", None)  # import hailo_platform raises ImportError
    forbid_cpu_models(server, monkeypatch)
    job_id = queued_job(server)
    server.process_job(job_id, tmp_path / "clip.mp4", SOURCE, compact_settings())
    job = server.read_job(job_id)
    assert job["status"] == "failed" and "hailo_platform" in job["error"]


def test_hef_hash_mismatch_fails_before_any_model_loads(fake_hailo, manifest, tmp_path, monkeypatch):
    fake = fake_hailo()
    (tmp_path / "hefs" / "vlm.hef").write_bytes(b"a different build")
    with pytest.raises(DeviceError, match="vlm HEF SHA-256 mismatch"):
        HailoRuntime(manifest, tmp_path)
    assert not fake.vdevices and not fake.models and not fake.vlms
    server = hailo_server(monkeypatch, tmp_path, manifest)
    forbid_cpu_models(server, monkeypatch)
    job_id = queued_job(server)
    server.process_job(job_id, tmp_path / "clip.mp4", SOURCE, compact_settings())
    job = server.read_job(job_id)
    assert job["status"] == "failed" and "SHA-256 mismatch" in job["error"] and "Refusing to load" in job["error"]


def test_unrecorded_hash_and_other_hailort_release_are_refused(fake_hailo, manifest, tmp_path):
    value = json.loads(manifest.read_text(encoding="utf-8"))
    digest = value["models"]["detector"]["sha256"]
    value["models"]["detector"]["sha256"] = None
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DeviceError, match=f"records no SHA-256.*{digest}"):
        verify_manifest(manifest, tmp_path)
    value["models"]["detector"]["sha256"] = digest
    manifest.write_text(json.dumps(value), encoding="utf-8")
    assert verify_manifest(manifest, tmp_path, "5.3.1")["detector"]["sha256"] == digest
    with pytest.raises(DeviceError, match="for HailoRT 5.3, but HailoRT 5.5.0"):
        verify_manifest(manifest, tmp_path, "5.5.0")
    fake = fake_hailo(version="5.5.0")
    with pytest.raises(DeviceError, match="HailoRT 5.5.0 is installed"):
        HailoRuntime(manifest, tmp_path)
    assert not fake.vdevices


def test_device_fault_mid_job_fails_and_releases_the_runtime(fake_hailo, manifest, tmp_path, monkeypatch):
    fake = fake_hailo()
    fake.fault = RuntimeError("HAILO_TIMEOUT(4)")
    server = hailo_server(monkeypatch, tmp_path, manifest)
    forbid_cpu_models(server, monkeypatch)
    job_id = queued_job(server)
    server.process_job(job_id, synthetic_video(tmp_path / "clip.avi"), SOURCE, compact_settings())
    job = server.read_job(job_id)
    assert job["status"] == "failed" and job["error"] == "UGen300 detection failed: RuntimeError: HAILO_TIMEOUT(4)"
    assert server.DEVICE["runtime"] is None and fake.vdevices[0].released and fake.vlms[0].released


# Backend switch

@pytest.mark.parametrize("value, expected", [(None, "cpu"), ("", "cpu"), ("cpu", "cpu"), (" HAILO ", "hailo")])
def test_backend_switch_defaults_to_cpu(value, expected, monkeypatch):
    from tracebrief import server
    if value is None:
        monkeypatch.delenv("TRACEBRIEF_BACKEND", raising=False)
    else:
        monkeypatch.setenv("TRACEBRIEF_BACKEND", value)
    assert server.backend() == expected


def test_unknown_backend_is_rejected(monkeypatch):
    from tracebrief import server
    monkeypatch.setenv("TRACEBRIEF_BACKEND", "gpu")
    with pytest.raises(ValueError, match="cpu or hailo"):
        server.backend()


def get(server, path):
    handler = object.__new__(server.Handler)
    handler.path = path
    responses = []
    handler.json_response = lambda body, status=200: responses.append((status, body))
    handler.do_GET()
    return responses[0]


def test_health_reports_the_device_only_once_it_is_loaded(fake_hailo, manifest, tmp_path, monkeypatch):
    from tracebrief import server
    monkeypatch.delenv("TRACEBRIEF_BACKEND", raising=False)
    status, body = get(server, "/api/health")
    assert status == 200 and body["backend"] == "cpu" and body["hardware"] == "host-cpu"
    pytest.importorskip("numpy")
    fake_hailo()
    server = hailo_server(monkeypatch, tmp_path, manifest)
    status, body = get(server, "/api/health")
    assert body["backend"] == "hailo" and body["state"] == "not-loaded" and body["device"] is None
    assert body["hailort_version"] == "5.3.0" and "host" not in body["hardware"]
    server.device_runtime()
    status, body = get(server, "/api/health")
    assert body["state"] == "ready" and body["visual_check_available"] and body["local_generation_available"]
    assert body["hardware"] == "HAILO10H (UGen300) · device 0001:01:00.0 · firmware 5.3.0 (release,app) · HailoRT 5.3.0"
    device = body["device"]["devices"][0]
    assert device["architecture"] == "HAILO10H" and device["serial_number"] == "UGEN300-0001"
    assert body["device"]["hefs"]["vlm"]["sha256"] == json.loads(manifest.read_text(encoding="utf-8"))["models"]["vlm"]["sha256"]
    server.close_device()


def test_failed_load_is_reported_by_health(fake_hailo, manifest, tmp_path, monkeypatch):
    fake_hailo(devices=())
    server = hailo_server(monkeypatch, tmp_path, manifest)
    server.warm_device()
    status, body = get(server, "/api/health")
    assert body["state"] == "failed" and "No Hailo device" in body["error"] and body["device"] is None


def test_hailo_mode_offers_only_searches_that_run_on_the_device(fake_hailo, manifest, tmp_path, monkeypatch):
    server = hailo_server(monkeypatch, tmp_path, manifest)
    status, body = get(server, "/api/search-profiles")
    assert [p["id"] for p in body["profiles"]] == ["fast"] and body["default"] == "fast"
    handler = object.__new__(server.Handler)
    handler.path = "/api/jobs"
    handler.read_json = lambda: {"sample_id": "warehouse-1", "search_profile": "broader"}
    responses = []
    handler.json_response = lambda body, status=200: responses.append((status, body))
    monkeypatch.setattr(server, "samples", lambda: [{"id": "warehouse-1", "path": "warehouse-1.mp4", "name": "Fixture", "source_label": "Synthetic"}])
    handler.do_POST()
    assert responses[0][0] == 400 and "Fast search" in responses[0][1]["error"]


# A whole job on the device

def test_hailo_job_runs_detection_check_and_draft_on_the_device(fake_hailo, manifest, tmp_path, monkeypatch):
    np = pytest.importorskip("numpy"); Image = pytest.importorskip("PIL.Image")
    from tracebrief import media
    fake = fake_hailo()
    fake.boxes = lambda frame: [[normalized(PERSON, .9)], [normalized(FORKLIFT, .8)]]
    server = hailo_server(monkeypatch, tmp_path, manifest)

    def frame(video, output, seconds, fps=None):
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24)).save(output)

    def crop(video, seconds, region, output, fps=None):
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (region[2], region[3]), (90, 90, 90)).save(output)
        return output
    monkeypatch.setattr(server, "frame_image", frame)
    monkeypatch.setattr(media, "crop_at", crop)
    forbid_cpu_models(server, monkeypatch)
    job_id = queued_job(server)
    server.process_job(job_id, synthetic_video(tmp_path / "clip.avi"), SOURCE, compact_settings(), {"id": "fast"})
    job = server.read_job(job_id)
    assert job["status"] == "completed", job.get("error")
    result = job["result"]
    hashes = {name: entry["sha256"] for name, entry in json.loads(manifest.read_text(encoding="utf-8"))["models"].items()}
    assert result["backend"] == "hailo" and "device 0001:01:00.0" in result["hardware_label"]
    assert result["device"]["hefs"]["detector"]["sha256"] == hashes["detector"] == result["vision_model"]["sha256"]
    assert result["vision_model"]["backend"] == "YOLOX HEF on Hailo-10H (HailoRT 5.3.0)"
    event = result["events"][0]
    assert event["vision_record"]["evidence"][0]["pairs"][0]["person_bbox"] == [100.0, 50.0, 140.0, 200.0]
    check = event["visual_check"]
    assert check["status"] == "confirmed" and check["runtime"] == "hailo" and check["model_sha256"] == hashes["vlm"]
    assert check["prompt_version"] == "look-v2"
    assert event["generation_method"] == "ugen300-qwen2-vl-draft" and event["title"] == "Person beside forklift in aisle"
    vlm = fake.vlms[0]
    look_call, draft_call = vlm.calls
    assert look_call["prompt"][1]["content"][1]["text"] == look_prompt.PROMPT
    assert look_call["frames"][0].shape == (336, 336, 3) and look_call["frames"][0].dtype == np.uint8
    assert draft_call["frames"] == [] and draft_call["do_sample"] is False and draft_call["seed"] == 19
    assert vlm.events == ["clear", "generate", "clear"] * 2  # a fresh context for every item
    assert len(fake.vdevices) == 1 and not fake.vdevices[0].released  # stays resident for the next job
    page = server.export_html(result)
    assert "UGen300 Qwen2-VL-2B" in page and "Host CPU" not in page
    server.close_device()


# Phases: the detector and the VLM take turns on the device

def test_a_hailo_job_detects_the_whole_clip_then_checks_every_candidate_then_drafts(fake_hailo, manifest, tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    fake = fake_hailo(queue=8)
    fake.boxes = pair_when_bright
    server = hailo_server(monkeypatch, tmp_path, manifest)
    host_media(server, monkeypatch)
    forbid_cpu_models(server, monkeypatch)
    job_id = queued_job(server)
    seen, reply = [], fake.reply
    fake.reply = lambda prompt, frames: seen.append(server.read_job(job_id)["phase"]) or reply(prompt, frames)
    server.process_job(job_id, two_moment_video(tmp_path / "clip.avi"), SOURCE, compact_settings(), {"id": "fast"})
    job = server.read_job(job_id)
    assert job["status"] == "completed", job.get("error")
    assert len(job["result"]["events"]) == 2
    # All ten sampled frames in batches of eight, then both visual checks, then both drafts; nothing interleaved.
    assert fake.timeline == [("detect", 8), ("detect", 2), ("look",), ("look",), ("draft",), ("draft",)] and fake.overlaps == []
    assert seen == ["visual_check", "visual_check", "draft", "draft"]  # the job record names the running phase
    phases = job["phases"]
    assert job["phase"] == "done" and list(phases) == list(server.PHASES)
    assert phases["detection"]["sampled_frames"] == 10 and phases["detection"]["candidates"] == 2
    assert phases["visual_check"]["candidates"] == phases["draft"]["candidates"] == 2
    assert [phases[name]["model_switches"] for name in server.PHASES] == [0, 1, 0]  # one switch, detector to VLM
    assert all(phases[name]["seconds"] > 0 and phases[name]["finished_at"] for name in server.PHASES)
    schedule = job["result"]["schedule"]
    assert schedule["mode"] == "phased" and schedule["phases"] == phases
    assert schedule["total_s"] == pytest.approx(sum(phases[name]["seconds"] for name in server.PHASES), abs=0.002)
    server.close_device()


def test_the_detector_and_the_vlm_never_run_at_once_even_when_two_jobs_overlap(fake_hailo, manifest, tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    fake = fake_hailo(queue=8)
    fake.boxes, fake.delay = pair_when_bright, 0.003
    server = hailo_server(monkeypatch, tmp_path, manifest)
    host_media(server, monkeypatch)
    forbid_cpu_models(server, monkeypatch)
    runtime = server.device_runtime()
    assert runtime.detector.turns is runtime.vlm.turns is runtime.turns
    # Two clips analysed at once (two server threads): each job's detector, visual-check and draft calls take turns.
    first_job, second_job = queued_job(server), queued_job(server, "bbbbbbbbbbbbbbbb")
    jobs = [threading.Thread(target=server.process_job, args=(job_id, two_moment_video(tmp_path / f"{job_id}.avi"), SOURCE, compact_settings()))
            for job_id in (first_job, second_job)]
    for thread in jobs:
        thread.start()
    for thread in jobs:
        thread.join(60)
    assert all(server.read_job(job_id)["status"] == "completed" for job_id in (first_job, second_job))
    assert fake.overlaps == []
    kinds = [entry[0] for entry in fake.timeline]
    assert kinds.count("look") == 4 and kinds.count("draft") == 4 and "detect" in kinds
    server.close_device()


def test_the_fake_device_sees_an_overlap_when_nothing_makes_the_models_take_turns(fake_hailo, tmp_path):
    """The check above means something: a detector and a VLM with separate turns (as the benchmark builds them) overlap."""
    np = pytest.importorskip("numpy"); Image = pytest.importorskip("PIL.Image")
    from tracebrief import hailo_backend as hb
    fake = fake_hailo(queue=8)
    fake.delay = 0.002
    hef = tmp_path / "model.hef"; hef.write_bytes(b"hef")
    frames = [(np.zeros((416, 416, 3), np.uint8), 320, 240)] * 16

    def run(detector, vlm):
        worker = threading.Thread(target=lambda: vlm.check(Image.new("RGB", (64, 48))))
        worker.start()
        while worker.is_alive():
            detector.detect_batch(frames)
        worker.join()
    vdevice = hb.shared_vdevice(fake)
    run(hb.HailoDetector(vdevice, hef, batch_size=8), hb.HailoVLM(vdevice, hef))
    assert fake.overlaps
    fake.overlaps.clear()
    turns = hb.Turns()
    run(hb.HailoDetector(vdevice, hef, batch_size=8, turns=turns), hb.HailoVLM(vdevice, hef, turns=turns))
    assert fake.overlaps == [] and turns.switches >= 1


def test_a_failed_detector_batch_waits_for_its_submitted_jobs_before_the_vlm_may_run(fake_hailo, tmp_path):
    np = pytest.importorskip("numpy")
    from tracebrief import hailo_backend as hb
    fake = fake_hailo(queue=2)
    hef = tmp_path / "detector.hef"; hef.write_bytes(b"hef")
    detector = hb.HailoDetector(hb.shared_vdevice(fake), hef, batch_size=2)
    submit, calls = detector.configured.run_async, []

    def run_async(bindings):
        calls.append(len(bindings))
        if len(calls) == 2:
            raise RuntimeError("HAILO_TIMEOUT(4)")
        return submit(bindings)
    detector.configured.run_async = run_async
    with pytest.raises(RuntimeError, match="HAILO_TIMEOUT"):
        detector.detect_batch([(np.zeros((416, 416, 3), np.uint8), 320, 240)] * 6)
    assert calls == [2, 2] and fake.in_flight == 0  # the first job was collected before the turn was released
    assert not detector.turns.lock.locked()


def test_a_failed_phase_keeps_the_time_it_took(fake_hailo, manifest, tmp_path, monkeypatch):
    server = hailo_server(monkeypatch, tmp_path, manifest)
    job_id = queued_job(server)
    phases = server.Phases(job_id, None)
    with pytest.raises(OSError):
        with phases.run("visual_check"):
            time.sleep(0.02)
            raise OSError("the box lost power")
    saved = server.read_job(job_id)["phases"]["visual_check"]
    assert saved["seconds"] >= 0.02 and "finished_at" not in saved
    resumed = server.Phases(job_id, None, server.read_job(job_id)["phases"])  # the saved entries, added to
    with resumed.run("visual_check"):
        pass
    assert resumed.seconds("visual_check") >= 0.02 and resumed.done["visual_check"]["finished_at"]


def test_a_device_held_by_another_process_fails_at_once_with_the_reason(fake_hailo, manifest, tmp_path, monkeypatch):
    from tracebrief import hailo_backend as hb
    fake = fake_hailo(devices=("usb/001:024",))
    fake.held = True
    monkeypatch.setattr(hb, "sha256", lambda path: pytest.fail("a HEF was hashed although the device is held"))
    tick = time.monotonic()
    with pytest.raises(DeviceError, match="already open in another process") as caught:
        HailoRuntime(manifest, tmp_path)
    assert time.monotonic() - tick < 1 and not fake.vdevices
    message = str(caught.value)
    assert "UGen300 (usb/001:024)" in message and "HAILO_LIBUSB_FAILURE (98)" in message and "LIBUSB_ERROR_ACCESS" in message
    assert "bench_ugen300.py" in message and "reconnect" in message
    with pytest.raises(DeviceError, match="already open in another process"):  # taken between identify and the VDevice
        hb.shared_vdevice(fake)
    assert hb.held_elsewhere(fake.HailoRTStatusException("98")) and not hb.held_elsewhere(RuntimeError("HAILO_TIMEOUT(4)"))
    server = hailo_server(monkeypatch, tmp_path, manifest)
    forbid_cpu_models(server, monkeypatch)
    job_id = queued_job(server)
    server.process_job(job_id, tmp_path / "clip.mp4", SOURCE, compact_settings())
    job = server.read_job(job_id)
    assert job["status"] == "failed" and "already open in another process" in job["error"]
    assert "set TRACEBRIEF_BACKEND=cpu" in job["error"]


def test_cpu_results_are_labeled_as_cpu(tmp_path, monkeypatch):
    """The CPU path's label is unchanged and never names a device."""
    from tracebrief import server
    monkeypatch.delenv("TRACEBRIEF_BACKEND", raising=False)
    monkeypatch.setattr(server, "RUNTIME", tmp_path)

    class Process:
        pid, returncode = 1, 0

        def __init__(self, command, **kwargs):
            server.write_json(Path(command[3]), {"video": {"duration_s": 1}, "model": {"id": "fixture"}, "settings": {}, "events": []})

        def poll(self):
            return 0
    monkeypatch.setattr(server.subprocess, "Popen", Process)
    video = tmp_path / "clip.mp4"; video.write_bytes(b"fixture")
    job_id = queued_job(server)
    server.process_job(job_id, video, SOURCE, compact_settings())
    result = server.read_job(job_id)["result"]
    assert result["backend"] == "cpu" and result["hardware_label"] == "Host CPU · Stage I prototype" and "device" not in result


# Benchmark script

def load_bench():
    spec = importlib.util.spec_from_file_location("bench_ugen300", ROOT / "scripts" / "hailo" / "bench_ugen300.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bench_fails_fast_without_a_device(fake_hailo, manifest, tmp_path):
    fake = fake_hailo(devices=())
    output = tmp_path / "bench.json"
    with pytest.raises(SystemExit, match="bench_ugen300: No Hailo device was found"):
        load_bench().main(["--manifest", str(manifest), "--output", str(output), "--video", str(tmp_path / "absent.mp4")])
    assert not output.exists() and not fake.vdevices


def test_bench_fails_fast_without_hailort(manifest, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "hailo_platform", None)
    with pytest.raises(SystemExit, match="hailo_platform"):
        load_bench().main(["--manifest", str(manifest), "--output", str(tmp_path / "bench.json")])


def test_bench_records_every_measurement_on_the_fake_device(fake_hailo, manifest, tmp_path, capsys):
    pytest.importorskip("PIL")
    fake = fake_hailo(queue=8)
    fake.boxes = lambda frame: [[normalized(PERSON, .9)], [normalized(FORKLIFT, .8)]]
    output, fixture = tmp_path / "bench.json", tmp_path / "nms_by_class.npz"
    load_bench().main(["--manifest", str(manifest), "--video", str(synthetic_video(tmp_path / "clip.avi")), "--frames", "16",
                       "--output", str(output), "--record-fixture", str(fixture), "--skip-process"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["device"]["devices"][0]["id"] == "0001:01:00.0" and report["hailort_version"] == "5.3.0"
    assert set(report["hefs"]) == {"detector", "vlm"} and all(len(h["sha256"]) == 64 for h in report["hefs"].values())
    assert [run["batch_size"] for run in report["detector"]] == [1, 8] and all(run["fps"] > 0 for run in report["detector"])
    assert report["vlm"]["visual_check"]["parsed"] and report["vlm"]["draft"]["tokens"] > 1
    assert report["concurrent"]["same_process"]["detector"]["frames"] >= 0 and "second_process" not in report["concurrent"]
    assert set(report["load_s"]) == {"detector", "vlm"}
    assert len(fake.vdevices) == 1 and fake.vdevices[0].released
    assert "Detector batch 8" in capsys.readouterr().out
    check_fixture(fixture)


class FrozenMonotonic:
    """The clocks as seen on a coarse host: monotonic() never advances within a short step, perf_counter() does.

    On the Windows device host (Python 3.10) time.monotonic() is GetTickCount64 with a 15.6 ms tick, longer than one
    detector frame, so every duration that is reported has to come from perf_counter().
    """

    def __init__(self):
        import time
        self.perf_counter, self.sleep = time.perf_counter, time.sleep

    @staticmethod
    def monotonic():
        return 1000.0


def test_bench_reports_a_second_process_that_cannot_open_the_device(fake_hailo, manifest, tmp_path, monkeypatch, capsys):
    """Windows over USB: the second process's VDevice fails (LIBUSB_ERROR_ACCESS); the report says so and stands."""
    pytest.importorskip("PIL")
    from tracebrief import hailo_backend as hb
    fake = fake_hailo(devices=("usb/001:024",), queue=8)
    fake.boxes = lambda frame: [[normalized(PERSON, .9)], [normalized(FORKLIFT, .8)]]
    bench = load_bench()

    def refused(*args):  # what second_process raises when the child sends back its DeviceError
        raise RuntimeError(f"DeviceError: {hb.held_error('usb/001:024')}")
    monkeypatch.setattr(bench, "second_process", refused)
    output = tmp_path / "bench.json"
    arguments = ["--manifest", str(manifest), "--video", str(synthetic_video(tmp_path / "clip.avi")), "--frames", "16", "--output", str(output)]
    bench.main(arguments)
    second = json.loads(output.read_text(encoding="utf-8"))["concurrent"]["second_process"]
    assert second["measured"] is False and second["device_held_by_this_process"] is True
    assert "already open in another process" in second["reason"] and "LIBUSB_ERROR_ACCESS" in second["note"]
    assert "not measurable: the device admits one process (expected on Windows over USB)" in capsys.readouterr().out
    monkeypatch.setattr(bench, "second_process", lambda *args: (_ for _ in ()).throw(RuntimeError("the second process exited (code 1)")))
    bench.main(arguments)
    second = json.loads(output.read_text(encoding="utf-8"))["concurrent"]["second_process"]
    assert second == {"measured": False, "reason": "the second process exited (code 1)", "device_held_by_this_process": False, "note": None}
    fake.held = True  # another process holds the device before the bench starts
    output.unlink()
    with pytest.raises(SystemExit, match="bench_ugen300: The UGen300 \\(usb/001:024\\) is already open in another process"):
        bench.main(arguments)
    assert not output.exists()


def test_bench_durations_use_the_high_resolution_clock(fake_hailo, manifest, tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    fake = fake_hailo(queue=8)
    fake.boxes = lambda frame: [[normalized(PERSON, .9)], [normalized(FORKLIFT, .8)]]
    bench = load_bench()
    monkeypatch.setattr(bench, "time", FrozenMonotonic())
    output = tmp_path / "bench.json"
    bench.main(["--manifest", str(manifest), "--video", str(synthetic_video(tmp_path / "clip.avi")), "--frames", "16",
                "--output", str(output), "--skip-process"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert all(run["fps"] > 0 and run["latency_per_submission"]["p50_ms"] > 0 for run in report["detector"])


def test_vlm_and_detector_timings_use_the_high_resolution_clock(fake_hailo, manifest, tmp_path, monkeypatch):
    np = pytest.importorskip("numpy"); Image = pytest.importorskip("PIL.Image")
    from tracebrief import device_pipeline, hailo_backend
    fake_hailo(queue=4)
    monkeypatch.setattr(hailo_backend, "time", FrozenMonotonic())
    monkeypatch.setattr(device_pipeline, "time", FrozenMonotonic())
    runtime = HailoRuntime(manifest, tmp_path, confidence=.25, nms_iou=.45)
    try:
        assert runtime.vlm.check(Image.new("RGB", (64, 48)))["tokens_per_s"] is not None
        settings = {**compact_settings(), "output_dir": str(tmp_path / "frames")}
        result = device_pipeline.analyze_video(runtime, synthetic_video(tmp_path / "clip.avi"), settings)
        assert result["timing"]["inference_s"] > 0 and result["timing"]["wall_s"] > 0
    finally:
        runtime.close()


# A real device (skipped unless one is present)

@pytest.mark.ugen300
def test_real_device_runs_both_models():
    cv2 = pytest.importorskip("cv2"); Image = pytest.importorskip("PIL.Image")
    from tracebrief.hailo_backend import CLASSES, letterbox
    manifest = Path(os.environ.get("TRACEBRIEF_HAILO_MANIFEST", ROOT / "config" / "hailo-manifest.json"))
    runtime = HailoRuntime(manifest, ROOT, batch_size=8)
    try:
        cap = cv2.VideoCapture(str(ROOT / "data" / "samples" / "warehouse-1.mp4"))
        ok, image = cap.read(); cap.release()
        assert ok
        detections = runtime.detector.detect(letterbox(image)[0], image.shape[1], image.shape[0])
        assert all(d["class_name"] in CLASSES and 0 <= d["bbox_xyxy"][0] < d["bbox_xyxy"][2] <= image.shape[1] for d in detections)
        result = runtime.vlm.check(Image.fromarray(image[:, :, ::-1]))
        assert "answer" in result, result.get("raw_text")
        assert runtime.identity["devices"][0].get("architecture") == "HAILO10H"
    finally:
        runtime.close()
