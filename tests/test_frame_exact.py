"""Evidence frames and visual-check crops show the frame the detector analysed, on every path.

Numbered clips (tests/conftest.py) spell each frame's number in its pixels. A single-clip detector
(compact_vision, vision, device_pipeline) analyses the n-th frame OpenCV's sequential read returns
and records it at round(n / fps, 3). The fragmented layout is the SDG-Warehouse one: one fragment
per frame, every frame flagged as a sync sample, where the frozen look.crop_frame (-ss) returned a
later frame than the detector's (docs/ERRATA.md).
"""
import collections
from pathlib import Path

import pytest

from conftest import frame_number, has_encoder, make_numbered_video, needs_ffmpeg
from tracebrief import look, media

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 160, 32
PERSON = [0.0, 0.0, 60.0, 20.0]  # with the forklift box, a crop region of the whole frame width
LAYOUTS = {"mp4": {}, "fragmented": {"layout": "fragmented"}, "vfr": {"drop_after": 90}, "dropped-frame": {"drop": (100,)}}


def codec():
    """The SDG files are H.264; ffmpeg's own MPEG-4 encoder stands in where libx264 is missing."""
    return "libx264" if has_encoder("libx264") else "mpeg4"


def detector_frames(video):
    """What the detector sees: the frame number of every frame of OpenCV's sequential read, and its fps."""
    cv2 = pytest.importorskip("cv2")
    capture, numbers = cv2.VideoCapture(str(video)), []
    fps = capture.get(cv2.CAP_PROP_FPS)
    while True:
        ok, image = capture.read()
        if not ok:
            break
        numbers.append(frame_number(image))
    capture.release()
    return numbers, fps


@pytest.fixture(autouse=True)
def fresh_streams(monkeypatch):
    monkeypatch.setattr(media, "STREAMS", {})


@needs_ffmpeg
def test_the_frozen_seek_shows_a_later_frame_on_the_sdg_layout(tmp_path):
    """Why the app no longer uses look.crop_frame: frame 151 is a P-frame, every frame claims to be a keyframe."""
    pytest.importorskip("PIL")
    video = make_numbered_video(tmp_path / "clip.mp4", layout="fragmented", codec=codec())
    crop = look.crop_frame(video, round(151 / 30, 3), [0, 0, WIDTH, HEIGHT], tmp_path / "crop.jpg")
    shown = frame_number(crop)
    assert shown != 151
    if codec() == "libx264":
        assert shown == 180  # the next real keyframe, as with the SDG clips (frame 215 gave 240)


@needs_ffmpeg
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("path", ["count", "seek", "seek-checked"])
def test_frame_at_returns_the_frame_the_detector_analysed(layout, path, tmp_path, monkeypatch):
    """Counting from the start, a checked seek, and a seek that cannot be used and falls back to counting."""
    pytest.importorskip("PIL")
    video = make_numbered_video(tmp_path / "clip.mp4", codec=codec(), **LAYOUTS[layout])
    numbers, fps = detector_frames(video)
    if path != "count":
        monkeypatch.setattr(media, "FULL_DECODE_S", 0.0)
        monkeypatch.setattr(media, "SEEK_BACK_S", 2.0 if path == "seek" else 0.05)
    seeks, real_seek = collections.Counter(), media._seek

    def seek(*args):
        ok = real_seek(*args)
        seeks.update([ok])
        return ok
    monkeypatch.setattr(media, "_seek", seek)
    for n in sorted({n for n in (0, 1, 14, 29, 151, 179, 181, 215) if n < len(numbers)} | {len(numbers) - 1}):
        image = media.frame_at(video, round(n / fps, 3), tmp_path / "frame.jpg", fps)
        assert frame_number(image) == numbers[n], f"frame {n}"
    if path == "count":
        assert not seeks
    elif path == "seek":
        assert seeks[True] and (layout == "fragmented" or not seeks[False])
    elif layout == "fragmented":
        assert seeks[False]  # decoding began after the target, or on a damaged frame: counted instead


def test_the_seek_check_reads_exact_timestamps_from_showinfo():
    """Integer pts in the time base showinfo reports: no rounding, no exponent notation, no frame without a timestamp."""
    from fractions import Fraction
    log = ("[Parsed_showinfo_0 @ 0x1] config in time_base: 1/90000, frame_rate: 25/1\n"
           "[Parsed_showinfo_0 @ 0x1] n:   0 pts:      1 pts_time:1.11111e-05 duration: 3600 fmt:yuv420p iskey:1 type:I\n"
           "[Parsed_showinfo_0 @ 0x1] n:   1 pts:NOPTS pts_time:NOPTS duration: 3600 fmt:yuv420p iskey:0 type:P\n"
           "[Parsed_showinfo_2 @ 0x2] config in time_base: 1/90000, frame_rate: 25/1\n"
           "[Parsed_showinfo_2 @ 0x2] n:   0 pts:3600001 pts_time:40.0000 duration: 3600 fmt:yuv420p iskey:0 type:P\n")
    assert media._frames_seen(log) == {0: [(Fraction(1, 90000), True)], 2: [(Fraction(3600001, 90000), False)]}


@needs_ffmpeg
def test_crop_at_cuts_the_frozen_region_and_size_from_the_detector_frame(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    video = make_numbered_video(tmp_path / "clip.mp4", layout="fragmented", codec=codec())
    crop = media.crop_at(video, round(215 / 30, 3), [0, 0, WIDTH, HEIGHT], tmp_path / "crop.jpg", 30.0)
    assert frame_number(crop) == 215 and Image.open(crop).size == (WIDTH, HEIGHT)


@needs_ffmpeg
def test_frame_at_refuses_a_frame_past_the_end(tmp_path):
    video = make_numbered_video(tmp_path / "clip.mp4", count=40, codec="mpeg4")
    with pytest.raises(FileNotFoundError, match="no frame 40"):
        media.frame_at(video, 40 / 30, tmp_path / "frame.jpg", 30.0)
    assert not (tmp_path / "frame.jpg").exists()


@pytest.mark.parametrize("fps", [25.0, 29.97, 30.0, 59.94, 60.0])
def test_a_recorded_time_maps_back_to_its_frame_number(fps):
    """compact_vision records round(n / fps, 3); frame_index undoes it for any frame of a 10-hour file."""
    assert all(media.frame_index(round(n / fps, 3), fps) == n for n in range(0, int(10 * 3600 * fps), 997))


# The three paths of the app

def vision_record(video, fps, frames, peak, scores=(0.9, 0.8)):
    """compact_vision.analyze_video's record for one candidate on the given frames (closest pair at peak)."""
    observations, evidence = [], []
    for n in frames:
        forklift = [60.0 + abs(n - peak) / 4, 0.0, 160.0, 20.0]
        detections = [{"class_name": "person", "confidence": scores[0], "bbox_xyxy": PERSON},
                      {"class_name": "forklift", "confidence": scores[1], "bbox_xyxy": forklift}]
        gap = forklift[0] - PERSON[2]
        pairs = [{"person_bbox": PERSON, "forklift_bbox": forklift, "gap_px": gap, "gap_person_height_ratio": round(gap / 20, 4),
                  "confidence": min(scores)}]
        observations.append({"frame_index": n, "time_s": round(n / fps, 3), "detections": detections, "proximity_pairs": pairs})
        evidence.append({"frame_index": n, "time_s": round(n / fps, 3), "image_path": "", "pairs": pairs,
                         "fact": "Predicted person and forklift boxes are close in image space."})
    times = [round(n / fps, 3) for n in frames]
    return {"video": {"path": str(video), "duration_s": 10.0, "fps": fps, "width": WIDTH, "height": HEIGHT},
            "model": {"id": "fixture"}, "settings": {"backend": "compact-onnx"}, "observations": observations,
            "events": [{"id": "candidate-001", "start_s": times[0] - 1, "end_s": times[-1] + 1, "peak_s": round(peak / fps, 3),
                        "evidence": evidence, "limitations": []}]}


def job_frames(server, job_id, event):
    folder = server.job_path(job_id).parent / "frames"
    return [frame_number(folder / Path(o["original_frame_url"]).name) for o in event["observations"]]


@needs_ffmpeg
def test_a_cpu_job_shows_the_frames_the_detector_analysed(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from tracebrief import cpu_pipeline, server
    monkeypatch.delenv("TRACEBRIEF_BACKEND", raising=False)
    monkeypatch.setattr(server, "RUNTIME", tmp_path / "runtime")
    model = tmp_path / "vlm.gguf"; model.write_bytes(b"x")
    monkeypatch.setenv("TRACEBRIEF_VLM_MODEL", str(model)); monkeypatch.setenv("TRACEBRIEF_VLM_PROJECTOR", str(model))
    video = make_numbered_video(tmp_path / "clip.mp4", layout="fragmented", codec=codec())
    numbers, fps = detector_frames(video)
    raw = vision_record(video, fps, [151, 181, 215, 241, 269], peak=215)

    checked = []

    def check_items(items, root, output_dir, name, threads=4):
        checked.extend(items)
        return {"seconds": 0.1, "model_file": "vlm.gguf", "model_sha256": "0" * 64, "prompt_version": "look-v2",
                "results": [{"raw_text": "{}", "answer": {"forklift": "yes", "person": "yes", "description": "A worker beside a forklift."}}]}
    monkeypatch.setattr(server, "run_vision_process", lambda *args: raw)  # the vision worker's detector record
    monkeypatch.setattr(cpu_pipeline, "check_items", check_items)
    monkeypatch.setattr(server, "draft_candidate", lambda *args: None)
    job_id = "abcdefabcdefabcd"
    server.write_json(server.job_path(job_id), {"id": job_id, "status": "queued"})
    server.process_job(job_id, video, {"name": "Fixture", "source_label": "Synthetic fixture", "video_url": "/media/x"},
                       {"backend": "compact-onnx"})
    job = server.read_job(job_id)
    assert job["status"] == "completed", job.get("error")
    event = job["result"]["events"][0]
    assert job_frames(server, job_id, event) == [151, 215, 269]  # first, peak and last evidence frame
    assert frame_number(checked[0]["image"]) == 215 and event["visual_check"]["time"] == round(215 / fps, 3)


@needs_ffmpeg
def test_a_device_job_shows_the_frames_the_detector_analysed(fake_hailo, manifest, tmp_path, monkeypatch):
    """The UGen300 path: device_pipeline decodes with OpenCV, the fake HEF boxes frames 200-232 (closest at 216)."""
    pytest.importorskip("PIL"); pytest.importorskip("cv2")
    from test_hailo_backend import compact_settings, hailo_server, queued_job, SOURCE
    fake = fake_hailo()
    scale = 416 / WIDTH  # the letterbox's ratio; the frame fills the canvas width

    def boxes(canvas):
        n = frame_number(canvas[:int(HEIGHT * scale), :416])
        if not 200 <= n <= 232:
            return [[], []]
        forklift = [60.0 + abs(n - 216) / 4, 0.0, 160.0, 20.0]
        as_device = lambda box, score: [box[1] * scale / 416, box[0] * scale / 416, box[3] * scale / 416, box[2] * scale / 416, score]
        return [[as_device(PERSON, 0.9)], [as_device(forklift, 0.8)]]
    fake.boxes = boxes
    server = hailo_server(monkeypatch, tmp_path, manifest)
    for name in ("visual_check", "generate"):  # the CPU model steps
        monkeypatch.setattr(server, name, lambda *args, **kwargs: pytest.fail("a CPU model step ran in hailo mode"))
    video = make_numbered_video(tmp_path / "clip.mp4", layout="fragmented", codec=codec())
    job_id = queued_job(server)
    server.process_job(job_id, video, SOURCE, {**compact_settings(), "sample_fps": 7.5}, {"id": "fast"})  # every 4th frame
    job = server.read_job(job_id)
    assert job["status"] == "completed", job.get("error")
    event, = job["result"]["events"]
    assert [e["frame_index"] for e in event["vision_record"]["evidence"]] == list(range(200, 233, 4))
    assert job_frames(server, job_id, event) == [200, 216, 232]
    crop = server.job_path(job_id).parent / "frames" / "event-001-look.jpg"
    assert event["visual_check"]["runtime"] == "hailo" and frame_number(crop) == 216
    server.close_device()
