"""Local TraceBrief server. Only Python's standard library is required for HTTP.

Inference runs in configurable isolated Python environments. Runtime inputs and
review edits are saved locally; no cloud API is called by this application.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from urllib.parse import parse_qs, unquote, urlparse
import uuid

from . import cpu_pipeline, media, workers
from .cpu_pipeline import generate, visual_check
from .hailo_backend import DeviceError
from .look import configured as look_configured

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path(os.environ.get("TRACEBRIEF_RUNTIME", ROOT / ".runtime")).resolve()
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracebrief-analysis")
LOCK = threading.RLock()
MAX_UPLOAD = 500 * 1024 * 1024
BACKENDS = ("cpu", "hailo")
# The UGen300 runtime of this process: one VDevice with the detector and VLM, created once.
DEVICE = {"runtime": None, "error": None}
DEVICE_LOCK = threading.Lock()
# Served types for the files the app ships or stores; the Windows registry can map .js to text/plain,
# which browsers refuse under nosniff.
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska", ".webm": "video/webm"}
# Detection-stage time limit: the Stage I 30 minutes at least, longer for long videos, never more than 12 hours.
VISION_TIMEOUT_FLOOR, VISION_TIMEOUT_CAP = 1800, 12 * 3600
# Measured Stage I CPU seconds per second of video at 1 sampled frame/s, decode included.
VISION_SECONDS_PER_VIDEO_SECOND = {"compact-onnx": 0.08, "grounding-dino": 7.1}
CANDIDATE_STEP_SECONDS = 15  # a visual check or a draft, either path
# The phases of a job on the UGen300, in the order they run (Phases).
PHASES = ("detection", "visual_check", "draft")
def now():
    return datetime.now(timezone.utc).isoformat()


def retried(action, attempts=10):
    """action(), retried on PermissionError: Windows refuses to replace or open a file another thread has open."""
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def load_json(path: Path):
    return retried(lambda: json.loads(Path(path).read_text(encoding="utf-8")))


def write_json(path: Path, value):
    """Write atomically: a reader, such as the UI polling a job, sees the old or the new file, never part of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        retried(lambda: os.replace(tmp, path))
    finally:
        tmp.unlink(missing_ok=True)


def job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{16}", job_id):
        raise ValueError("Invalid review ID")
    return RUNTIME / "jobs" / job_id / "job.json"


def read_job(job_id):
    return load_json(job_path(job_id))


def update_job(job_id, **updates):
    with LOCK:
        path = job_path(job_id)
        job = load_json(path)
        job.update(updates, updated_at=now())
        write_json(path, job)
    return job


def samples():
    manifest = ROOT / "data" / "samples" / "manifest.json"
    return load_json(manifest)["samples"] if manifest.exists() else []


def backend():
    """TRACEBRIEF_BACKEND: "cpu" (default) runs every model on the host; "hailo" runs them on the UGen300."""
    name = os.environ.get("TRACEBRIEF_BACKEND", "cpu").strip().lower() or "cpu"
    if name not in BACKENDS:
        raise ValueError(f"TRACEBRIEF_BACKEND must be cpu or hailo, not {name!r}")
    return name


def device_profile(profile):
    """Only the compact detector is compiled to a HEF; Grounding DINO searches have no device path.

    A compact profile marked "device": false (the Stage I detector) has no HEF either: the detector HEF
    is compiled from the Fast search model only.
    """
    if profile.get("device") is False:
        return False
    settings = load_json(safe_file(ROOT / "config", profile["settings_file"]))
    return isinstance(settings, dict) and settings.get("backend") == "compact-onnx"


def search_profiles():
    profiles = load_json(ROOT / "config" / "search_profiles.json")
    if backend() != "hailo":
        return profiles
    # In hailo mode only searches that run on the device are offered, so none runs on the host instead.
    device = [p for p in profiles["profiles"] if device_profile(p)]
    default = profiles["default"] if any(p["id"] == profiles["default"] for p in device) else (device[0]["id"] if device else None)
    return {**profiles, "default": default, "profiles": device}


def device_runtime():
    """The process's single UGen300 runtime, loaded on first use and kept until shutdown.

    Missing HailoRT, no device or a HEF hash mismatch raises DeviceError; the job fails with that
    message and the next job tries again. There is no CPU fallback in hailo mode.
    """
    with DEVICE_LOCK:
        if DEVICE["runtime"] is None:
            from .hailo_backend import HailoRuntime
            try:
                profiles = search_profiles()["profiles"]
                if not profiles:
                    raise DeviceError("No search approach uses the compact detector, the only detector compiled for the UGen300.")
                settings = load_json(safe_file(ROOT / "config", profiles[0]["settings_file"]))
                manifest = Path(os.environ.get("TRACEBRIEF_HAILO_MANIFEST", ROOT / "config" / "hailo-manifest.json"))
                DEVICE["runtime"] = HailoRuntime(manifest, ROOT, confidence=settings.get("confidence", 0.40),
                                                 nms_iou=settings.get("nms_iou", 0.45),
                                                 batch_size=int(os.environ.get("TRACEBRIEF_HAILO_BATCH", "8")))
                DEVICE["error"] = None
            except DeviceError as exc:
                DEVICE["error"] = f"UGen300 backend (TRACEBRIEF_BACKEND=hailo) not ready: {exc} To run on the host CPU instead, set TRACEBRIEF_BACKEND=cpu."
                raise DeviceError(DEVICE["error"]) from exc
            except Exception as exc:
                DEVICE["error"] = str(exc)
                raise
        return DEVICE["runtime"]


def close_device():
    """Release the UGen300 models and VDevice: on shutdown, and after a device fault so the next job reloads."""
    with DEVICE_LOCK:
        runtime, DEVICE["runtime"] = DEVICE["runtime"], None
    if runtime is not None:
        runtime.close()


def warm_device():
    try:
        device_runtime()
    except Exception as exc:
        print(f"TraceBrief: UGen300 backend not ready: {exc}", file=sys.stderr, flush=True)


def device_health():
    """Health in hailo mode; reads the loaded runtime and never loads models itself."""
    runtime = DEVICE["runtime"]
    version = runtime.version if runtime else None
    if version is None:
        try:
            from .hailo_backend import hailort
            version = hailort().__version__
        except Exception:
            pass
    return {"ok": True, "version": "0.1.0", "backend": "hailo",
            "state": "ready" if runtime else "failed" if DEVICE["error"] else "not-loaded", "error": DEVICE["error"],
            "hardware": runtime.hardware_label() if runtime else "UGen300 not loaded", "hailort_version": version,
            "device": runtime.record() if runtime else None, "local_generation_available": runtime is not None,
            "visual_check_available": runtime is not None, "offline": True}


def public_profile(profile):
    return {key: profile[key] for key in ("id", "label", "description")}


def safe_file(base: Path, name: str) -> Path:
    target = (base / name).resolve()
    if not target.is_relative_to(base.resolve()) or not target.is_file():
        raise FileNotFoundError(name)
    return target


def frame_image(video: Path, output: Path, seconds: float, fps: float | None = None):
    """An evidence frame, 960 pixels wide: the frame the detector analysed at seconds (media.frame_at)."""
    media.frame_at(video, max(0.0, float(seconds)), output, fps, "scale=960:-2")


def detector_rate(raw) -> float | None:
    """The frame rate a single-clip detector record counted its frames at (compact_vision, vision)."""
    try:
        fps = float((raw.get("video") or {}).get("fps") or 0)
    except (TypeError, ValueError):
        return None
    return fps if math.isfinite(fps) and fps > 0 else None


def detector_crop(raw):
    """The visual check's crop maker for a single-clip record: the pair crop of the frame the detector analysed."""
    fps = detector_rate(raw)

    def crop(video, seconds, region, output):
        return media.crop_at(video, seconds, region, output, fps)
    return crop


def normalize_vision(raw, job_id, video, frame=None):
    """Adapt the vision worker output without inventing unobserved actions.

    frame(video, output, seconds) writes each evidence frame; by default frame_image, the frame the
    detector analysed at that time.
    """
    fps = detector_rate(raw)
    frame = frame or (lambda video, output, seconds: frame_image(video, output, seconds, fps))
    events = []
    directory = job_path(job_id).parent
    for index, item in enumerate(raw.get("events", [])):
        event_id = f"event-{index+1:03d}"
        start = float(item.get("start_s", item.get("start", 0)))
        end = float(item.get("end_s", item.get("end", start)))
        peak = float(item.get("peak_s", item.get("peak", (start+end)/2)))
        observations = []
        evidence = item.get("evidence", [])
        # Keep beginning, strongest moment, and end; avoid a wall of near-duplicate frames.
        if len(evidence) > 3:
            evidence = [evidence[0], min(evidence, key=lambda e: abs(float(e.get("time_s", peak))-peak)), evidence[-1]]
            evidence = list({float(e.get("time_s", peak)): e for e in evidence}.values())
        for i, ev in enumerate(evidence):
            if not isinstance(ev, dict):
                continue
            seconds = float(ev.get("time_s", ev.get("timestamp", ev.get("time", peak))))
            text = ev.get("text") or ev.get("description") or ev.get("observation") or ev.get("fact")
            if not text:
                text = "A person and a forklift were detected in the configured candidate region."
            pairs = ev.get("pairs", [])
            if pairs:
                # These are model-box facts, not world-space distance or contact.
                overlaps = any(
                    min(p["person_bbox"][2], p["forklift_bbox"][2]) > max(p["person_bbox"][0], p["forklift_bbox"][0])
                    and min(p["person_bbox"][3], p["forklift_bbox"][3]) > max(p["person_bbox"][1], p["forklift_bbox"][1])
                    for p in pairs
                )
                text = ("Predicted person and forklift boxes overlap in the image."
                        if overlaps else "Predicted person and forklift boxes are close but do not overlap in the image.")
            text = f"At {seconds:g}s: {text}"
            observations.append({"id": f"{event_id}-obs-{i+1}", "text": str(text), "time": seconds})
        if not observations:
            observations = [
                {"id": f"{event_id}-obs-1", "text": "The detector reported a person and a forklift in an image-space proximity candidate.", "time": peak},
                {"id": f"{event_id}-obs-2", "text": "This candidate was selected by the configured image-space rule; physical distance and contact have not been established.", "time": peak},
            ]
        for i, observation in enumerate(observations):
            image_name = f"{event_id}-{i+1}.jpg"
            try:
                frame(video, directory / "frames" / image_name, observation["time"])
                observation["frame_url"] = f"/media/jobs/{job_id}/frames/{image_name}"
                observation["image_kind"] = "Original video frame; predicted geometry is recorded in the audit JSON."
                observation["original_frame_url"] = observation["frame_url"]
                matching = next((e for e in evidence if float(e.get("time_s", -1)) == observation["time"]), {})
                predicted = Path(matching.get("image_path", ""))
                if predicted.is_file() and predicted.resolve().is_relative_to((directory / "vision").resolve()):
                    annotated_name = f"{event_id}-{i+1}-predicted.jpg"
                    shutil.copyfile(predicted, directory / "frames" / annotated_name)
                    observation["frame_url"] = f"/media/jobs/{job_id}/frames/{annotated_name}"
                    observation["image_kind"] = "Model-predicted boxes overlaid on the source frame; labels require visual verification."
            except (OSError, subprocess.SubprocessError):
                observation["frame_url"] = None
        uncertainties = list(item.get("limitations", [])) or [
            "Image-space proximity does not establish physical clearance, contact, or a safety violation.",
            "Site procedures and any required follow-up need supervisor review.",
        ]
        events.append({
            "id": event_id, "title": "Interaction awaiting review", "start": start,
            "end": end, "peak": peak, "status": "draft", "summary": "",
            "observations": observations, "uncertainties": uncertainties,
            "review_questions": [], "notes": "", "reviewer_notes": "", "history": [], "vision_record": item,
        })
    return events


def vision_estimate(settings: dict, duration) -> float:
    """Expected seconds of the detection stage on the host CPU; an unknown duration counts as a 10-second clip."""
    kind = "compact-onnx" if settings.get("backend") == "compact-onnx" else "grounding-dino"
    rate = max(1.0, float(settings.get("sample_fps", 1.0) or 1.0))
    return 10 + float(duration or 10) * VISION_SECONDS_PER_VIDEO_SECOND[kind] * rate


def vision_timeout(estimate: float) -> int:
    """Four times the estimate plus ten minutes, kept between VISION_TIMEOUT_FLOOR and VISION_TIMEOUT_CAP."""
    return int(min(VISION_TIMEOUT_CAP, max(VISION_TIMEOUT_FLOOR, 4 * estimate + 600)))


def stage_progress(low: float, high: float, elapsed: float, expected: float) -> float:
    """Progress within one stage: linear to 80% of its span at the expected time, then slower, never past high."""
    fraction = max(0.0, elapsed) / max(expected, 1e-6)
    share = 0.8 * fraction if fraction <= 1 else 0.8 + 0.2 * (1 - math.exp(1 - fraction))
    return round(low + (high - low) * share, 1)


@contextmanager
def ticking(job_id, low, high, expected, message):
    """Keep progress and elapsed time moving, once a second, while one blocking step runs."""
    update_job(job_id, progress=round(low, 1), message=message)
    stop = threading.Event()
    started = time.monotonic()

    def tick():
        while not stop.wait(1):
            elapsed = time.monotonic() - started
            try:
                update_job(job_id, progress=stage_progress(low, high, elapsed, expected), message=f"{message} · {int(elapsed)}s elapsed")
            except OSError as exc:
                print(f"TraceBrief: progress update failed: {exc}", file=sys.stderr, flush=True)
    thread = threading.Thread(target=tick, daemon=True, name="tracebrief-progress")
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()  # no late update may overwrite the next step's


class Phases:
    """A job's phases on the UGen300, kept in its job record: "phase" names the running (or next) one, "phases" times each.

    On the device the detector and the VLM take turns, so a job runs in phases instead of interleaving
    them: detection of the whole clip (batched), then the visual check of every candidate, then every
    draft. Measured on a UGen300 over USB (HailoRT 5.3.2, Windows host, 0.9 A): while the VLM generated
    in the same process the detector ran at 0.37-0.38 FPS (72-74 FPS alone), because the genai binding
    holds the GIL while it reads tokens, and a second process cannot open the device at all. done: the
    entries of phases already run, whose times are added to.
    """

    def __init__(self, job_id, device, done=None):
        self.job_id, self.device = job_id, device
        self.done = json.loads(json.dumps(done or {}))

    @contextmanager
    def run(self, name):
        """Time one phase; the entry it yields takes counts such as the candidates it handled."""
        entry = self.done.setdefault(name, {"seconds": 0.0})
        entry.setdefault("started_at", now())
        update_job(self.job_id, phase=name, phases=self.done)
        turns = getattr(self.device, "turns", None)
        switches, tick = (turns.switches if turns else 0), time.perf_counter()
        try:
            yield entry
            entry["finished_at"] = now()
        finally:  # also kept when the phase fails
            entry["seconds"] = round(entry["seconds"] + time.perf_counter() - tick, 3)
            if turns is not None:  # how often the device changed model during this phase: at most once, at its start
                entry["model_switches"] = entry.get("model_switches", 0) + turns.switches - switches
            update_job(self.job_id, phases=self.done)

    def seconds(self, name) -> float:
        return (self.done.get(name) or {}).get("seconds", 0.0)

    def total(self) -> float:
        return round(sum(entry.get("seconds", 0.0) for entry in self.done.values()), 3)

    def record(self) -> dict:
        """result["schedule"]: how the job used the device."""
        return {"mode": "phased", "order": [name for name in PHASES if name in self.done], "phases": self.done,
                "total_s": self.total()}


def source_sha256(raw: dict, path: Path) -> str:
    """The SHA-256 the detector streamed while reading this video; hashed here, in chunks, only if it is missing."""
    video = raw.get("video") or {}
    digest = str(video.get("sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", digest) and video.get("path") == str(path):
        return digest
    return workers.sha256(path)


def run_vision_process(job_id, directory, script, request, output, expected, message):
    """Run a detection script of scripts/ in the vision environment; progress moves until it exits or passes its limit."""
    interpreter = os.environ.get("TRACEBRIEF_VISION_PYTHON", sys.executable)
    timeout = vision_timeout(expected)
    with (directory / "vision.log").open("wb") as log:
        process = subprocess.Popen(
            [interpreter, str(ROOT / "scripts" / script), str(request), str(output)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"},
        )
        write_json(directory / "process.json", {"pid": process.pid, "started_at": now(), "command": script, "timeout_seconds": timeout})
        tick = time.monotonic()
        while process.poll() is None:
            elapsed = time.monotonic() - tick
            if elapsed > timeout:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise RuntimeError(f"Vision analysis exceeded the {timeout // 60}-minute runtime limit for this video.")
            update_job(job_id, progress=stage_progress(10, 69, elapsed, expected), message=message(elapsed))
            time.sleep(1)
        if process.returncode:
            raise RuntimeError(f"Vision analysis failed (exit {process.returncode}). See the local run log.")
    return load_json(output)


def apply_draft(event, audit, directory):
    """Record a draft's audit and put its accepted text on the candidate, keeping the original for provenance."""
    write_json(directory / f"generation-{event['id']}-audit.json", audit)
    accepted = audit["accepted"]
    event.update(title=accepted["title"], summary=accepted["summary"], review_questions=accepted["review_questions"],
                 generation_method=audit["method"], generation_warning=audit.get("warning"),
                 cited_evidence_ids=accepted["evidence_ids"])
    event["generation_provenance"] = {
        "original_input_observations": [{k:o[k] for k in ("id", "time", "text")} for o in event["observations"]],
        "raw_model_output": audit.get("model_output"),
        "automatic_check_errors": audit.get("validation_errors", []),
        "original_accepted_draft": accepted,
    }


def check_candidate(job_id, event, input_path, video_size, directory, device, extra):
    """The visual check of one candidate, on the device or the host CPU; a failed crop is recorded, a device fault raised."""
    if device is not None:
        from . import device_pipeline
        try:
            event["visual_check"] = device_pipeline.visual_check(device, event, input_path, video_size, directory / "frames",
                                                                 f"/media/jobs/{job_id}/frames", **extra)
        except (OSError, subprocess.SubprocessError) as exc:  # the host-side ffmpeg crop; device faults propagate
            event["visual_check"] = {"status": "unavailable", "reason": f"Visual check crop failed: {exc}"}
    else:
        try:
            event["visual_check"] = visual_check(event, input_path, video_size, directory / "frames",
                                                 f"/media/jobs/{job_id}/frames", ROOT, directory, **extra)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            event["visual_check"] = {"status": "unavailable", "reason": f"Visual check failed: {exc}"}


def draft_candidate(event, directory, device):
    if device is not None:
        from . import device_pipeline
        audit = device_pipeline.generate(device, event, directory)
    else:
        audit = generate(event, ROOT, directory)
    apply_draft(event, audit, directory)


def review_candidates(job_id, events, input_path, video_size, directory, device, crop=None, phases=None):
    """The visual check and the draft of every candidate, in 72-97% of the job's progress.

    On the host CPU each candidate is checked, then drafted, in its own slice of the range. On the
    UGen300 the steps run in phases (see Phases): every candidate's visual check, then every draft,
    so the device changes model once per phase rather than once per candidate. crop: how the pair crop is
    made (a single clip passes detector_crop; default media.crop_at at the file's own frame rate).
    phases (a Phases) times each device phase.
    """
    extra = {"make_crop": crop} if crop else {}
    if device is None:
        for index, event in enumerate(events):
            # Each candidate has its own slice of 72-97%: the visual check, then the draft.
            low, high = 72 + 25 * index / len(events), 72 + 25 * (index + 1) / len(events)
            middle = low + 0.45 * (high - low)
            with ticking(job_id, low, middle, CANDIDATE_STEP_SECONDS, f"Visual check of candidate {index+1} of {len(events)}"):
                check_candidate(job_id, event, input_path, video_size, directory, None, extra)
            with ticking(job_id, middle, high, CANDIDATE_STEP_SECONDS, f"Drafting local review {index+1} of {len(events)}"):
                draft_candidate(event, directory, None)
        return
    shares = {"visual_check": 0.45, "draft": 0.55}
    low = 72.0
    for step in ("visual_check", "draft"):
        high = low + 25 * shares[step]
        with phases.run(step) if phases else nullcontext({}) as entry:
            for index, event in enumerate(events):
                start, end = low + (high - low) * index / len(events), low + (high - low) * (index + 1) / len(events)
                message = (f"Visual check of candidate {index+1} of {len(events)} on the UGen300" if step == "visual_check"
                           else f"Drafting review {index+1} of {len(events)} on the UGen300")
                with ticking(job_id, start, end, CANDIDATE_STEP_SECONDS, message):
                    if step == "visual_check":
                        check_candidate(job_id, event, input_path, video_size, directory, device, extra)
                    else:
                        draft_candidate(event, directory, device)
            entry["candidates"] = len(events)
        low = high


def build_result(job_id, raw, events, source, input_path, inference_settings, search_profile, device, started):
    """The review record of a completed job: the source, its candidates and how each step was run."""
    video_info = raw.get("video", {})
    method = raw.get("model", {})
    result = {
        "id": job_id, "title": source["name"], "video_url": source["video_url"],
        "duration": video_info.get("duration_s", source.get("duration", 0)),
        "source_label": source["source_label"],
        "source_url": source.get("source_url"), "source_sha256": source_sha256(raw, input_path),
        "method_label": method.get("id", "Local RGB vision") if isinstance(method, dict) else str(method),
        "events": events, "warnings": raw.get("warnings", []),
        "analysis_seconds": round(time.monotonic()-started, 2),
        # Labeled from the device that ran the job; a CPU run is never labeled as a device run.
        "hardware_label": device.hardware_label() if device is not None else "Host CPU · Stage I prototype",
        "backend": "hailo" if device is not None else "cpu",
        "generated_at": now(), "evidence_version": "1.0", "vision_timing": raw.get("timing", {}),
        "vision_model": raw.get("model", {}),
        "vision_implementation": raw.get("implementation", {}),
        "requested_vision_settings": inference_settings,
        "search_profile": search_profile,
        "vision_settings": {k:v for k,v in raw.get("settings", {}).items() if k not in {"model_dir", "output_dir"}},
        "review_status": "unreviewed", "source": source,
    }
    if device is not None:
        result["device"] = device.record()
    return result


def process_job(job_id, input_path, source, inference_settings, search_profile=None):
    started = time.monotonic()
    directory = job_path(job_id).parent
    device = phases = None
    try:
        if backend() == "hailo":
            from . import device_pipeline, hailo_backend
            update_job(job_id, status="running", progress=2, message="Loading the UGen300 models")
            device = device_runtime()
            phases = Phases(job_id, device)
        update_job(job_id, status="running", progress=5, message="Reading local video and starting detection")
        settings = json.loads(json.dumps(inference_settings))
        settings["output_dir"] = str(directory / "vision")
        request = directory / "vision-input.json"
        output = directory / "vision-output.json"
        write_json(request, {"path": str(input_path), "settings": settings})
        expected = vision_estimate(settings, source.get("duration"))
        if device is not None:
            # The detection phase: the whole clip on the detector, then the evidence frames on the host.
            with phases.run("detection") as entry:
                with ticking(job_id, 10, 69, expected, "Detecting on the UGen300"):
                    raw = device_pipeline.analyze_video(device, input_path, settings)
                raw["implementation"] = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                         for name, module in (("vision_sha256", device_pipeline), ("backend_sha256", hailo_backend))}
                write_json(output, raw)
                update_job(job_id, progress=70, message="Building observations and linked frames")
                events = normalize_vision(raw, job_id, input_path)
                entry.update(sampled_frames=raw["timing"]["sampled_frames"], inference_s=raw["timing"]["inference_s"],
                             candidates=len(events))
        else:
            hint = "" if settings.get("backend") == "compact-onnx" else " · Grounding DINO takes about a minute per 10-second clip on CPU"
            raw = run_vision_process(job_id, directory, "vision_worker.py", request, output, expected,
                                     lambda elapsed: f"Analyzing video locally · {int(elapsed)}s elapsed{hint}")
            update_job(job_id, progress=70, message="Building observations and linked frames")
            events = normalize_vision(raw, job_id, input_path)
        video_info = raw.get("video", {})
        video_size = (int(video_info.get("width", 0)), int(video_info.get("height", 0)))
        review_candidates(job_id, events, input_path, video_size, directory, device, crop=detector_crop(raw), phases=phases)
        result = build_result(job_id, raw, events, source, input_path, inference_settings, search_profile, device, started)
        extra = {}
        if phases is not None:
            result["schedule"] = phases.record()
            extra["phase"] = "done"
        update_job(job_id, status="completed", progress=100, message="Analysis complete. Review each observation before approval.",
                   result=result, **extra)
    except Exception as exc:
        if isinstance(exc, DeviceError):
            close_device()  # a device fault can leave the runtime unusable; the next job reloads it
        update_job(job_id, status="failed", progress=0, message=str(exc), error=str(exc))


METHOD_LABELS = {
    "local-qwen-draft": "Local Qwen draft",
    "fixed-template": "Fixed template",
    "fixed-template-after-local-rejection": "Fixed template (model draft rejected by source checks)",
    "ugen300-qwen2-vl-draft": "UGen300 Qwen2-VL draft",
    "ugen300-qwen2.5-draft": "UGen300 Qwen2.5 draft",
}


def embedded_frame(url, alt, caption):
    """Inline one saved job frame as a data URI so the exported report stands alone."""
    relative = str(url or "").removeprefix("/media/jobs/")
    if not re.fullmatch(r"[a-f0-9]{16}/frames/[A-Za-z0-9_.-]+\.jpg", relative):
        return ""
    try:
        encoded = base64.b64encode(safe_file(RUNTIME / "jobs", relative).read_bytes()).decode("ascii")
    except FileNotFoundError:
        return "<p>Evidence image unavailable; consult the original video.</p>"
    return f"<figure><img style='width:100%;max-width:640px' alt='{html.escape(alt)}' src='data:image/jpeg;base64,{encoded}'><figcaption>{html.escape(caption)}</figcaption></figure>"


def visual_check_html(event):
    check = event.get("visual_check") or {}
    if check.get("status") not in ("confirmed", "check-suggested"):
        reason = check.get("reason", "Not run for this record.")
        return f"<h3 id='{html.escape(event.get('id', ''))}-look'>Visual check</h3><p class='meta'>Unavailable: {html.escape(reason)}</p>"
    seen = f"Forklift seen: {check['forklift']} · Person seen: {check['person']}"
    advice = ("" if check["status"] == "confirmed" else
              "<p><strong>The visual check did not see both a person and a forklift. Check whether this is a false candidate.</strong></p>")
    image = embedded_frame(check.get("crop_url"), f"Visual check crop at {check['time']:.2f} seconds",
                           "Crop around the predicted person and forklift boxes, as checked by the local vision-language model.")
    return (f"<h3 id='{html.escape(event.get('id', ''))}-look'>Visual check</h3><p class='meta'>{html.escape(check.get('runtime_label', 'Local Qwen2-VL-2B'))} · key frame at {check['time']:.2f}s · {html.escape(seen)}</p>"
            f"<p>{html.escape(check['description'])}</p>{advice}{image}")


def export_html(result):
    esc = lambda text: html.escape(str(text))
    blocks = []
    for event in result["events"]:
        observation_blocks = []
        for o in event["observations"]:
            image = embedded_frame(o.get("frame_url"), f"Evidence frame at {o['time']:.2f} seconds", o.get("image_kind", "Original video frame"))
            observation_blocks.append(f"<li id='{esc(o['id'])}'><strong>{o['time']:.2f}s</strong> — {esc(o['text'])} <small>{esc(o['id'])}</small>{image}</li>")
        observations = "".join(observation_blocks)
        unknowns = "".join(f"<li>{esc(x)}</li>" for x in event["uncertainties"])
        questions = "".join(f"<li>{esc(x)}</li>" for x in event.get("review_questions", []))
        blocks.append(f"<section><p class='label'>{esc(event['status'])} · {event['start']:.2f}–{event['end']:.2f}s</p><h2>{esc(event['title'])}</h2><p>{esc(event['summary'])}</p>{visual_check_html(event)}<h3>Linked observations</h3><ol>{observations}</ol><h3>Unresolved details</h3><ul>{unknowns}</ul><h3>Review questions</h3><ul>{questions}</ul><h3>Reviewer notes</h3><p>{esc(event.get('notes', '')) or 'None recorded.'}</p><p class='meta'>Draft method: {esc(METHOD_LABELS.get(event.get('generation_method'), event.get('generation_method') or 'not recorded'))}<br>Review history entries: {len(event.get('history', []))}</p></section>")
    return f"""<!doctype html><html lang='en'><meta charset='utf-8'><title>TraceBrief — {esc(result['title'])}</title><style>body{{font:16px/1.6 Arial,sans-serif;max-width:900px;margin:48px auto;color:#20313a;padding:24px}}h1{{font-size:36px}}h2{{font-size:25px}}h3{{font-size:17px}}.label{{text-transform:uppercase;font-size:12px;letter-spacing:1px;color:#146b62}}.meta,small{{color:#65747a;font-size:12px}}section{{border-top:1px solid #cbd5d8;padding:24px 0}}li{{margin-bottom:10px}}@media print{{body{{margin:0;max-width:none}}section{{break-inside:avoid}}}}</style><p class='label'>TraceBrief / Local incident review</p><h1>{esc(result['title'])}</h1><p>{esc(result['source_label'])}</p><p class='meta'>Created {esc(result['generated_at'])}<br>Source SHA-256: {esc(result['source_sha256'])}<br>{esc(result['hardware_label'])}<br>Search approach: {esc((result.get('search_profile') or {}).get('label', 'Earlier analysis; see recorded settings in JSON'))}</p><p>This document records candidate interactions for human review. It does not establish physical distance, contact, responsibility, or safety compliance.</p>{''.join(blocks) or '<section>No candidate events were found. This does not establish absence of risk.</section>'}<footer class='meta'>Generated locally by TraceBrief. The original video and audit JSON should accompany formal review.</footer></html>"""


def export_record(result):
    """Keep evidence facts while omitting machine-specific inference image paths."""
    portable = json.loads(json.dumps(result))
    for event in portable.get("events", []):
        for evidence in event.get("vision_record", {}).get("evidence", []):
            if "image_path" in evidence:
                evidence["image_file"] = Path(evidence.pop("image_path")).name
    return portable


class Handler(BaseHTTPRequestHandler):
    server_version = "TraceBrief/0.1"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.log_date_time_string()} {fmt % args}\n")

    def json_response(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD": self.wfile.write(data)

    def internal_error(self, exc):
        """Log the traceback and answer 500, rather than closing the connection with no response."""
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        try:
            self.json_response({"error": f"Internal error ({type(exc).__name__}); see the server log."}, 500)
        except OSError:
            pass  # the client has gone

    def read_json(self, limit=1024*1024):
        n = int(self.headers.get("Content-Length", 0))
        if not 0 <= n <= limit: raise ValueError("Request is too large")
        value = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(value, dict): raise ValueError("Request body must be a JSON object")
        return value

    def send_file(self, path):
        size = path.stat().st_size
        start, end = 0, size-1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
            if not match: raise ValueError("Unsupported byte range")
            left, right = match.groups()
            if left:
                start, end = int(left), min(int(right), end) if right else end
            elif right:
                start = max(0, size-int(right))
            if start >= size or start > end:
                self.send_response(416); self.send_header("Content-Range", f"bytes */{size}"); self.end_headers(); return
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", MIME.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end-start+1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        if range_header: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD": return
        with path.open("rb") as file:
            file.seek(start)
            remaining = end-start+1
            while remaining:
                data = file.read(min(65536, remaining))
                if not data: break
                self.wfile.write(data); remaining -= len(data)

    def do_HEAD(self): self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        try:
            if path == "/api/health":
                if backend() == "hailo":
                    return self.json_response(device_health())
                model = os.environ.get("TRACEBRIEF_LLM_MODEL", "")
                return self.json_response({"ok": True, "version": "0.1.0", "backend": "cpu", "local_generation_available": bool(model and Path(model).is_file()), "visual_check_available": look_configured(), "hardware": "host-cpu", "offline": True,
                                           "resident_workers": cpu_pipeline.enabled(), "workers": cpu_pipeline.status()})
            if path == "/api/samples":
                rows = [{**{k:v for k,v in row.items() if k != "path"}, "video_url": f"/media/samples/{row['id']}"} for row in samples()]
                return self.json_response({"samples": rows})
            if path == "/api/search-profiles":
                profiles = search_profiles()
                return self.json_response({"default": profiles["default"], "profiles": [public_profile(p) for p in profiles["profiles"]]})
            match = re.fullmatch(r"/api/jobs/([a-f0-9]{16})(/export)?", path)
            if match:
                job = read_job(match[1])
                if not match[2]: return self.json_response(job)
                if job["status"] != "completed": return self.json_response({"error": "Analysis has not completed"}, 409)
                form = parse_qs(url.query).get("format", ["html"])[0]
                if form == "json": data = json.dumps(export_record(job["result"]), ensure_ascii=False, indent=2).encode(); mime="application/json"
                elif form == "html": data = export_html(job["result"]).encode(); mime="text/html; charset=utf-8"
                else: raise ValueError("Export must be html or json")
                self.send_response(200); self.send_header("Content-Type", mime)
                self.send_header("Content-Disposition", f'attachment; filename="TraceBrief-{match[1]}.{form}"')
                self.send_header("Content-Length", str(len(data))); self.end_headers()
                if self.command != "HEAD": self.wfile.write(data)
                return
            if path.startswith("/media/samples/"):
                row = next(r for r in samples() if r["id"] == path.rsplit("/", 1)[1])
                return self.send_file(safe_file(ROOT / "data" / "samples", row["path"]))
            if path.startswith("/media/uploads/"):
                return self.send_file(safe_file(RUNTIME / "uploads", path.removeprefix("/media/uploads/")))
            if path.startswith("/media/jobs/"):
                relative = path.removeprefix("/media/jobs/")
                if not re.fullmatch(r"[a-f0-9]{16}/frames/[A-Za-z0-9_.-]+\.jpg", relative): raise FileNotFoundError(path)
                return self.send_file(safe_file(RUNTIME / "jobs", relative))
            static_name = "index.html" if path == "/" else path.removeprefix("/static/").lstrip("/")
            return self.send_file(safe_file(ROOT / "web", static_name))
        except (FileNotFoundError, StopIteration): self.json_response({"error": "Not found"}, 404)
        except (ValueError, json.JSONDecodeError) as exc: self.json_response({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError): pass
        except Exception as exc: self.internal_error(exc)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/uploads":
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= MAX_UPLOAD: raise ValueError("Choose a video smaller than 500 MB")
                name = Path(unquote(self.headers.get("X-Filename", "video.mp4"))).name
                ext = Path(name).suffix.lower()
                if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}: raise ValueError("Unsupported video format")
                upload_id = uuid.uuid4().hex[:16]
                target = RUNTIME / "uploads" / (upload_id+ext)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("wb") as file:
                    remaining = size
                    while remaining:
                        data = self.rfile.read(min(65536, remaining))
                        if not data: raise ValueError("Upload ended early")
                        file.write(data); remaining -= len(data)
                # The duration scales the detection time limit; None when ffprobe cannot read it.
                write_json(target.with_suffix(".json"), {"name": name, "path": target.name, "duration": media.duration(target)})
                return self.json_response({"id": upload_id, "filename": name}, 201)
            if path == "/api/jobs":
                data = self.read_json()
                if data.get("sample_id"):
                    source = dict(next(s for s in samples() if s["id"] == data["sample_id"]))
                    input_path = safe_file(ROOT / "data" / "samples", source["path"])
                    source["video_url"] = f"/media/samples/{source['id']}"
                elif isinstance(data.get("upload_id"), str) and re.fullmatch(r"[a-f0-9]{16}", data["upload_id"]):
                    meta = load_json(safe_file(RUNTIME / "uploads", data["upload_id"]+".json"))
                    input_path = safe_file(RUNTIME / "uploads", meta["path"])
                    source = {"name": meta["name"], "source_label": "User-provided video · locally stored", "video_url": f"/media/uploads/{meta['path']}"}
                    if meta.get("duration"):
                        source["duration"] = meta["duration"]
                else: raise ValueError("Select a sample or upload a video")
                profiles = search_profiles()
                profile_id = data.get("search_profile", profiles["default"])
                if not isinstance(profile_id, str): raise ValueError("Invalid search approach")
                profile = next((p for p in profiles["profiles"] if p["id"] == profile_id), None)
                if profile is None: raise ValueError("Only the compact detector (Fast search) runs on the UGen300 backend" if backend() == "hailo" else "Invalid search approach")
                inference_settings = load_json(safe_file(ROOT / "config", profile["settings_file"]))
                if not isinstance(inference_settings, dict): raise ValueError("Invalid local vision configuration")
                recorded_profile = public_profile(profile)
                job_id = uuid.uuid4().hex[:16]
                job = {"id": job_id, "status": "queued", "progress": 0, "message": "Queued for local analysis", "created_at": now(), "source": source,
                       "requested_vision_settings": inference_settings, "search_profile": recorded_profile}
                write_json(job_path(job_id), job)
                POOL.submit(process_job, job_id, input_path, source, json.loads(json.dumps(inference_settings)), json.loads(json.dumps(recorded_profile)))
                return self.json_response(job, 202)
            self.json_response({"error": "Not found"}, 404)
        except (FileNotFoundError, StopIteration): self.json_response({"error": "Source not found"}, 404)
        except (ValueError, json.JSONDecodeError) as exc: self.json_response({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError): pass
        except Exception as exc: self.internal_error(exc)

    def do_PATCH(self):
        try:
            match = re.fullmatch(r"/api/jobs/([a-f0-9]{16})/events/(event-\d{3})", urlparse(self.path).path)
            if not match: return self.json_response({"error": "Not found"}, 404)
            job_id, event_id = match.groups()
            data = self.read_json()
            if "reviewer_notes" in data:
                data["notes"] = data["reviewer_notes"]
            with LOCK:
                job = read_job(job_id)
                if job["status"] != "completed": return self.json_response({"error": "Analysis has not completed"}, 409)
                event = next(e for e in job["result"]["events"] if e["id"] == event_id)
                before = json.loads(json.dumps(event))
                for key in ("title", "summary", "notes"):
                    if key in data:
                        if not isinstance(data[key], str) or len(data[key]) > 10000: raise ValueError(f"Invalid {key}")
                        event[key] = data[key]
                event["reviewer_notes"] = event.get("notes", "")
                if "status" in data:
                    if not isinstance(data["status"], str) or data["status"] not in {"draft", "reviewed", "approved", "dismissed"}: raise ValueError("Invalid status")
                    event["status"] = data["status"]
                if "observations" in data:
                    if not isinstance(data["observations"], list): raise ValueError("Invalid observations")
                    existing = {o["id"]:o for o in event["observations"]}
                    for change in data["observations"]:
                        if not isinstance(change, dict) or not isinstance(change.get("id"), str) or change["id"] not in existing or not isinstance(change.get("text"), str): raise ValueError("Invalid observation edit")
                        existing[change["id"]]["text"] = change["text"][:10000]
                if event["status"] == "dismissed" and not event.get("notes", "").strip():
                    raise ValueError("Add a reviewer note explaining why this candidate is dismissed")
                stamp = now()
                event["history"].append({"at": stamp, "action": "reviewer-edit", "previous_status": before["status"], "new_status": event["status"],
                                         "changes": {k:{"before":before.get(k),"after":event.get(k)} for k in ("title","summary","notes","observations","status") if before.get(k)!=event.get(k)}})
                if event["status"] == "approved": event["approved_at"] = stamp
                elif "approved_at" in event: del event["approved_at"]
                states = [e["status"] for e in job["result"]["events"]]
                reviewed = sum(s in {"reviewed", "approved", "dismissed"} for s in states)
                job["result"]["review_status"] = "reviewed" if reviewed == len(states) else "partially-reviewed" if reviewed else "unreviewed"
                job["updated_at"] = now()
                write_json(job_path(job_id), job)
            return self.json_response({"event": event, "result": job["result"]})
        except (FileNotFoundError, StopIteration): self.json_response({"error": "Review not found"}, 404)
        except (ValueError, json.JSONDecodeError) as exc: self.json_response({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError): pass
        except Exception as exc: self.internal_error(exc)


def recover_interrupted():
    """At startup: a previous queued/running record is not evidence that its process survived."""
    for path in (RUNTIME / "jobs").glob("*/job.json"):
        job = load_json(path)
        if job.get("status") in {"queued", "running"}:
            job.update(status="failed", error="Application restarted before analysis completed.", message="Interrupted analysis. Start a new review.")
            write_json(path, job)


def main():
    parser = argparse.ArgumentParser(description="Run TraceBrief locally")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    selected = backend()  # an invalid TRACEBRIEF_BACKEND stops here
    # Child interpreters, the frozen per-call scripts included, read the UTF-8 files this server writes.
    os.environ.setdefault("PYTHONUTF8", "1")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    recover_interrupted()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if selected == "hailo":
        # Load the device models before the first job; /api/health reports the state or the error.
        POOL.submit(warm_device)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f"TraceBrief available at http://{args.host}:{args.port}" + (" · UGen300 backend" if selected == "hailo" else ""), flush=True)
    try:
        server.serve_forever()
    finally:
        if selected == "hailo":
            print("Waiting for a running analysis, then releasing the UGen300.", file=sys.stderr, flush=True)
            POOL.shutdown(wait=True, cancel_futures=True)
            close_device()
        workers.close_all()  # the resident CPU workers; they also exit on their own when this process's pipes close


if __name__ == "__main__": main()
