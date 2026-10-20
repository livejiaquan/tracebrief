"""Local TraceBrief server. Only Python's standard library is required for HTTP.

Inference runs in configurable isolated Python environments. Runtime inputs and
review edits are saved locally; no cloud API is called by this application.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, unquote, urlparse
import uuid

from .generation import generate
from .look import configured as look_configured, visual_check

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path(os.environ.get("TRACEBRIEF_RUNTIME", ROOT / ".runtime")).resolve()
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracebrief-analysis")
LOCK = threading.RLock()
MAX_UPLOAD = 500 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{16}", job_id):
        raise ValueError("Invalid review ID")
    return RUNTIME / "jobs" / job_id / "job.json"


def read_job(job_id):
    return json.loads(job_path(job_id).read_text())


def update_job(job_id, **updates):
    with LOCK:
        path = job_path(job_id)
        job = json.loads(path.read_text())
        job.update(updates, updated_at=now())
        write_json(path, job)
    return job


def samples():
    manifest = ROOT / "data" / "samples" / "manifest.json"
    return json.loads(manifest.read_text())["samples"] if manifest.exists() else []


def search_profiles():
    return json.loads((ROOT / "config" / "search_profiles.json").read_text())


def public_profile(profile):
    return {key: profile[key] for key in ("id", "label", "description")}


def safe_file(base: Path, name: str) -> Path:
    target = (base / name).resolve()
    if not target.is_relative_to(base.resolve()) or not target.is_file():
        raise FileNotFoundError(name)
    return target


def frame_image(video: Path, output: Path, seconds: float):
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(max(0, seconds)),
         "-i", str(video), "-frames:v", "1", "-vf", "scale=960:-2", "-y", str(output)],
        check=True, capture_output=True, timeout=30,
    )


def normalize_vision(raw, job_id, video):
    """Adapt the vision worker output without inventing unobserved actions."""
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
                frame_image(video, directory / "frames" / image_name, observation["time"])
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


def process_job(job_id, input_path, source, inference_settings, search_profile=None):
    started = time.monotonic()
    directory = job_path(job_id).parent
    try:
        update_job(job_id, status="running", progress=5, message="Reading local video and starting detection")
        settings = json.loads(json.dumps(inference_settings))
        settings["output_dir"] = str(directory / "vision")
        request = directory / "vision-input.json"
        output = directory / "vision-output.json"
        write_json(request, {"path": str(input_path), "settings": settings})
        interpreter = os.environ.get("TRACEBRIEF_VISION_PYTHON", sys.executable)
        with (directory / "vision.log").open("w") as log:
            process = subprocess.Popen(
                [interpreter, str(ROOT / "scripts" / "vision_worker.py"), str(request), str(output)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"},
            )
            write_json(directory / "process.json", {"pid": process.pid, "started_at": now(), "command": "vision_worker.py", "timeout_seconds": 1800})
            tick = time.monotonic()
            hint = "" if settings.get("backend") == "compact-onnx" else " · Grounding DINO takes about a minute per 10-second clip on CPU"
            while process.poll() is None:
                elapsed = time.monotonic() - tick
                if elapsed > 1800:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise RuntimeError("Vision analysis exceeded the 30-minute runtime limit.")
                update_job(job_id, progress=min(65, 10+int(elapsed/10)), message=f"Analyzing video locally · {int(elapsed)}s elapsed{hint}")
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f"Vision analysis failed (exit {process.returncode}). See the local run log.")
        raw = json.loads(output.read_text())
        update_job(job_id, progress=70, message="Building observations and linked frames")
        events = normalize_vision(raw, job_id, input_path)
        video_info = raw.get("video", {})
        video_size = (int(video_info.get("width", 0)), int(video_info.get("height", 0)))
        for index, event in enumerate(events):
            update_job(job_id, progress=72+int(index/max(len(events), 1)*10), message=f"Visual check of candidate {index+1} of {len(events)}")
            try:
                event["visual_check"] = visual_check(event, input_path, video_size, directory / "frames",
                                                     f"/media/jobs/{job_id}/frames", ROOT, directory)
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
                event["visual_check"] = {"status": "unavailable", "reason": f"Visual check failed: {exc}"}
            update_job(job_id, progress=82+int(index/max(len(events), 1)*15), message=f"Drafting local review {index+1} of {len(events)}")
            audit = generate(event, ROOT, directory)
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
        method = raw.get("model", {})
        result = {
            "id": job_id, "title": source["name"], "video_url": source["video_url"],
            "duration": video_info.get("duration_s", source.get("duration", 0)),
            "source_label": source["source_label"],
            "source_url": source.get("source_url"), "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "method_label": method.get("id", "Local RGB vision") if isinstance(method, dict) else str(method),
            "events": events, "warnings": raw.get("warnings", []),
            "analysis_seconds": round(time.monotonic()-started, 2), "hardware_label": "Host CPU · Stage I prototype",
            "generated_at": now(), "evidence_version": "1.0", "vision_timing": raw.get("timing", {}),
            "vision_model": raw.get("model", {}),
            "vision_implementation": raw.get("implementation", {}),
            "requested_vision_settings": inference_settings,
            "search_profile": search_profile,
            "vision_settings": {k:v for k,v in raw.get("settings", {}).items() if k not in {"model_dir", "output_dir"}},
            "review_status": "unreviewed", "source": source,
        }
        update_job(job_id, status="completed", progress=100, message="Analysis complete. Review each observation before approval.", result=result)
    except Exception as exc:
        update_job(job_id, status="failed", progress=0, message=str(exc), error=str(exc))


METHOD_LABELS = {
    "local-qwen-draft": "Local Qwen draft",
    "fixed-template": "Fixed template",
    "fixed-template-after-local-rejection": "Fixed template (model draft rejected by source checks)",
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
        return f"<h3>Visual check</h3><p class='meta'>Unavailable: {html.escape(reason)}</p>"
    seen = f"Forklift seen: {check['forklift']} · Person seen: {check['person']}"
    advice = ("" if check["status"] == "confirmed" else
              "<p><strong>The visual check did not see both a person and a forklift. Check whether this is a false candidate.</strong></p>")
    image = embedded_frame(check.get("crop_url"), f"Visual check crop at {check['time']:.2f} seconds",
                           "Crop around the predicted person and forklift boxes, as checked by the local vision-language model.")
    return (f"<h3>Visual check</h3><p class='meta'>Local Qwen2-VL-2B · key frame at {check['time']:.2f}s · {html.escape(seen)}</p>"
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

    def read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        if not 0 <= n <= 1024*1024: raise ValueError("Request is too large")
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
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
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
                model = os.environ.get("TRACEBRIEF_LLM_MODEL", "")
                return self.json_response({"ok": True, "version": "0.1.0", "local_generation_available": bool(model and Path(model).is_file()), "visual_check_available": look_configured(), "hardware": "host-cpu", "offline": True})
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
                write_json(target.with_suffix(".json"), {"name": name, "path": target.name})
                return self.json_response({"id": upload_id, "filename": name}, 201)
            if path == "/api/jobs":
                data = self.read_json()
                if data.get("sample_id"):
                    source = dict(next(s for s in samples() if s["id"] == data["sample_id"]))
                    input_path = safe_file(ROOT / "data" / "samples", source["path"])
                    source["video_url"] = f"/media/samples/{source['id']}"
                elif isinstance(data.get("upload_id"), str) and re.fullmatch(r"[a-f0-9]{16}", data["upload_id"]):
                    meta = json.loads(safe_file(RUNTIME / "uploads", data["upload_id"]+".json").read_text())
                    input_path = safe_file(RUNTIME / "uploads", meta["path"])
                    source = {"name": meta["name"], "source_label": "User-provided video · locally stored", "video_url": f"/media/uploads/{meta['path']}"}
                else: raise ValueError("Select a sample or upload a video")
                profiles = search_profiles()
                profile_id = data.get("search_profile", profiles["default"])
                if not isinstance(profile_id, str): raise ValueError("Invalid search approach")
                profile = next((p for p in profiles["profiles"] if p["id"] == profile_id), None)
                if profile is None: raise ValueError("Invalid search approach")
                inference_settings = json.loads(safe_file(ROOT / "config", profile["settings_file"]).read_text())
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
                event["history"].append({"at": now(), "action": "reviewer-edit", "previous_status": before["status"], "new_status": event["status"],
                                         "changes": {k:{"before":before.get(k),"after":event.get(k)} for k in ("title","summary","notes","observations","status") if before.get(k)!=event.get(k)}})
                if event["status"] == "approved": event["approved_at"] = now()
                elif "approved_at" in event: del event["approved_at"]
                states = [e["status"] for e in job["result"]["events"]]
                reviewed = sum(s in {"reviewed", "approved", "dismissed"} for s in states)
                job["result"]["review_status"] = "reviewed" if reviewed == len(states) else "partially-reviewed" if reviewed else "unreviewed"
                job["updated_at"] = now()
                write_json(job_path(job_id), job)
            return self.json_response({"event": event, "result": job["result"]})
        except (FileNotFoundError, StopIteration): self.json_response({"error": "Review not found"}, 404)
        except (ValueError, json.JSONDecodeError) as exc: self.json_response({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description="Run TraceBrief locally")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    # A previous queued/running record is not evidence that its process survived.
    for path in (RUNTIME / "jobs").glob("*/job.json"):
        job = json.loads(path.read_text())
        if job.get("status") in {"queued", "running"}:
            job.update(status="failed", error="Application restarted before analysis completed.", message="Interrupted analysis. Start a new review.")
            write_json(path, job)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"TraceBrief available at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__": main()
