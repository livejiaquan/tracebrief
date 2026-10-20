"""Visual check of candidate moments with a local vision-language model.

The detector says where a person and a forklift are; this step crops the frame
around that pair and asks Qwen2-VL-2B whether both are actually there and what
they are doing. The answer is advisory: it can flag a likely false candidate or
add a scene description, but a reviewer still decides.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

CROP_MARGIN = 0.6
MAX_SIDE = 448


def crop_region(boxes: list[list[float]], width: int, height: int) -> list[int]:
    """Union of the boxes, expanded by CROP_MARGIN of its size on each side, clamped to the frame."""
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    mx, my = (x1 - x0) * CROP_MARGIN, (y1 - y0) * CROP_MARGIN
    left, top = max(0, int(x0 - mx)), max(0, int(y0 - my))
    right, bottom = min(width, int(x1 + mx)), min(height, int(y1 + my))
    return [left, top, right - left, bottom - top]


def crop_frame(video: Path, seconds: float, region: list[int], output: Path) -> Path:
    """Decode one source frame with ffmpeg and save the pair crop, longest side at most MAX_SIDE."""
    x, y, w, h = region
    scale = "scale='if(gt(iw,ih),min(iw,{0}),-2)':'if(gt(iw,ih),-2,min(ih,{0}))'".format(MAX_SIDE)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{seconds:.3f}", "-i", str(video),
                    "-frames:v", "1", "-vf", f"crop={w}:{h}:{x}:{y},{scale}", "-q:v", "2", "-y", str(output)],
                   check=True, timeout=60)
    return output


def configured() -> bool:
    model, projector = os.environ.get("TRACEBRIEF_VLM_MODEL"), os.environ.get("TRACEBRIEF_VLM_PROJECTOR")
    return bool(model and projector and Path(model).is_file() and Path(projector).is_file())


def key_pair(event: dict) -> tuple[dict, dict] | None:
    """The evidence frame closest to the candidate's peak that has a predicted pair, and its closest pair."""
    record = event.get("vision_record", {})
    peak = float(record.get("peak_s", event.get("peak", 0)))
    frames = [e for e in record.get("evidence", []) if isinstance(e, dict) and e.get("pairs")]
    if not frames:
        return None
    frame = min(frames, key=lambda e: abs(float(e.get("time_s", peak)) - peak))
    pair = min(frame["pairs"], key=lambda p: float(p.get("gap_px", 0)))
    return frame, pair


def verdict(answer: dict) -> str:
    """Both seen: the detector's pair is confirmed. Either missing: suggest the reviewer checks for a false candidate."""
    return "confirmed" if answer.get("forklift") == "yes" and answer.get("person") == "yes" else "check-suggested"


def visual_check(event: dict, video: Path, video_size: tuple[int, int], frames_dir: Path, frame_url: str,
                 root: Path, output_dir: Path) -> dict:
    """Crop the key pair of one candidate, ask the local vision-language model, and summarize the answer."""
    if not configured():
        return {"status": "unavailable", "reason": "Visual check model is not configured."}
    chosen = key_pair(event)
    if chosen is None:
        return {"status": "unavailable", "reason": "No predicted person-forklift pair to check."}
    frame, pair = chosen
    seconds = float(frame.get("time_s", 0))
    region = crop_region([pair["person_bbox"], pair["forklift_bbox"]], *video_size)
    crop = crop_frame(video, seconds, region, frames_dir / f"{event['id']}-look.jpg")
    threads = int(os.environ.get("TRACEBRIEF_VLM_THREADS", min(8, os.cpu_count() or 4)))
    output = check_items([{"id": event["id"], "image": str(crop)}], root, output_dir, event["id"], threads)
    result = output["results"][0]
    record = {"time": seconds, "crop_region": region, "crop_url": f"{frame_url}/{crop.name}",
              "seconds": output["seconds"], "model_file": output["model_file"], "model_sha256": output["model_sha256"],
              "prompt_version": output["prompt_version"], "raw_text": result.get("raw_text")}
    if "answer" not in result:
        return {**record, "status": "unavailable", "reason": f"Model answer could not be read: {result.get('error')}"}
    return {**record, **result["answer"], "status": verdict(result["answer"])}


def check_items(items: list[dict], root: Path, output_dir: Path, name: str, threads: int = 4) -> dict:
    """Run scripts/look_local.py on prepared crops; return its output plus timing, or an error record."""
    started = time.monotonic()
    request = output_dir / f"look-{name}-input.json"
    response = output_dir / f"look-{name}-output.json"
    request.write_text(json.dumps({"model": os.environ["TRACEBRIEF_VLM_MODEL"],
                                   "projector": os.environ["TRACEBRIEF_VLM_PROJECTOR"],
                                   "threads": threads, "items": items}, indent=2))
    interpreter = os.environ.get("TRACEBRIEF_VLM_PYTHON", os.environ.get("TRACEBRIEF_LLM_PYTHON", sys.executable))
    process = subprocess.run([interpreter, str(root / "scripts" / "look_local.py"), str(request), str(response)],
                             cwd=root, capture_output=True, text=True, timeout=600,
                             env={**os.environ, "OMP_NUM_THREADS": str(threads)})
    (output_dir / f"look-{name}.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(f"Visual check exited with code {process.returncode}; see run log.")
    output = json.loads(response.read_text())
    output["seconds"] = round(time.monotonic() - started, 3)
    return output
