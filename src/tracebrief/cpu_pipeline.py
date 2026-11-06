"""Job steps of the CPU backend, with the visual-check and drafting models resident in workers.

visual_check and generate take the arguments and return the records of look.visual_check and
generation.generate, the Stage I per-call path, and keep its crop region, prompts, decoding and checks:
only the model load and hash leave the per-candidate loop (src/tracebrief/workers.py), and the crop is
cut from the frame the detector analysed (media.crop_at) instead of look.crop_frame's -ss seek. A
candidate whose worker cannot start, or crashes, runs the per-call script (scripts/look_local.py,
scripts/generate_local.py); TRACEBRIEF_CPU_WORKERS=0 uses the per-call scripts throughout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

from . import generation, look, media, workers

LOOK_TIMEOUT = 600  # look.check_items' per-call limit
DRAFT_TIMEOUT = 240  # generation.generate's per-call limit


def enabled() -> bool:
    return os.environ.get("TRACEBRIEF_CPU_WORKERS", "1").strip().lower() not in {"0", "false", "no", "off"}


def log_dir(root: Path) -> Path:
    """Where each worker's own log goes: the runtime directory the server uses."""
    return Path(os.environ.get("TRACEBRIEF_RUNTIME", Path(root) / ".runtime")).resolve() / "workers"


def fallback(kind: str, exc: Exception):
    print(f"TraceBrief: {exc} Running the per-call {kind} script for this candidate.", file=sys.stderr, flush=True)


def visual_check(event: dict, video: Path, video_size: tuple[int, int], frames_dir: Path, frame_url: str,
                 root: Path, output_dir: Path, make_crop=None) -> dict:
    """look.visual_check, step for step, with the model in the resident look worker, on a frame-exact crop.

    make_crop(video, seconds, region, output) writes the pair crop: the single-clip job passes
    server.detector_crop; the default is media.crop_at at the file's frame rate. The frozen look.crop_frame (an -ss seek) is not used: it can
    return a later frame than the detector's (docs/ERRATA.md).
    """
    if not look.configured():
        return {"status": "unavailable", "reason": "Visual check model is not configured."}
    chosen = look.key_pair(event)
    if chosen is None:
        return {"status": "unavailable", "reason": "No predicted person-forklift pair to check."}
    frame, pair = chosen
    seconds = float(frame.get("time_s", 0))
    region = look.crop_region([pair["person_bbox"], pair["forklift_bbox"]], *video_size)
    crop = (make_crop or media.crop_at)(video, seconds, region, frames_dir / f"{event['id']}-look.jpg")
    threads = int(os.environ.get("TRACEBRIEF_VLM_THREADS", min(8, os.cpu_count() or 4)))
    output = check_items([{"id": event["id"], "image": str(crop)}], root, output_dir, event["id"], threads)
    result = output["results"][0]
    record = {"time": seconds, "crop_region": region, "crop_url": f"{frame_url}/{crop.name}",
              "seconds": output["seconds"], "model_file": output["model_file"], "model_sha256": output["model_sha256"],
              "prompt_version": output["prompt_version"], "raw_text": result.get("raw_text")}
    if "answer" not in result:
        return {**record, "status": "unavailable", "reason": f"Model answer could not be read: {result.get('error')}"}
    return {**record, **result["answer"], "status": look.verdict(result["answer"])}


def check_items(items: list[dict], root: Path, output_dir: Path, name: str, threads: int = 4) -> dict:
    """look.check_items through the resident look worker: the same request, output and log files."""
    if not enabled():
        return look.check_items(items, root, output_dir, name, threads)
    started = time.monotonic()
    model, projector = os.environ["TRACEBRIEF_VLM_MODEL"], os.environ["TRACEBRIEF_VLM_PROJECTOR"]
    request = output_dir / f"look-{name}-input.json"
    response = output_dir / f"look-{name}-output.json"
    request.write_text(json.dumps({"model": model, "projector": projector, "threads": threads, "items": items}, indent=2),
                       encoding="utf-8")
    interpreter = os.environ.get("TRACEBRIEF_VLM_PYTHON", os.environ.get("TRACEBRIEF_LLM_PYTHON", sys.executable))
    command = [interpreter, str(root / "scripts" / "look_worker.py"), "--model", model, "--projector", projector,
               "--threads", str(threads)]
    worker = workers.get("look", command, {**os.environ, "OMP_NUM_THREADS": str(threads)}, root, log_dir(root) / "look.log")
    try:
        worker.run(request, response, output_dir / f"look-{name}.log", LOOK_TIMEOUT)
    except (workers.WorkerUnavailable, workers.WorkerCrashed) as exc:
        fallback("look_local.py", exc)
        return look.check_items(items, root, output_dir, name, threads)
    output = json.loads(response.read_text(encoding="utf-8"))
    output["seconds"] = round(time.monotonic() - started, 3)
    return output


def draft_worker(root: Path) -> workers.Worker:
    """The resident Qwen2.5 worker that writes the drafts."""
    interpreter = os.environ.get("TRACEBRIEF_LLM_PYTHON", sys.executable)
    command = [interpreter, str(root / "scripts" / "draft_worker.py"), "--model", os.environ["TRACEBRIEF_LLM_MODEL"]]
    return workers.get("draft", command, {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}, root,
                       log_dir(root) / "draft.log")


def run_draft_worker(root: Path, request: Path, response: Path, log: Path) -> dict:
    """generation.run_generate_local through the resident draft worker."""
    worker = draft_worker(root)
    try:
        worker.run(request, response, log, DRAFT_TIMEOUT)
    except (workers.WorkerUnavailable, workers.WorkerCrashed) as exc:
        fallback("generate_local.py", exc)
        return generation.run_generate_local(root, request, response, log)
    return json.loads(response.read_text(encoding="utf-8"))


def generate(event: dict, root: Path, output_dir: Path) -> dict:
    """generation.generate with the draft model resident in a worker."""
    return generation.generate(event, root, output_dir, run=run_draft_worker if enabled() else None)


def status() -> dict:
    """The workers' state for /api/health; never starts one."""
    return {kind: worker.status() for kind, worker in list(workers.WORKERS.items())}
