#!/usr/bin/env python3
"""Benchmark TraceBrief's two device models on a UGen300 (Hailo-10H) and record exactly what ran.

Usage: python scripts/hailo/bench_ugen300.py [--manifest config/hailo-manifest.json] [--video data/samples/warehouse-1.mp4]

Records the device identity (the Python-API equivalent of `hailortcli scan` and
`hailortcli fw-control identify`), the HailoRT version, HEF hashes, model load times, detector FPS
and latency at batch 1 and 8, Qwen2-VL-2B time to first token and tokens/s for the visual check and
for the text-only draft, and detector FPS while the VLM generates: once in this process, sharing one
VDevice as the server does, and once with the VLM in a second process on the "SHARED" group. Writes
JSON and prints a table. Exits before loading anything when HailoRT or the device is missing, another
process holds it, or a HEF hash does not match. A run that fails later still writes what it measured, with
"incomplete" naming the error.

The second-process run is expected to fail on a Windows host with the UGen300 on USB: there only one
process can open the device (HailoRT 5.3.2, LIBUSB_ERROR_ACCESS), so it is reported as not measurable,
with the reason, and the rest of the report stands. That is why the server runs a job in phases.
"""
import argparse
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tracebrief import hailo_backend as hb  # noqa: E402
from tracebrief.look import crop_region  # noqa: E402

EVENT = {  # a fixed candidate for the text-only draft; the visual check sentence is replaced by the measured one
    "id": "event-001", "uncertainties": ["Image-space proximity does not establish physical distance, collision, near miss, or unsafe conduct."],
    "observations": [{"id": "event-001-obs-1", "time": 2.0, "text": "At 2s: Predicted person and forklift boxes are close but do not overlap in the image."},
                     {"id": "event-001-obs-2", "time": 3.0, "text": "At 3s: Predicted person and forklift boxes overlap in the image."}],
    "visual_check": {"status": "confirmed", "time": 2.0, "description": "A worker walks beside a forklift in the aisle."},
}
# Durations use time.perf_counter(): on the Windows device host (Python 3.10) time.monotonic() ticks every 15.6 ms,
# coarser than one detector frame.
GIL_NOTE = ("In one process the VLM's token read holds the Python GIL (the HailoRT 5.x binding does not release it), so the "
            "same-process figure is what the TraceBrief server would see; the second-process figure isolates device contention.")
SECOND_PROCESS_NOTE = ("Expected on a Windows host with the UGen300 on USB (HailoRT 5.3.2): only one process can open the device, so "
                       "the second process's VDevice fails with HAILO_LIBUSB_FAILURE (LIBUSB_ERROR_ACCESS) and device contention "
                       "between processes cannot be measured there. It may work on a Linux host or over M.2; not tried.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/hailo-manifest.json")
    parser.add_argument("--video", type=Path, default=ROOT / "data/samples/warehouse-1.mp4")
    parser.add_argument("--frames", type=int, default=256, help="letterboxed frames per detector run")
    parser.add_argument("--confidence", type=float, default=0.25, help="Fast search's frozen confidence (config/vision-compact-v4.json)")
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--output", type=Path, help="JSON path (default build/hailo/bench-<UTC time>.json)")
    parser.add_argument("--record-fixture", type=Path, help="also save one raw NMS buffer, e.g. tests/fixtures/hailo/nms_by_class.npz")
    parser.add_argument("--skip-process", action="store_true", help="skip the second-process concurrency run")
    return parser.parse_args(argv)


def load_frames(video: Path, count: int) -> tuple[list, list]:
    """count letterboxed frames spread over the video, and the source index of each."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    if total <= 0:
        cap.release()
        raise SystemExit(f"bench_ugen300: cannot read frames from {video}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, indices = [], []
    for i in range(count):
        index = i * total // count
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
        if ok:
            frames.append((hb.letterbox(image)[0], width, height)); indices.append(index)
    cap.release()
    if not frames:
        raise SystemExit(f"bench_ugen300: cannot decode {video}")
    return frames, indices


def source_frame(video: Path, index: int):
    import cv2
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read(); cap.release()
    if not ok:
        raise SystemExit(f"bench_ugen300: cannot decode frame {index} of {video}")
    return image


def ms(values) -> dict:
    values = sorted(values)
    return {"p50_ms": round(1000 * statistics.median(values), 3),
            "p95_ms": round(1000 * values[min(len(values) - 1, int(0.95 * len(values)))], 3), "samples": len(values)}


def bench_detector(vdevice, hef: Path, frames: list, batch: int, args) -> tuple[dict, object, list]:
    tick = time.perf_counter()
    detector = hb.HailoDetector(vdevice, hef, args.confidence, args.nms_iou, batch_size=batch)
    load_s = time.perf_counter() - tick
    detector.detect_batch(frames[:2 * detector.group])  # warm-up
    if batch == 1:
        results, latencies = [], []
        started = time.perf_counter()
        for frame in frames:
            tick = time.perf_counter(); results.append(detector.detect(*frame)); latencies.append(time.perf_counter() - tick)
        elapsed = time.perf_counter() - started
    else:
        started = time.perf_counter()
        results = detector.detect_batch(frames)
        elapsed = time.perf_counter() - started
        latencies = []
        for start in range(0, len(frames) - detector.group + 1, detector.group):
            tick = time.perf_counter(); detector.detect_batch(frames[start:start + detector.group]); latencies.append(time.perf_counter() - tick)
    record = {"batch_size": batch, "frames_per_submission": detector.group, "load_s": round(load_s, 3),
              "frames": len(frames), "fps": round(len(frames) / elapsed, 2),
              "latency_per_submission": ms(latencies) if latencies else None,
              "detections": sum(len(r) for r in results)}
    return record, detector, results


def pair_crop(video: Path, indices: list, results: list):
    """A person-forklift crop from the frame with the most detections, as the server would cut it."""
    from PIL import Image
    best = max(range(len(results)), key=lambda i: len(results[i]))
    image = source_frame(video, indices[best])
    people = [d["bbox_xyxy"] for d in results[best] if d["class_name"] == "person"]
    forklifts = [d["bbox_xyxy"] for d in results[best] if d["class_name"] == "forklift"]
    if people and forklifts:
        x, y, w, h = crop_region([people[0], forklifts[0]], image.shape[1], image.shape[0])
    else:
        x, y, w, h = 0, 0, image.shape[1], image.shape[0]
    return Image.fromarray(image[y:y + h, x:x + w, ::-1].copy())


def vlm_record(result: dict) -> dict:
    record = {k: result.get(k) for k in ("ttft_s", "tokens_per_s", "tokens", "seconds")}
    if "answer" in result or "error" in result:
        record["parsed"] = "answer" in result
    return record


def detector_while(detector, frames: list, busy) -> dict:
    count, started = 0, time.perf_counter()
    while busy():
        detector.detect(*frames[count % len(frames)]); count += 1
    elapsed = time.perf_counter() - started
    return {"frames": count, "seconds": round(elapsed, 3), "fps": round(count / elapsed, 2) if elapsed else None}


def same_process(detector, frames: list, vlm, crop) -> dict:
    outcome = {}
    worker = threading.Thread(target=lambda: outcome.update(vlm.check(crop)))
    worker.start()
    record = detector_while(detector, frames, worker.is_alive)
    worker.join()
    return {"detector": record, "vlm": vlm_record(outcome)}


def _vlm_worker(hef: str, crop, connection):
    """Second process: its own VDevice on group "SHARED", loads the VLM and answers one visual check."""
    from PIL import Image
    vdevice = vlm = None
    try:
        vdevice = hb.shared_vdevice()
        vlm = hb.HailoVLM(vdevice, Path(hef))
        connection.send(("loaded", None))
        connection.send(("done", vlm.check(Image.fromarray(crop))))
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if vlm is not None:
            vlm.close()
        if vdevice is not None:
            vdevice.release()


def second_process(detector, frames: list, hef: Path, crop) -> dict:
    import numpy as np
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_vlm_worker, args=(str(hef), np.asarray(crop), child))
    process.start()
    child.close()  # the child's end is the child's now: if it dies, the pipe reports end of file
    try:
        waited = 0
        while not parent.poll(1):  # a child that dies before it reports is not waited for
            waited += 1
            if not process.is_alive():
                raise RuntimeError(f"the second process exited (code {process.exitcode}) before it loaded the VLM")
            if waited >= 300:
                raise RuntimeError("the second process did not load the VLM within 300 s")
        kind, value = parent.recv()
        if kind != "loaded":
            raise RuntimeError(value)
        record = detector_while(detector, frames, lambda: not parent.poll() and process.is_alive())
        kind, value = parent.recv()  # EOFError if the child died without an answer
        if kind != "done":
            raise RuntimeError(value)
        return {"detector": record, "vlm": vlm_record(value)}
    finally:
        process.join(60)
        if process.is_alive():
            process.terminate()


def second_process_failure(exc: Exception) -> dict:
    """The second-process run that could not run, as a record; a device held by this process is the expected case."""
    reason = str(exc)
    held = hb.HELD in reason or hb.held_elsewhere(exc)
    return {"measured": False, "reason": reason, "device_held_by_this_process": held,
            "note": SECOND_PROCESS_NOTE if held else None}


def record_fixture(path: Path, detector, frames: list, results: list, version):
    """One raw HAILO_NMS_BY_CLASS buffer and HailoRT's own conversion of it, for tests/test_hailo_backend.py."""
    import numpy as np
    best = max(range(len(results)), key=lambda i: len(results[i]))
    detector.detect(*frames[best])
    raw, converted = detector.last_buffers()
    rows = np.concatenate([np.column_stack([np.asarray(b, dtype=np.float64).reshape(-1, hb.BOX), np.full(len(b), c)])
                           for c, b in enumerate(converted)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, raw=raw, converted=rows, width=frames[best][1], height=frames[best][2],
             order="HAILO_NMS_BY_CLASS", hailort=str(version), confidence=detector.confidence)


def table(report: dict) -> str:
    device = report["device"]["devices"][0]
    lines = [f"UGen300 benchmark · device {device['id']} · {device.get('architecture')} · firmware {device.get('firmware')} · HailoRT {report['hailort_version']}"]
    for name, model in report["hefs"].items():
        lines.append(f"{name + ' HEF':<36} {model['file']:<32} sha256 {model['sha256'][:16]}…  load {report['load_s'].get(name)} s")
    for run in report["detector"]:
        latency = run["latency_per_submission"] or {}
        lines.append(f"{'Detector batch ' + str(run['batch_size']):<36} {str(run['fps']):>8} FPS   p50 {latency.get('p50_ms')} ms  p95 {latency.get('p95_ms')} ms")
    for name in ("visual_check", "draft"):
        run = report["vlm"][name]
        lines.append(f"{name.replace('_', ' ').capitalize():<36} TTFT {run['ttft_s']} s   {run['tokens_per_s']} tok/s   {run['tokens']} tokens   {run['seconds']} s")
    for name, label in (("same_process", "Detector during VLM (same process)"), ("second_process", "Detector during VLM (second process)")):
        run = report["concurrent"].get(name)
        if isinstance(run, dict) and "detector" in run:
            lines.append(f"{label:<36} {str(run['detector']['fps']):>8} FPS   VLM {run['vlm']['tokens_per_s']} tok/s")
        elif isinstance(run, dict) and run.get("measured") is False:
            lines.append(f"{label:<36} not measurable: " + ("the device admits one process (expected on Windows over USB)"
                                                            if run["device_held_by_this_process"] else run["reason"]))
        elif run:
            lines.append(f"{label:<36} {run}")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    try:  # fail before touching the video or loading a model
        hailo = hb.hailort()
        version = getattr(hailo, "__version__", None)
        identity = hb.identify(hailo)
        models = hb.verify_manifest(args.manifest, ROOT, version)
    except hb.DeviceError as exc:
        raise SystemExit(f"bench_ugen300: {exc}")
    frames, indices = load_frames(args.video, args.frames)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "hailort_version": version, "device": identity,
              "hefs": {name: {k: m[k] for k in ("file", "sha256", "source", "hash_s")} for name, m in models.items()},
              "video": str(args.video), "settings": {"confidence": args.confidence, "nms_iou": args.nms_iou},
              "load_s": {}, "detector": [], "vlm": {}, "concurrent": {"note": GIL_NOTE}}
    output = args.output or ROOT / "build/hailo" / f"bench-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    vdevice = detector = vlm = None
    failure = None
    try:
        vdevice = hb.shared_vdevice(hailo)
        run, detector, _ = bench_detector(vdevice, models["detector"]["path"], frames, 8, args)
        report["detector"].append(run); detector.close(); detector = None
        run, detector, results = bench_detector(vdevice, models["detector"]["path"], frames, 1, args)
        report["detector"].insert(0, run); report["load_s"]["detector"] = run["load_s"]
        if args.record_fixture:
            record_fixture(args.record_fixture, detector, frames, results, version)
        crop = pair_crop(args.video, indices, results)
        tick = time.perf_counter(); vlm = hb.HailoVLM(vdevice, models["vlm"]["path"]); report["load_s"]["vlm"] = round(time.perf_counter() - tick, 3)
        check = vlm.check(crop)
        report["vlm"]["visual_check"] = {**vlm_record(check), "raw_text": check.get("raw_text")}
        event = json.loads(json.dumps(EVENT))
        if "answer" in check:
            event["visual_check"]["description"] = check["answer"]["description"]
        draft = vlm.draft(event)
        report["vlm"]["draft"] = {**vlm_record(draft), "raw_text": draft.get("text")}
        report["concurrent"]["same_process"] = same_process(detector, frames, vlm, crop)
        vlm.close(); vlm = None  # one GenAI model per device: release it before the second process loads its own
        if not args.skip_process:
            try:
                report["concurrent"]["second_process"] = second_process(detector, frames, models["vlm"]["path"], crop)
            except Exception as exc:
                report["concurrent"]["second_process"] = second_process_failure(exc)
    except BaseException as exc:  # for example another process took the device after identify()
        failure = exc
        report["incomplete"] = {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(exc, hb.DeviceError):
            raise SystemExit(f"bench_ugen300: {exc}")
        raise
    finally:
        try:
            for part in (vlm, detector):
                if part is not None:
                    part.close()
            if vdevice is not None:
                vdevice.release()
        finally:  # what was measured before a failure is kept, marked "incomplete"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            if failure is not None:
                print(f"bench_ugen300: saved the partial report to {output}", file=sys.stderr)
    print(table(report))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
