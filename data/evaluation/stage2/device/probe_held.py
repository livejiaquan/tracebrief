#!/usr/bin/env python3
"""While one process holds the UGen300, a second asks TraceBrief for it: how fast that fails, and with what message.

    python probe_held.py --manifest hailo-manifest.local.json --runtime RUNTIME --out held-guard.json

The first process opens the device as the server does (HailoRuntime: one VDevice, the detector and
the VLM). A second process then builds a HailoRuntime, runs one server job in hailo mode and starts
scripts/hailo/bench_ugen300.py; each must stop at once with the reason, not wait for the device.
"""
import argparse
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
CONDITIONS = "UGen300 over USB at 0.9 A (reduced-performance mode)"


def second(manifest: str, connection):
    from tracebrief import hailo_backend, server
    out = {}
    tick = time.perf_counter()
    try:
        hailo_backend.HailoRuntime(Path(manifest), ROOT)
        out["runtime"] = {"opened": True}
    except hailo_backend.DeviceError as exc:
        out["runtime"] = {"opened": False, "seconds": round(time.perf_counter() - tick, 3), "error": str(exc)}
    job_id = "abcdefabcdefabcd"
    server.write_json(server.job_path(job_id), {"id": job_id, "status": "queued"})
    settings = server.load_json(ROOT / "config" / "vision-compact-v4.json")
    tick = time.perf_counter()
    server.process_job(job_id, ROOT / "data" / "samples" / "warehouse-1.mp4", {"name": "warehouse-1", "source_label": "sample",
                                                                              "video_url": "/media/samples/warehouse-1"}, settings)
    job = server.read_job(job_id)
    out["server_job"] = {"status": job["status"], "seconds": round(time.perf_counter() - tick, 3), "error": job.get("error")}
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench_ugen300", ROOT / "scripts" / "hailo" / "bench_ugen300.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    tick = time.perf_counter()
    try:
        bench.main(["--manifest", manifest, "--output", str(server.RUNTIME / "bench-held.json")])
        out["bench"] = {"exited": False}
    except SystemExit as exc:
        out["bench"] = {"exited": True, "seconds": round(time.perf_counter() - tick, 3), "message": str(exc)}
    connection.send(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    os.environ.update(TRACEBRIEF_BACKEND="hailo", TRACEBRIEF_HAILO_MANIFEST=str(Path(args.manifest).resolve()),
                      TRACEBRIEF_RUNTIME=str(Path(args.runtime).resolve()), PYTHONUTF8="1")
    from tracebrief import hailo_backend
    tick = time.perf_counter()
    holder = hailo_backend.HailoRuntime(Path(args.manifest), ROOT, confidence=0.25, batch_size=8)
    record = {"conditions": CONDITIONS, "measured_at": datetime.now(timezone.utc).isoformat(),
              "holder": {"hardware": holder.hardware_label(), "load_s": round(time.perf_counter() - tick, 3)}}
    try:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=second, args=(str(Path(args.manifest).resolve()), child))
        tick = time.perf_counter()
        process.start()
        record["second_process"] = parent.recv() if parent.poll(300) else {"error": "no answer within 300 s"}
        record["second_process"]["wall_s_including_start"] = round(time.perf_counter() - tick, 3)
        process.join(60)
        import numpy as np
        frame = np.zeros((416, 416, 3), np.uint8)
        holder.detector.detect(frame, 640, 480)  # the holder still works afterwards
        record["holder"]["detects_afterwards"] = True
    finally:
        holder.close()
    Path(args.out).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
