"""CPU drafts of the 20 fixed candidates: generation.generate through the resident Qwen2.5 draft worker, as the app runs it.

Run from the repository root in WSL:

    TRACEBRIEF_LLM_MODEL=<qwen2.5-1.5b-instruct-q4_k_m.gguf> TRACEBRIEF_LLM_PYTHON=<interpreter with llama-cpp-python> \
    TRACEBRIEF_RUNTIME=<scratch> PYTHONPATH=src python data/evaluation/stage2/device/look-agreement/draft_cpu.py --out <dir>

The worker uses 4 threads (scripts/draft_worker.py), the draft-v2 prompt, the JSON grammar, temperature 0 and seed 19.
Every candidate gets its own folder for the request, output and log files; <out>/cpu_drafts.json holds the audits.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))
from tracebrief import cpu_pipeline, workers  # noqa: E402

EVENTS = Path(__file__).with_name("draft_events.json")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    rows = json.loads(EVENTS.read_text(encoding="utf-8"))["events"]
    record = {"started_utc": now(), "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                                             "loadavg_before": os.getloadavg()},
              "model": Path(os.environ["TRACEBRIEF_LLM_MODEL"]).name, "drafts": []}
    try:
        for row in rows:
            directory = out / row["key"]
            directory.mkdir(parents=True, exist_ok=True)
            tick = time.perf_counter()
            audit = cpu_pipeline.generate(row["event"], ROOT, directory)
            record["drafts"].append({"key": row["key"], "wall_s": round(time.perf_counter() - tick, 3), "audit": audit})
            print(row["key"], audit["method"], audit["seconds"], flush=True)
    finally:
        workers.close_all()
    record["ended_utc"] = now()
    record["host"]["loadavg_after"] = os.getloadavg()
    (out / "cpu_drafts.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
