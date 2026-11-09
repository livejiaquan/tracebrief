"""Device drafts of the 20 fixed candidates: device_pipeline.generate with the resident Qwen2-VL-2B, text only.

Run on the device host (Windows, HailoRT 5.3.2) from a Windows working folder:

    python draft_device.py --manifest hailo-manifest.local.json --out drafts [--format draft-v2-device1] [--split selection]

device_pipeline.generate is the app's device drafting step: HailoVLM.draft (the draft-v2 messages with the device
format hint appended to the system message, greedy, 320 tokens), then parse, generation.validate_draft and, on a
pass, generation.compose; a failure gives the labeled fixed template. --format picks a device format hint from
drafting.DEVICE_FORMATS (default: the app's draft-v2-device). Only the VLM is loaded. Each candidate gets its own
folder for the request and output files; <out>/device_drafts.json holds the audits.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import types


def source_folder():
    """--src DIR: the TraceBrief src/ folder to import (default: this checkout's src/); taken out of the arguments.

    On the device host the runs imported TraceBrief from the WSL checkout over a network path; that fixed path is
    now this option, the scripts' only change since the recorded runs.
    """
    if "--src" in sys.argv:
        at = sys.argv.index("--src")
        folder = sys.argv[at + 1]
        del sys.argv[at:at + 2]
        return folder
    return str(Path(__file__).resolve().parents[5] / "src")


SRC = source_folder()
sys.path.insert(0, SRC)
from tracebrief import device_pipeline, drafting, hailo_backend as hb  # noqa: E402

EVENTS = Path(__file__).with_name("draft_events.json")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--format", default=drafting.DEVICE_PROMPT_VERSION)
    parser.add_argument("--split", choices=("selection", "held-out"), default=None)
    args = parser.parse_args(argv)
    if args.format not in drafting.DEVICE_FORMATS:
        raise SystemExit(f"Unknown device format {args.format!r}; drafting has {sorted(drafting.DEVICE_FORMATS)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads(EVENTS.read_text(encoding="utf-8"))["events"] if args.split in (None, r["split"])]
    hailo = hb.hailort()
    record = {"started_utc": now(), "conditions": "UGen300 over USB at 0.9 A (reduced-performance mode)",
              "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
              "hailort_version": getattr(hailo, "__version__", None), "format": args.format, "format_text": drafting.DEVICE_FORMATS[args.format], "split": args.split,
              "max_generated_tokens": hb.DRAFT_TOKENS, "seed": hb.SEED, "decoding": "greedy (do_sample=False)", "drafts": []}
    record["identity"] = hb.identify(hailo)
    models = hb.verify_manifest(Path(args.manifest), Path.cwd(), record["hailort_version"])
    record["hefs"] = {name: {k: m[k] for k in ("file", "sha256", "source", "hash_s")} for name, m in models.items()}
    vdevice = hb.shared_vdevice(hailo)
    vlm = None
    try:
        tick = time.perf_counter()
        vlm = hb.HailoVLM(vdevice, models["vlm"]["path"], device_format=args.format)
        record["vlm_load_s"] = round(time.perf_counter() - tick, 3)
        runtime = types.SimpleNamespace(vlm=vlm, models=models, version=record["hailort_version"])
        for row in rows:
            directory = out / row["key"]
            directory.mkdir(parents=True, exist_ok=True)
            tick = time.perf_counter()
            audit = device_pipeline.generate(runtime, row["event"], directory)
            record["drafts"].append({"key": row["key"], "wall_s": round(time.perf_counter() - tick, 3), "audit": audit})
            print(row["key"], audit["method"], audit["model_output"]["generation_seconds"], audit["model_output"]["tokens"], flush=True)
    finally:
        if vlm is not None:
            vlm.close()
        vdevice.release()
    record["ended_utc"] = now()
    (out / "device_drafts.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
