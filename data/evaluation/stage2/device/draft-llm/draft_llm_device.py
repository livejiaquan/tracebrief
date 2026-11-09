"""Device drafts of the 20 fixed candidates with the Qwen2.5-1.5B-Instruct HEF, and what swapping the GenAI model costs.

Run on the device host (Windows, HailoRT 5.3.2) from a Windows working folder:

    python draft_llm_device.py --manifest hailo-manifest.local.json --llm <Qwen2.5-1.5B-Instruct.hef> --crop <jpg> --out OUT
        [--format draft-v2-device1] [--split selection] [--cycles 3]

One process, one VDevice, as the app holds them: the detector HEF and the VLM are loaded and the VLM checks --crop. Then
each cycle releases the VLM, loads the LLM (HailoLLM, the hint --format), drafts, releases the LLM, loads the VLM again,
checks the same crop and runs one detector batch. The first cycle drafts the candidates of draft_events.json (those of
--split, or all 20); later cycles draft d01 once, to time the swap again (--switch-only: every cycle drafts d01 once).
Each draft is device_pipeline.generate with runtime.draft_model "llm": HailoLLM.draft, then parse,
generation.validate_draft and compose, or the labeled template. Only one GenAI model is loaded at any time.
<out>/llm_drafts.json holds the audits and every swap's times; it is rewritten after every step, so a run the device
ends early keeps what it measured, with the failed step and HailoRT's error.
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

HERE = Path(__file__).resolve().parent
EVENTS = HERE.parent / "look-agreement" / "draft_events.json"
PREREG = HERE / "PREREG.json"


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timed(function):
    tick = time.perf_counter()
    value = function()
    return value, round(time.perf_counter() - tick, 3)


def look(vlm, crop):
    from PIL import Image
    with Image.open(crop) as image:
        entry = vlm.check(image)
    return {k: entry.get(k) for k in ("raw_text", "answer", "error", "seconds", "ttft_s", "tokens", "tokens_per_s")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--format", default=drafting.DEVICE_PROMPT_VERSION)
    parser.add_argument("--split", choices=("selection", "held-out"), default=None)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--switch-only", action="store_true", help="draft only d01 in every cycle")
    args = parser.parse_args(argv)
    if args.format not in drafting.DEVICE_FORMATS:
        raise SystemExit(f"Unknown device format {args.format!r}; drafting has {sorted(drafting.DEVICE_FORMATS)}")
    import numpy as np
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads(EVENTS.read_text(encoding="utf-8"))["events"] if args.split in (None, r["split"])]
    first_row = next(r for r in json.loads(EVENTS.read_text(encoding="utf-8"))["events"] if r["key"] == "d01")
    hailo = hb.hailort()
    record = {"started_utc": now(), "conditions": "UGen300 over USB at 0.9 A (reduced-performance mode)",
              "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
              "hailort_version": getattr(hailo, "__version__", None), "format": args.format,
              "format_text": drafting.DEVICE_FORMATS[args.format], "split": args.split, "max_generated_tokens": hb.DRAFT_TOKENS,
              "seed": hb.SEED, "decoding": "greedy (do_sample=False)", "frequency_penalty": hb.LLM_FREQUENCY_PENALTY,
              "switch_only": args.switch_only, "crop": {"file": Path(args.crop).name, "sha256": hb.sha256(Path(args.crop))},
              "drafts": [], "cycles": [], "errors": []}

    def save():
        (out / "llm_drafts.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    record["identity"] = hb.identify(hailo)
    models = hb.verify_manifest(Path(args.manifest), Path.cwd(), record["hailort_version"])
    expected = json.loads(PREREG.read_text(encoding="utf-8"))["model"]
    llm_path = Path(args.llm)
    digest, hash_s = timed(lambda: hb.sha256(llm_path))
    if digest != expected["sha256"]:
        raise SystemExit(f"{llm_path.name} hashes to {digest}, not the {expected['sha256']} PREREG.json names")
    models["llm"] = {"path": llm_path, "file": llm_path.name, "sha256": digest, "source": expected["source"], "hash_s": hash_s}
    record["hefs"] = {name: {k: m[k] for k in ("file", "sha256", "source", "hash_s")} for name, m in models.items()}
    vdevice = hb.shared_vdevice(hailo)
    turns = hb.Turns()
    detector = vlm = llm = None
    frames = [(np.zeros((hb.INPUT_SIZE, hb.INPUT_SIZE, 3), np.uint8), 640, 480)] * 8
    step = "load the detector"
    try:
        detector, record["detector_load_s"] = timed(lambda: hb.HailoDetector(vdevice, models["detector"]["path"], batch_size=8, turns=turns))
        step = "load the VLM"
        vlm, record["vlm_load_s"] = timed(lambda: hb.HailoVLM(vdevice, models["vlm"]["path"], turns=turns))
        step = "visual check before the first swap"
        record["check_before"] = look(vlm, args.crop)
        print("check before", record["check_before"]["seconds"], flush=True)
        for cycle in range(args.cycles):
            entry = {"cycle": cycle + 1}
            record["cycles"].append(entry)
            step = f"cycle {cycle + 1}: release the VLM"
            _, entry["vlm_release_s"] = timed(vlm.close)
            vlm = None
            step = f"cycle {cycle + 1}: load the LLM"
            llm, entry["llm_load_s"] = timed(lambda: hb.HailoLLM(vdevice, llm_path, turns=turns, device_format=args.format))
            runtime = types.SimpleNamespace(draft_model="llm", llm=llm, models=models, version=record["hailort_version"])
            entry["drafts"] = drafted = []
            for row in (rows if cycle == 0 and not args.switch_only else [first_row]):
                step = f"cycle {cycle + 1}: draft {row['key']}"
                directory = out / (row["key"] if cycle == 0 else f"cycle{cycle + 1}-{row['key']}")
                directory.mkdir(parents=True, exist_ok=True)
                audit, wall = timed(lambda: device_pipeline.generate(runtime, row["event"], directory))
                output = audit["model_output"]
                drafted.append({"key": row["key"], "ttft_s": output.get("ttft_s"), "generation_seconds": output.get("generation_seconds"),
                                "method": audit["method"], "raw_text": output.get("raw_text")})
                if cycle == 0 and not args.switch_only:
                    record["drafts"].append({"key": row["key"], "wall_s": wall, "audit": audit})
                save()
                print(cycle + 1, row["key"], audit["method"], output["generation_seconds"], output["tokens"], flush=True)
            entry["first_draft_ttft_s"] = drafted[0]["ttft_s"]
            step = f"cycle {cycle + 1}: release the LLM"
            _, entry["llm_release_s"] = timed(llm.close)
            llm = None
            step = f"cycle {cycle + 1}: load the VLM again"
            vlm, entry["vlm_load_s"] = timed(lambda: hb.HailoVLM(vdevice, models["vlm"]["path"], turns=turns))
            step = f"cycle {cycle + 1}: visual check after the swap back"
            entry["check_after"] = look(vlm, args.crop)
            entry["check_same_reply"] = entry["check_after"]["raw_text"] == record["check_before"]["raw_text"]
            step = f"cycle {cycle + 1}: detector batch after the swap back"
            results, entry["detector_batch_s"] = timed(lambda: detector.detect_batch(frames))
            entry["detector_batch_frames"] = len(results)
            entry["swap_to_llm_s"] = round(entry["vlm_release_s"] + entry["llm_load_s"], 3)
            entry["swap_back_s"] = round(entry["llm_release_s"] + entry["vlm_load_s"], 3)
            entry["round_trip_s"] = round(entry["swap_to_llm_s"] + entry["swap_back_s"], 3)
            save()
            print("cycle", cycle + 1, {k: entry[k] for k in ("vlm_release_s", "llm_load_s", "llm_release_s", "vlm_load_s",
                                                              "first_draft_ttft_s", "check_same_reply")}, flush=True)
        record["turn_switches"] = turns.switches
    except Exception as exc:
        record["errors"].append({"step": step, "at_utc": now(), "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        for name, part in (("LLM", llm), ("VLM", vlm), ("detector", detector)):
            if part is not None:
                try:
                    part.close()
                except Exception as exc:
                    record["errors"].append({"step": f"release the {name}", "at_utc": now(), "error": f"{type(exc).__name__}: {exc}"})
        try:
            vdevice.release()
        except Exception as exc:
            record["errors"].append({"step": "release the VDevice", "at_utc": now(), "error": f"{type(exc).__name__}: {exc}"})
        record["ended_utc"] = now()
        save()




if __name__ == "__main__":
    main()
