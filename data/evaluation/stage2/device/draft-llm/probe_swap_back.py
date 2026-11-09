"""Which part of the swap back (LLM released, VLM loaded again) makes the UGen300 drop off USB.

Run on the device host (Windows, HailoRT 5.3.2) from a Windows working folder, one variant per process, because a
failed load leaves the device to re-enumerate:

    python probe_swap_back.py --manifest hailo-manifest.local.json --llm <Qwen2.5-1.5B-Instruct.hef> --out OUT.json
        [--no-detector] [--no-draft] [--settle 10] [--no-llm]

In draft_llm_device.py the VLM failed to load again both times it was asked to, after the LLM had drafted and been
released, with the detector HEF configured on the same VDevice (HAILO_COMMUNICATION_CLOSED while sending the HEF, then
LIBUSB_ERROR_IO; the device came back under a new USB address). probe_defaults.py, with no detector and no generation,
had loaded the LLM, released it and loaded the VLM without trouble. Each variant here changes one of those: no detector
(--no-detector), no generation (--no-draft), or a pause between releasing the LLM and loading the VLM (--settle);
--no-llm releases the VLM and loads it again with no LLM in between.
Sequence: [detector,] VLM, release VLM, [LLM, [one draft of d01,] release LLM,] [pause,] VLM, one visual check.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


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
from tracebrief import hailo_backend as hb  # noqa: E402

HERE = Path(__file__).resolve().parent
EVENTS = HERE.parent / "look-agreement" / "draft_events.json"


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-detector", action="store_true")
    parser.add_argument("--no-draft", action="store_true")
    parser.add_argument("--settle", type=float, default=0.0)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args(argv)
    from PIL import Image
    event = next(r for r in json.loads(EVENTS.read_text(encoding="utf-8"))["events"] if r["key"] == "d01")["event"]
    hailo = hb.hailort()
    record = {"started_utc": now(), "conditions": "UGen300 over USB at 0.9 A (reduced-performance mode)",
              "hailort_version": getattr(hailo, "__version__", None), "variant": {"detector": not args.no_detector,
              "draft": not args.no_draft and not args.no_llm, "llm": not args.no_llm, "settle_s": args.settle}, "steps": [], "identity": hb.identify(hailo)}
    models = hb.verify_manifest(Path(args.manifest), Path.cwd(), record["hailort_version"])
    turns = hb.Turns()
    vdevice = hb.shared_vdevice(hailo)
    parts = {}

    def step(name, function):
        tick = time.perf_counter()
        entry = {"step": name}
        record["steps"].append(entry)
        try:
            return function()
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            entry["seconds"] = round(time.perf_counter() - tick, 3)
            print(entry, flush=True)
    try:
        if not args.no_detector:
            parts["detector"] = step("load the detector", lambda: hb.HailoDetector(vdevice, models["detector"]["path"], batch_size=8, turns=turns))
        parts["vlm"] = step("load the VLM", lambda: hb.HailoVLM(vdevice, models["vlm"]["path"], turns=turns))
        step("release the VLM", parts.pop("vlm").close)
        if not args.no_llm:
            parts["llm"] = step("load the LLM", lambda: hb.HailoLLM(vdevice, Path(args.llm), turns=turns))
            if not args.no_draft:
                step("draft d01", lambda: parts["llm"].draft(event))
            step("release the LLM", parts.pop("llm").close)
        if args.settle:
            step("pause", lambda: time.sleep(args.settle))
        parts["vlm"] = step("load the VLM again", lambda: hb.HailoVLM(vdevice, models["vlm"]["path"], turns=turns))
        with Image.open(args.crop) as image:
            step("visual check", lambda: parts["vlm"].check(image))
        record["swap_back"] = "ok"
    except Exception:
        record["swap_back"] = "failed"
    finally:
        for name in ("llm", "vlm", "detector"):
            if name in parts:
                try:
                    parts[name].close()
                except Exception as exc:
                    record["steps"].append({"step": f"release the {name}", "error": f"{type(exc).__name__}: {exc}"})
        try:
            vdevice.release()
        except Exception as exc:
            record["steps"].append({"step": "release the VDevice", "error": f"{type(exc).__name__}: {exc}"})
        record["ended_utc"] = now()
        Path(args.out).write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(record["swap_back"], flush=True)


if __name__ == "__main__":
    main()
