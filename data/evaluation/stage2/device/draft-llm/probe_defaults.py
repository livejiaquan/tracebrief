"""What the two GenAI HEFs default to, before any draft is made: generation settings, context capacity, chat template.

Run on the device host (Windows, HailoRT 5.3.2) from a Windows working folder, before PREREG.json was written:

    python probe_defaults.py --llm <Qwen2.5-1.5B-Instruct.hef> --vlm <Qwen2-VL-2B-Instruct.hef> --out defaults.json

HailoRT fills every generation setting a call leaves out from the HEF (pyhailort.py: LLM.generate and VLM.generate
start from create_generator_params()). The app's device calls set only do_sample, seed and max_generated_tokens, so the
rest, notably frequency_penalty, are the HEF's. This reads those defaults on the shared VDevice, one GenAI model at a
time (LLM loaded, read, released; then the VLM), and generates nothing.
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

FIELDS = ("temperature", "top_p", "top_k", "frequency_penalty", "max_generated_tokens", "do_sample", "seed")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def params(wrapper) -> dict:
    value = wrapper.create_generator_params()
    return {field: getattr(value, field, None) for field in FIELDS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", required=True)
    parser.add_argument("--vlm", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    hailo = hb.hailort()
    from hailo_platform.genai import LLM, VLM
    record = {"started_utc": now(), "conditions": "UGen300 over USB at 0.9 A (reduced-performance mode)",
              "hailort_version": getattr(hailo, "__version__", None), "identity": hb.identify(hailo)}
    vdevice = hb.shared_vdevice(hailo)
    try:
        tick = time.perf_counter()
        llm = LLM(vdevice, args.llm)
        loaded = time.perf_counter()
        record["llm"] = {"file": Path(args.llm).name, "load_s": round(loaded - tick, 3), "defaults": params(llm._llm),
                         "max_context_capacity": llm.max_context_capacity(), "stop_tokens": llm.get_stop_tokens(),
                         "prompt_template": llm.prompt_template()}
        tick = time.perf_counter()
        llm.release()
        record["llm"]["release_s"] = round(time.perf_counter() - tick, 3)
        tick = time.perf_counter()
        vlm = VLM(vdevice, args.vlm)
        loaded = time.perf_counter()
        record["vlm"] = {"file": Path(args.vlm).name, "load_s": round(loaded - tick, 3), "defaults": params(vlm._vlm),
                         "frame_shape": list(vlm.input_frame_shape())}
        tick = time.perf_counter()
        vlm.release()
        record["vlm"]["release_s"] = round(time.perf_counter() - tick, 3)
    finally:
        vdevice.release()
    record["ended_utc"] = now()
    Path(args.out).write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("llm", "vlm")}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
