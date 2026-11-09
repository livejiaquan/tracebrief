"""The Stage I look-v2 test crops on the UGen300: HailoVLM.check on every published crop, one resident model.

Run on the device host (Windows, HailoRT 5.3.2) from a Windows working folder, so the HailoRT log lands there:

    python look_device.py --manifest hailo-manifest.local.json --crops <dir of t-*.jpg> \
        --freeze <tracebrief>/data/evaluation/look/FREEZE.json --out look

Every crop is checked against the SHA-256 that look/FREEZE.json records before the device is opened. Only the
VLM is loaded (the detector HEF is hashed with the manifest but never configured). HailoVLM.check clears the
context before and after each crop, so nothing carries over between items. One JSON line per crop goes to
<out>/device_answers.jsonl as soon as it is answered; a rerun skips the crops already answered.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
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
import numpy as np  # noqa: E402
import PIL  # noqa: E402
from PIL import Image  # noqa: E402
from tracebrief import hailo_backend as hb, look, look_prompt  # noqa: E402


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--crops", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    ids = list(freeze["test_items"]["ids"])[: args.limit]
    expected = freeze["test_items"]["crop_sha256"]
    crops = {}
    for item in ids:
        path = Path(args.crops) / f"{item}.jpg"
        digest = file_sha256(path)
        if digest != expected[item]:
            raise SystemExit(f"{path}: SHA-256 {digest} differs from look/FREEZE.json ({expected[item]})")
        crops[item] = (path, digest)
    answers = out / "device_answers.jsonl"
    done = set()
    if answers.is_file():
        done = {json.loads(line)["id"] for line in answers.read_text(encoding="utf-8").splitlines() if line.strip()}
    meta = {"started_utc": now(), "conditions": "UGen300 over USB at 0.9 A (reduced-performance mode)",
            "host": {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__,
                     "pillow": PIL.__version__},
            "freeze_sha256": file_sha256(args.freeze), "items": len(ids), "already_answered": len(done),
            "prompt_version": look_prompt.PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(look_prompt.PROMPT.encode()).hexdigest(),
            "system_prompt": hb.VLM_SYSTEM, "max_generated_tokens": hb.LOOK_TOKENS, "seed": hb.SEED,
            "decoding": "greedy (do_sample=False)", "context": "cleared before and after every crop"}
    hailo = hb.hailort()
    meta["hailort_version"] = getattr(hailo, "__version__", None)
    meta["identity"] = hb.identify(hailo)
    models = hb.verify_manifest(Path(args.manifest), Path.cwd(), meta["hailort_version"])
    meta["hefs"] = {name: {k: m[k] for k in ("file", "sha256", "source", "hash_s")} for name, m in models.items()}
    vdevice = hb.shared_vdevice(hailo)
    vlm = None
    try:
        tick = time.perf_counter()
        vlm = hb.HailoVLM(vdevice, models["vlm"]["path"])
        meta["vlm_load_s"] = round(time.perf_counter() - tick, 3)
        meta["frame_shape"] = list(vlm.frame_shape)
        (out / "device_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        with open(answers, "a", encoding="utf-8") as handle:
            for index, item in enumerate(ids):
                if item in done:
                    continue
                path, digest = crops[item]
                with Image.open(path) as image:
                    size = list(image.size)
                    frame = hb.vlm_frame(image, vlm.frame_shape)
                    tick = time.perf_counter()
                    entry = vlm.check(image)
                    wall = time.perf_counter() - tick
                record = {"id": item, "crop_sha256": digest, "vlm_frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                          "crop_size": size, **entry, "wall_s": round(wall, 3),
                          "verdict": look.verdict(entry["answer"]) if "answer" in entry else "unavailable",
                          "finished_utc": now()}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"{index + 1}/{len(ids)} {item} {record['verdict']} {entry['seconds']:.1f}s {entry['tokens']} tokens", flush=True)
    finally:
        if vlm is not None:
            vlm.close()
        vdevice.release()
    meta["ended_utc"] = now()
    (out / "device_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
