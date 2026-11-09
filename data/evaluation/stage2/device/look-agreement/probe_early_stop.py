"""Probe: does leaving the token stream once the JSON answer is complete end the generation on the UGen300?

Run on the device host after look_device.py, with the crops whose device answer went on after the JSON object:

    python probe_early_stop.py --manifest hailo-manifest.local.json --crops <dir> --answers look/device_answers.jsonl --out probe.json

For each crop: the look-v2 check with the stream left at the end of the first complete JSON object, then a full
HailoVLM.check of the next crop to show the device and context are usable afterwards. Records times and answers.
"""
import argparse
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
from PIL import Image  # noqa: E402
from tracebrief import hailo_backend as hb, look_prompt  # noqa: E402


def object_end(text):
    depth, quoted, escaped, start = 0, False, False, text.find("{")
    if start < 0:
        return None
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            escaped, quoted = (False, quoted) if escaped else (char == "\\", char != '"')
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def stopped_check(vlm, image):
    prompt = [{"role": "system", "content": [{"type": "text", "text": hb.VLM_SYSTEM}]},
              {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": look_prompt.PROMPT}]}]
    frame = hb.vlm_frame(image, vlm.frame_shape)
    vlm.vlm.clear_context()
    tokens, started = [], time.perf_counter()
    left = None
    try:
        with vlm.vlm.generate(prompt=prompt, frames=[frame], max_generated_tokens=hb.LOOK_TOKENS, do_sample=False, seed=hb.SEED) as stream:
            for token in stream:
                tokens.append(token)
                if object_end("".join(tokens)) is not None:
                    left = time.perf_counter()
                    break
        released = time.perf_counter()
    finally:
        vlm.vlm.clear_context()
    text = "".join(tokens)
    return {"raw_text": text, "tokens": len(tokens), "left_stream_s": round((left or released) - started, 3),
            "seconds": round(time.perf_counter() - started, 3), "release_s": round(released - (left or released), 3),
            "answer": look_prompt.parse(text)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--crops", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.answers).read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = [r for r in rows if (lambda e: e is not None and r["raw_text"][e:].replace("```", "").strip())(object_end(r["raw_text"]))][: args.limit]
    after = [r for r in rows if r not in chosen][: len(chosen)]
    hailo = hb.hailort()
    models = hb.verify_manifest(Path(args.manifest), Path.cwd(), hailo.__version__)
    vdevice = hb.shared_vdevice(hailo)
    vlm = hb.HailoVLM(vdevice, models["vlm"]["path"])
    report = []
    try:
        for row, following in zip(chosen, after):
            with Image.open(Path(args.crops) / f"{row['id']}.jpg") as image:
                stopped = stopped_check(vlm, image)
            with Image.open(Path(args.crops) / f"{following['id']}.jpg") as image:
                full = vlm.check(image)
            report.append({"id": row["id"], "full_run": {k: row[k] for k in ("seconds", "tokens", "answer")}, "stopped": stopped,
                           "same_answer": stopped["answer"] == row["answer"],
                           "next_check": {"id": following["id"], "seconds": full["seconds"], "tokens": full["tokens"],
                                          "same_raw_text_as_full_run": full["raw_text"] == following["raw_text"]}})
            print(json.dumps(report[-1], ensure_ascii=False)[:400], flush=True)
    finally:
        vlm.close()
        vdevice.release()
    Path(args.out).write_text(json.dumps({"conditions": "UGen300 over USB at 0.9 A (reduced-performance mode)", "probes": report},
                                         indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
