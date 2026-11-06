"""Resident visual-check worker: scripts/look_local.py's model, prompt and decoding, loaded once.

Usage: python scripts/look_worker.py --model Qwen2-VL-2B.gguf --projector mmproj.gguf --threads 8

Started by the server (src/tracebrief/cpu_pipeline.py); see src/tracebrief/workers.py for the
protocol. Each request names an input file of look_local.py's format and gets look_local.py's output
record. The prompt and parser are imported from tracebrief.look_prompt, which tests check against
look_local.py; the model is reset and reseeded before every item, so each answer is the one a fresh
look_local.py process gives. No network access is used.
"""
import argparse
import base64
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tracebrief import workers  # noqa: E402
from tracebrief.look_prompt import PROMPT, PROMPT_VERSION, parse  # noqa: E402

SEED = 19
CONTEXT = 4096
MAX_TOKENS = 160


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Resident Qwen2-VL-2B visual-check worker (JSON lines on stdin/stdout)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--projector", required=True)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args(argv)


def load(args):
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MTMDChatHandler
    started = time.monotonic()
    handler = MTMDChatHandler(clip_model_path=args.projector, verbose=False, use_gpu=False)
    llm = Llama(model_path=args.model, chat_handler=handler, n_ctx=CONTEXT, n_threads=args.threads,
                n_threads_batch=args.threads, seed=SEED, verbose=False)
    loaded = time.monotonic()
    record = {"model": args.model, "projector": args.projector, "threads": args.threads,
              "model_file": Path(args.model).name, "model_sha256": workers.sha256(args.model),
              "projector_file": Path(args.projector).name, "projector_sha256": workers.sha256(args.projector),
              "runtime": {"llama-cpp-python": version("llama-cpp-python")},
              "load_seconds": round(loaded - started, 3), "hash_seconds": round(time.monotonic() - loaded, 3),
              "started_at": workers.now()}
    return {"llm": llm, "record": record, "requests": 0}, record


def check(llm, item: dict) -> dict:
    """One crop, exactly as look_local.py asks it."""
    uri = "data:image/jpeg;base64," + base64.b64encode(Path(item["image"]).read_bytes()).decode()
    llm.reset()
    llm.set_seed(SEED)  # the seed state of a freshly loaded model
    tick = time.monotonic()
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": PROMPT}]}],
        temperature=0.0, max_tokens=MAX_TOKENS)
    raw = response["choices"][0]["message"]["content"]
    entry = {"id": item["id"], "raw_text": raw,
             "seconds": round(time.monotonic() - tick, 3), "usage": response.get("usage", {})}
    try:
        entry["answer"] = parse(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        entry["error"] = str(exc)
    return entry


def answer(state, request):
    job = json.loads(Path(request["input"]).read_text(encoding="utf-8"))
    record = state["record"]
    for key in ("model", "projector", "threads"):
        if job.get(key) != record[key]:
            raise ValueError(f"This worker has {key}={record[key]!r} loaded; the request asks for {job.get(key)!r}.")
    results = [check(state["llm"], item) for item in job["items"]]
    state["requests"] += 1
    output = {
        "prompt_version": PROMPT_VERSION, "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "model_file": record["model_file"], "model_sha256": record["model_sha256"],
        "projector_file": record["projector_file"], "projector_sha256": record["projector_sha256"],
        "runtime": record["runtime"], "threads": record["threads"],
        # The model stays loaded; its one-time load and hash times are in "worker".
        "load_seconds": 0.0, "results": results, "temperature": 0.0, "seed": SEED,
        "worker": workers.provenance(record, state["requests"]),
    }
    Path(request["output"]).write_text(json.dumps(output, indent=2), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    return workers.serve(lambda: load(args), answer)


if __name__ == "__main__":
    sys.exit(main())
