"""Resident drafting worker: scripts/generate_local.py's model, prompt, grammar and decoding, loaded once.

Usage: python scripts/draft_worker.py --model qwen2.5-1.5b-instruct-q4_k_m.gguf

Started by the server (src/tracebrief/cpu_pipeline.py); see src/tracebrief/workers.py for the
protocol. Each request names an input file of generate_local.py's format and gets generate_local.py's
output record. The draft-v2 messages and schema come from tracebrief.drafting, which tests check
against generate_local.py; the model is reset and reseeded before every draft, so no cached prefix of
an earlier candidate is reused and each draft is the one a fresh generate_local.py process writes.
No network access is used.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tracebrief import drafting, workers  # noqa: E402

SEED = 19
CONTEXT = 2048
THREADS = 4
MAX_TOKENS = 320


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Resident Qwen2.5 drafting worker (JSON lines on stdin/stdout)")
    parser.add_argument("--model", required=True)
    return parser.parse_args(argv)


def load(args):
    from llama_cpp import Llama, LlamaGrammar
    started = time.monotonic()
    llm = Llama(model_path=args.model, n_ctx=CONTEXT, n_threads=THREADS, n_threads_batch=THREADS, seed=SEED, verbose=False)
    loaded = time.monotonic()
    record = {"model": args.model, "model_file": Path(args.model).name, "model_sha256": workers.sha256(args.model),
              "runtime": {"llama-cpp-python": version("llama-cpp-python")},
              "load_seconds": round(loaded - started, 3), "hash_seconds": round(time.monotonic() - loaded, 3),
              "started_at": workers.now()}
    return {"llm": llm, "grammar": LlamaGrammar, "record": record, "requests": 0}, record


def answer(state, request):
    job = json.loads(Path(request["input"]).read_text(encoding="utf-8"))
    record = state["record"]
    if job.get("model") != record["model"]:
        raise ValueError(f"This worker has {record['model']!r} loaded; the request asks for {job.get('model')!r}.")
    event = job["event"]
    system, user = drafting.prompt(event)
    llm = state["llm"]
    llm.reset()
    llm.set_seed(SEED)  # the seed state of a freshly loaded model
    started = time.monotonic()
    result = llm.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        grammar=state["grammar"].from_json_schema(json.dumps(drafting.schema(event)), verbose=False),
        temperature=0.0, max_tokens=MAX_TOKENS,
    )
    raw = result["choices"][0]["message"]["content"]
    state["requests"] += 1
    output = {
        "draft": json.loads(raw), "raw_text": raw,
        "model_file": record["model_file"],
        "model_sha256": record["model_sha256"],
        "runtime": record["runtime"],
        "prompt_sha256": hashlib.sha256((system + user).encode()).hexdigest(), "prompt_version": drafting.PROMPT_VERSION,
        # The model stays loaded; its one-time load and hash times are in "worker".
        "load_seconds": 0.0, "generation_seconds": round(time.monotonic() - started, 3),
        "usage": result.get("usage", {}), "seed": SEED, "temperature": 0.0,
        "worker": workers.provenance(record, state["requests"]),
    }
    Path(request["output"]).write_text(json.dumps(output, indent=2), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    return workers.serve(lambda: load(args), answer)


if __name__ == "__main__":
    sys.exit(main())
