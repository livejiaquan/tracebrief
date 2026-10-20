"""Check candidate crops with a local vision-language model (Qwen2-VL-2B GGUF).

Usage: python scripts/look_local.py input.json output.json

Each input image is a crop around one predicted person-forklift pair, prepared by
the server (see src/tracebrief/look.py). No network access is used; the GGUF model
and its projector must already be present locally.
"""
import base64
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import time

from llama_cpp import Llama
from llama_cpp.llama_chat_format import MTMDChatHandler

PROMPT_VERSION = "look-v2"
PROMPT = (
    "This image is cropped from a warehouse CCTV frame. Answer in JSON with exactly these keys:\n"
    "\"forklift\": \"yes\" if a forklift or similar powered industrial truck is visible, otherwise \"no\".\n"
    "\"person\": \"yes\" if a person is visible, otherwise \"no\".\n"
    "\"description\": one short sentence, at most 20 words, saying where the person is relative to the "
    "forklift and what each is doing. Describe only what is visible."
)
YES_NO = {"yes", "no"}


def parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON object in model output.")
    value = json.loads(match.group(0))
    result = {}
    for key in ("forklift", "person"):
        answer = str(value.get(key, "")).strip().lower()
        if answer not in YES_NO:
            raise ValueError(f"Invalid {key} answer: {answer!r}")
        result[key] = answer
    description = str(value.get("description", "")).strip()
    if not description:
        raise ValueError("Missing description.")
    result["description"] = " ".join(description.split()[:40])
    return result


def sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    threads = int(request.get("threads", 4))
    started = time.monotonic()
    handler = MTMDChatHandler(clip_model_path=request["projector"], verbose=False, use_gpu=False)
    llm = Llama(model_path=request["model"], chat_handler=handler, n_ctx=4096, n_threads=threads,
                n_threads_batch=threads, seed=19, verbose=False)
    loaded = time.monotonic()
    results = []
    for item in request["items"]:
        uri = "data:image/jpeg;base64," + base64.b64encode(Path(item["image"]).read_bytes()).decode()
        tick = time.monotonic()
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": PROMPT}]}],
            temperature=0.0, max_tokens=160)
        raw = response["choices"][0]["message"]["content"]
        entry = {"id": item["id"], "raw_text": raw,
                 "seconds": round(time.monotonic() - tick, 3), "usage": response.get("usage", {})}
        try:
            entry["answer"] = parse(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            entry["error"] = str(exc)
        results.append(entry)
    output = {
        "prompt_version": PROMPT_VERSION, "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "model_file": Path(request["model"]).name, "model_sha256": sha256(request["model"]),
        "projector_file": Path(request["projector"]).name, "projector_sha256": sha256(request["projector"]),
        "runtime": {"llama-cpp-python": version("llama-cpp-python")}, "threads": threads,
        "load_seconds": round(loaded - started, 3), "results": results, "temperature": 0.0, "seed": 19,
    }
    Path(sys.argv[2]).write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
