"""The look-v2 visual-check prompt and answer parser, without llama.cpp.

scripts/look_local.py is frozen for Stage I and imports llama_cpp at module level, so the
UGen300 path cannot import it. This module carries the same prompt and parser;
tests/test_look_prompt.py reads look_local.py's source and checks both are byte-identical.
"""
from __future__ import annotations

import json
import re

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
