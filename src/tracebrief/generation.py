"""Local generation transport and conservative draft checks.

The checks reject malformed citations and unsupported numbers; they do not prove
semantic truth. All drafts remain subject to reviewer confirmation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


# Conclusions the detector cannot support: contact, harm, blame or a safety verdict.
# Mentioning these topics as open questions is allowed; asserting them is not.
UNSUPPORTED = re.compile(
    r"\b(caused|negligent|at fault|injured|injury|collided|struck by|was hit|were hit|collision confirmed|"
    r"violation confirmed|safe to proceed)\b", re.I)
# Questions may ask about contact or a near miss, but must not presume blame or harm.
PRESUMPTIVE = re.compile(r"\b(caused|negligent|at fault|injured|injury|safe to proceed)\b", re.I)
# Every number, with or without a unit suffix: "97s", "12m" and "3.4 seconds" are all checked.
NUMBER = re.compile(r"(?<![\d.])(?:\d+(?:\.\d+)?|\.\d+)")
# Kana and CJK ideographs are written without spaces, so each character counts as a term.
CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
TERM = re.compile(rf"[{CJK}]|[^\W\d_{CJK}]+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])\s*")


def numbers(text: str) -> set[float]:
    """The values of the numbers in text, so "2" and "2.0" are the same number."""
    return {float(n) for n in NUMBER.findall(text)}


def terms(text: str) -> set[str]:
    """Lower-case words, and single CJK characters, for the visual-check coverage test."""
    return set(TERM.findall(text.lower()))


def visual_text(event: dict) -> str:
    """The visual-check sentence the draft may use, or an empty string."""
    check = event.get("visual_check") or {}
    if check.get("status") not in ("confirmed", "check-suggested"):
        return ""
    return f"Visual check at {check['time']:g}s: {check['description']}"


def timeline(event: dict) -> str:
    """One sentence restating the detector record verbatim, so a model draft cannot drop a timestamp."""
    close, overlap = [], []
    for o in event.get("observations", []):
        (overlap if "boxes overlap" in o["text"] else close).append(f"{o['time']:g}s")
    join = lambda times: times[0] if len(times) == 1 else ", ".join(times[:-1]) + " and " + times[-1]
    parts = ([f"are close at {join(close)}"] if close else []) + ([f"overlap at {join(overlap)}"] if overlap else [])
    return f"Detector record: predicted person and forklift boxes {' and '.join(parts)}." if parts else ""


def compose(draft: dict, event: dict) -> dict:
    """Keep the model's wording of the scene; restate the visual check and detector timeline as recorded.

    Box relations always come from the detector record, never from the model's paraphrase, and the
    visual-check sentence is added verbatim when the model left it out.
    """
    composed = dict(draft)
    title = re.sub(r"\s+at\s+\d+(?:\.\d+)?\s*s\b", "", draft["title"]).strip()
    composed["title"] = baseline(event)["title"] if not title or re.search(r"\b(box|boxes|overlap\w*)\b", title, re.I) else title
    sentences = [s for s in SENTENCE_END.split(draft["summary"].strip()) if s and not re.search(r"\bboxes\b", s, re.I)]
    visual = visual_text(event)
    if visual:
        description = terms(visual.split(":", 1)[1])
        covered = any(len(description & terms(s)) >= 0.6 * len(description) for s in sentences)
        if not covered:
            sentences.insert(0, visual)
    composed["summary"] = " ".join(sentences + [timeline(event)]).strip()
    return composed


def baseline(event: dict) -> dict:
    observations = event.get("observations", [])
    parts = [visual_text(event)] + list(dict.fromkeys(o["text"] for o in observations))
    return {
        "title": "Person–forklift interaction for review",
        "summary": " ".join(p for p in parts if p) or "No supported observations are available.",
        "review_questions": [
            "Does the original footage support each observation?",
            "Does this interaction require follow-up under the site's procedures?",
        ],
        "evidence_ids": [o["id"] for o in observations],
    }


def validate_draft(draft: dict, event: dict) -> list[str]:
    errors = []
    if not isinstance(draft, dict):
        return ["The model did not return an object."]
    for key in ("title", "summary"):
        if not isinstance(draft.get(key), str) or not draft[key].strip():
            errors.append(f"Missing {key}.")
    refs = draft.get("evidence_ids", [])
    valid = {o["id"] for o in event.get("observations", [])}
    if not isinstance(refs, list) or not refs or any(r not in valid for r in refs):
        errors.append("Evidence references do not match the observation record.")
    questions = draft.get("review_questions")
    if not isinstance(questions, list) or not 1 <= len(questions) <= 4 or any(not isinstance(q, str) or not q.strip().endswith("?") for q in questions):
        errors.append("Review questions must be a short list of questions, not declarations.")
    factual_text = " ".join(str(draft.get(k, "")) for k in ("title", "summary"))
    source_text = " ".join([o["text"] for o in event.get("observations", [])] + [visual_text(event)])
    allowed_numbers = numbers(source_text)
    if numbers(factual_text) - allowed_numbers:
        errors.append("The draft introduced numbers absent from the observations.")
    question_text = " ".join(q for q in questions if isinstance(q, str)) if isinstance(questions, list) else ""
    if numbers(question_text) - allowed_numbers:
        errors.append("A review question introduced numbers absent from the observations.")
    if UNSUPPORTED.search(factual_text) or PRESUMPTIVE.search(question_text):
        errors.append("The draft inferred an unsupported consequence or responsibility.")
    return errors


def run_generate_local(root: Path, request: Path, response: Path, log: Path) -> dict:
    """scripts/generate_local.py in a fresh process, the Stage I per-call path; returns its output record."""
    interpreter = os.environ.get("TRACEBRIEF_LLM_PYTHON", sys.executable)
    process = subprocess.run(
        [interpreter, str(root / "scripts" / "generate_local.py"), str(request), str(response)],
        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240,
        env={**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"},
    )
    log.write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"Local generation exited with code {process.returncode}; see run log.")
    return json.loads(response.read_text(encoding="utf-8"))


def generate(event: dict, root: Path, output_dir: Path, run=None) -> dict:
    """Draft one candidate and check it; any failure leaves the labeled fixed template.

    run(root, request, response, log) returns the model's output record: run_generate_local by
    default, or the resident draft worker (cpu_pipeline.run_draft_worker).
    """
    started = time.monotonic()
    base = baseline(event)
    model = os.environ.get("TRACEBRIEF_LLM_MODEL")
    audit = {"baseline": base, "model_output": None, "validation_errors": [], "method": "fixed-template", "seconds": 0.0}
    if not model or not Path(model).is_file():
        audit["warning"] = "Local language model is not configured. Fixed-template draft shown."
        audit["accepted"] = base
        return audit
    request = output_dir / f"generation-{event['id']}-input.json"
    response = output_dir / f"generation-{event['id']}-output.json"
    log = output_dir / f"generation-{event['id']}.log"
    try:
        # Inside the try: text that cannot be written (an encoding error) gives the template, not a failed job.
        request.write_text(json.dumps({"event": event, "model": model}, ensure_ascii=False, indent=2), encoding="utf-8")
        raw = (run or run_generate_local)(root, request, response, log)
        draft = raw["draft"]
        errors = validate_draft(draft, event)
        audit.update(model_output=raw, validation_errors=errors)
        if errors:
            audit["warning"] = "Local draft failed automatic source checks. Fixed-template draft shown for review."
            audit["accepted"] = base
            audit["method"] = "fixed-template-after-local-rejection"
        else:
            audit["accepted"] = compose(draft, event)
            audit["method"] = "local-qwen-draft"
            audit["warning"] = "Local AI draft. Citation and number checks do not establish factual correctness; review the footage."
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        audit["accepted"] = base
        audit["warning"] = f"Local generation unavailable: {exc} Fixed-template draft shown."
        audit["validation_errors"] = [str(exc)]
    audit["seconds"] = round(time.monotonic() - started, 3)
    return audit
