"""Resident CPU workers: real worker processes driven with a stub llama_cpp (tests/conftest.py).

The stub's reply digests the prompt, the decoding settings and the model state, so equal replies mean
equal requests to the model. A real-model comparison runs only with TRACEBRIEF_TEST_MODELS=1.
"""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tracebrief import cpu_pipeline, generation, look, workers

ROOT = Path(__file__).resolve().parents[1]
EVENT = {"id": "event-001", "uncertainties": ["No physical distance has been established."],
         "observations": [{"id": "event-001-obs-1", "time": 2.0, "text": "At 2s: Predicted person and forklift boxes overlap in the image."},
                          {"id": "event-001-obs-2", "time": 3.0, "text": "At 3s: Predicted person and forklift boxes are close but do not overlap in the image."}],
         "visual_check": {"status": "confirmed", "time": 2.0, "description": "A worker walks beside a forklift in the aisle."}}


@pytest.fixture
def models(tmp_path, monkeypatch, stub_llama):
    """Stand-in GGUF files and the environment the server passes to the model steps."""
    files = {}
    for key, name in (("TRACEBRIEF_VLM_MODEL", "vlm.gguf"), ("TRACEBRIEF_VLM_PROJECTOR", "mmproj.gguf"), ("TRACEBRIEF_LLM_MODEL", "llm.gguf")):
        path = tmp_path / "models" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode() * 1000)
        monkeypatch.setenv(key, str(path))
        files[key] = path
    monkeypatch.setenv("TRACEBRIEF_LLM_PYTHON", sys.executable)
    monkeypatch.delenv("TRACEBRIEF_VLM_PYTHON", raising=False)
    monkeypatch.delenv("TRACEBRIEF_CPU_WORKERS", raising=False)
    return files


def crop(tmp_path, name="crop.jpg", content=b"fixture crop bytes"):
    path = tmp_path / "frames" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(content)
    return path


def job_dir(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    return path


def test_look_worker_answers_every_request_like_a_freshly_loaded_model(tmp_path, models):
    image = crop(tmp_path)
    first = cpu_pipeline.check_items([{"id": "event-001", "image": str(image)}], ROOT, job_dir(tmp_path, "a"), "event-001", 4)
    second = cpu_pipeline.check_items([{"id": "event-001", "image": str(image)}], ROOT, job_dir(tmp_path, "b"), "event-001", 4)
    worker = workers.WORKERS["look"]
    assert worker.starts == 1 and worker.requests == 2  # loaded once
    assert first["results"][0]["raw_text"] == second["results"][0]["raw_text"]  # reset and reseeded per request
    assert first["results"][0]["answer"]["forklift"] == "yes"
    assert first["model_sha256"] == workers.sha256(models["TRACEBRIEF_VLM_MODEL"])
    assert first["prompt_version"] == "look-v2" and first["worker"]["request"] == 1 and second["worker"]["request"] == 2
    # What the library prints goes to the request's log, never into the protocol.
    log = (tmp_path / "b" / "look-event-001.log").read_text(encoding="utf-8")
    assert "C-level stdout while answering" in log and "stderr while answering" in log


def test_draft_worker_drafts_are_independent_of_earlier_candidates(tmp_path, models):
    directories = [job_dir(tmp_path, name) for name in ("a", "b", "c")]
    other = {**EVENT, "id": "event-002", "observations": [{**EVENT["observations"][0], "id": "event-002-obs-1"}]}
    audits = [cpu_pipeline.generate(EVENT, ROOT, directories[0]), cpu_pipeline.generate(other, ROOT, directories[1]),
              cpu_pipeline.generate(EVENT, ROOT, directories[2])]
    assert workers.WORKERS["draft"].starts == 1
    assert [a["method"] for a in audits] == ["local-qwen-draft"] * 3
    assert audits[0]["model_output"]["raw_text"] == audits[2]["model_output"]["raw_text"]
    assert audits[0]["model_output"]["raw_text"] != audits[1]["model_output"]["raw_text"]
    saved = json.loads((directories[0] / "generation-event-001-output.json").read_text(encoding="utf-8"))
    assert saved["prompt_version"] == "draft-v2" and saved["model_sha256"] == workers.sha256(models["TRACEBRIEF_LLM_MODEL"])
    assert (directories[0] / "generation-event-001-input.json").is_file() and (directories[0] / "generation-event-001.log").is_file()


@pytest.mark.skipif(sys.version_info < (3, 11), reason="the frozen per-call scripts use hashlib.file_digest (Python 3.11+)")
def test_workers_give_the_records_of_the_frozen_per_call_scripts(tmp_path, models):
    image = crop(tmp_path)
    item = [{"id": "event-001", "image": str(image)}]
    resident = cpu_pipeline.check_items(item, ROOT, job_dir(tmp_path, "resident-look"), "event-001", 4)
    per_call = look.check_items(item, ROOT, job_dir(tmp_path, "per-call-look"), "event-001", 4)
    for key in ("prompt_version", "prompt_sha256", "model_file", "model_sha256", "projector_file", "projector_sha256",
                "runtime", "threads", "temperature", "seed"):
        assert resident[key] == per_call[key], key
    for key in ("id", "raw_text", "usage", "answer"):
        assert resident["results"][0][key] == per_call["results"][0][key], key
    resident = cpu_pipeline.generate(EVENT, ROOT, job_dir(tmp_path, "resident-draft"))
    per_call = generation.generate(EVENT, ROOT, job_dir(tmp_path, "per-call-draft"))
    assert resident["accepted"] == per_call["accepted"] and resident["method"] == per_call["method"] == "local-qwen-draft"
    for key in ("draft", "raw_text", "model_file", "model_sha256", "runtime", "prompt_sha256", "prompt_version", "usage", "seed", "temperature"):
        assert resident["model_output"][key] == per_call["model_output"][key], key


def test_worker_that_cannot_start_falls_back_to_the_per_call_script(tmp_path, models, monkeypatch):
    monkeypatch.setenv("STUB_LLAMA_FAIL", "1")
    calls = []
    monkeypatch.setattr(look, "check_items", lambda *args: calls.append(args) or {"per_call": True})
    item = [{"id": "event-001", "image": str(crop(tmp_path))}]
    assert cpu_pipeline.check_items(item, ROOT, job_dir(tmp_path, "a"), "event-001", 4) == {"per_call": True}
    worker = workers.WORKERS["look"]
    assert worker.starts == 0 and "stub load failure" in worker.failed[1]
    started = []
    monkeypatch.setattr(workers.subprocess, "Popen", lambda *args, **kwargs: started.append(args))
    assert cpu_pipeline.check_items(item, ROOT, job_dir(tmp_path, "b"), "event-001", 4) == {"per_call": True}
    assert started == [] and len(calls) == 2  # no new start attempt within RETRY_AFTER
    log = (tmp_path / "runtime" / "workers" / "look.log").read_text(encoding="utf-8")
    assert "stub load failure" in log


def test_missing_interpreter_falls_back_to_the_per_call_script(tmp_path, models, monkeypatch):
    monkeypatch.setenv("TRACEBRIEF_LLM_PYTHON", str(tmp_path / "no-such-python"))
    monkeypatch.setattr(generation, "run_generate_local", lambda root, request, response, log: {"draft": {"title": "x"}, "per_call": True})
    audit = cpu_pipeline.generate(EVENT, ROOT, job_dir(tmp_path, "a"))
    assert audit["model_output"]["per_call"] is True


def test_crashed_worker_falls_back_and_is_restarted_for_the_next_candidate(tmp_path, models, monkeypatch):
    calls = []
    monkeypatch.setattr(look, "check_items", lambda *args: calls.append(args) or {"per_call": True})
    good, bad = crop(tmp_path), crop(tmp_path, "crash.jpg", b"crash")
    cpu_pipeline.check_items([{"id": "event-001", "image": str(good)}], ROOT, job_dir(tmp_path, "a"), "event-001", 4)
    first = workers.WORKERS["look"].process.pid
    assert cpu_pipeline.check_items([{"id": "event-002", "image": str(bad)}], ROOT, job_dir(tmp_path, "b"), "event-002", 4) == {"per_call": True}
    assert len(calls) == 1 and not workers.WORKERS["look"].alive()
    output = cpu_pipeline.check_items([{"id": "event-003", "image": str(good)}], ROOT, job_dir(tmp_path, "c"), "event-003", 4)
    worker = workers.WORKERS["look"]
    assert worker.starts == 2 and worker.process.pid != first and output["results"][0]["answer"]["person"] == "yes"


def test_stuck_worker_is_killed_and_times_out_like_the_per_call_script(tmp_path, models, monkeypatch):
    monkeypatch.setattr(cpu_pipeline, "LOOK_TIMEOUT", 3)
    with pytest.raises(subprocess.TimeoutExpired):
        cpu_pipeline.check_items([{"id": "event-001", "image": str(crop(tmp_path, "slow.jpg", b"slow"))}], ROOT, job_dir(tmp_path, "a"), "event-001", 4)
    assert not workers.WORKERS["look"].alive()


def test_worker_error_is_a_runtime_error_and_the_draft_falls_back_to_the_template(tmp_path, models):
    other_model = tmp_path / "models" / "other.gguf"
    other_model.write_bytes(b"other")
    request = tmp_path / "input.json"
    request.write_text(json.dumps({"event": EVENT, "model": str(other_model)}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="the request asks for"):
        cpu_pipeline.run_draft_worker(ROOT, request, tmp_path / "output.json", tmp_path / "draft.log")
    assert workers.WORKERS["draft"].alive()  # an answered error does not cost the loaded model


def test_close_all_stops_the_workers(tmp_path, models):
    cpu_pipeline.check_items([{"id": "event-001", "image": str(crop(tmp_path))}], ROOT, job_dir(tmp_path, "a"), "event-001", 4)
    process = workers.WORKERS["look"].process
    workers.close_all()
    assert process.poll() is not None and workers.WORKERS == {}


def test_workers_can_be_switched_off(tmp_path, models, monkeypatch):
    monkeypatch.setenv("TRACEBRIEF_CPU_WORKERS", "0")
    monkeypatch.setattr(look, "check_items", lambda *args: {"per_call": True})
    assert cpu_pipeline.check_items([], ROOT, tmp_path, "event-001", 4) == {"per_call": True}
    assert "look" not in workers.WORKERS


def test_visual_check_record_is_the_frozen_record_on_a_frame_exact_crop(tmp_path, monkeypatch):
    """cpu_pipeline.visual_check repeats look.visual_check step for step; only the model call and the crop differ.

    The crop has the frozen region, filters and quality, but its frame is chosen by number (the
    detector's frame 60 at 2 s, 30 frames/s) instead of the frozen -ss seek (docs/ERRATA.md).
    """
    model = tmp_path / "vlm.gguf"; model.write_bytes(b"x")
    monkeypatch.setenv("TRACEBRIEF_VLM_MODEL", str(model)); monkeypatch.setenv("TRACEBRIEF_VLM_PROJECTOR", str(model))
    from tracebrief import media
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"jpeg")
    monkeypatch.setattr(subprocess, "run", run)
    output = {"seconds": 1.0, "model_file": "vlm.gguf", "model_sha256": "0" * 64, "prompt_version": "look-v2",
              "results": [{"raw_text": "{}", "answer": {"forklift": "yes", "person": "no", "description": "A forklift."}}]}
    monkeypatch.setattr(look, "check_items", lambda *args: output)
    monkeypatch.setattr(cpu_pipeline, "check_items", lambda *args: output)
    event = {"id": "event-001", "peak": 2.0, "vision_record": {"peak_s": 2.0, "evidence": [
        {"time_s": 2.0, "pairs": [{"person_bbox": [10, 10, 20, 40], "forklift_bbox": [30, 10, 60, 50], "gap_px": 10}]}]}}
    frozen = look.visual_check(event, tmp_path / "clip.mp4", (640, 480), tmp_path, "/media/jobs/x/frames", ROOT, tmp_path)
    crop = lambda video, seconds, region, path: media.crop_at(video, seconds, region, path, 30.0)
    resident = cpu_pipeline.visual_check(event, tmp_path / "clip.mp4", (640, 480), tmp_path, "/media/jobs/x/frames", ROOT, tmp_path,
                                         make_crop=crop)
    assert resident == frozen and resident["status"] == "check-suggested"
    seek, exact = commands
    assert seek[seek.index("-ss") + 1] == "2.000" and "-ss" not in exact
    graph = exact[exact.index("-vf") + 1]
    assert graph == "select='eq(n,60)'," + seek[seek.index("-vf") + 1]  # the frozen crop filter, after the frame is chosen
    assert exact[-4:] == seek[-4:] == ["-q:v", "2", "-y", str(tmp_path / "event-001-look.jpg")]


def test_worker_scripts_import_the_frozen_prompts_instead_of_copying_them():
    for script, phrase in (("look_worker.py", "This image is cropped"), ("draft_worker.py", "Write a concise warehouse review")):
        source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert phrase not in source
        imported = {alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom) for alias in node.names}
        assert {"PROMPT", "parse"} <= imported if script == "look_worker.py" else "drafting" in imported


@pytest.mark.skipif(os.environ.get("TRACEBRIEF_TEST_MODELS") != "1", reason="set TRACEBRIEF_TEST_MODELS=1 and the model variables to compare on the real GGUF models")
def test_real_models_answer_the_same_through_the_workers(tmp_path):
    """On the real models: the per-call scripts and the resident workers give identical answers and drafts."""
    from tracebrief import media
    if not look.configured() or not Path(os.environ.get("TRACEBRIEF_LLM_MODEL", "")).is_file():
        pytest.skip("TRACEBRIEF_VLM_MODEL, TRACEBRIEF_VLM_PROJECTOR and TRACEBRIEF_LLM_MODEL must name the GGUF files")
    image = media.crop_at(ROOT / "data" / "samples" / "warehouse-1.mp4", 3.0, [640, 240, 640, 600], tmp_path / "crop.jpg", 30.0)
    item = [{"id": "event-001", "image": str(image)}]
    try:
        per_call = look.check_items(item, ROOT, job_dir(tmp_path, "per-call"), "event-001", 8)
        resident = [cpu_pipeline.check_items(item, ROOT, job_dir(tmp_path, f"resident-{i}"), "event-001", 8) for i in range(2)]
        assert [r["results"][0]["raw_text"] for r in resident] == [per_call["results"][0]["raw_text"]] * 2
        event = {**EVENT, "visual_check": {"status": "confirmed", "time": 3.0, **per_call["results"][0]["answer"]}}
        per_call = generation.generate(event, ROOT, job_dir(tmp_path, "per-call-draft"))
        resident = [cpu_pipeline.generate(event, ROOT, job_dir(tmp_path, f"resident-draft-{i}")) for i in range(2)]
        assert [r["model_output"]["raw_text"] for r in resident] == [per_call["model_output"]["raw_text"]] * 2
    finally:
        workers.close_all()
