"""The Qwen2.5-1.5B-Instruct HEF as the UGen300 draft model (HailoLLM), on the fake hailo_platform.

HailoRT's genai.LLM takes plain-text chat messages and, left to its HEF, decodes with frequency_penalty 1.1
(data/evaluation/stage2/device/draft-llm/defaults.json); the CPU draft worker applies no repetition penalty.
"""
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from tracebrief import device_pipeline, drafting, hailo_backend as hb

EVENT = {"id": "event-001", "observations": [
    {"id": "event-001-obs-1", "time": 5.0, "text": "At 5s: Predicted person and forklift boxes are close but do not overlap in the image."},
    {"id": "event-001-obs-2", "time": 8.0, "text": "At 8s: Predicted person and forklift boxes overlap in the image."}],
    "uncertainties": ["Predicted object labels and boxes require visual verification."],
    "visual_check": {"status": "confirmed", "time": 8.0, "description": "A person stands beside a forklift.", "forklift": "yes", "person": "yes"}}


@pytest.fixture
def llm(fake_hailo, tmp_path):
    fake = fake_hailo()
    hef = tmp_path / "llm.hef"; hef.write_bytes(b"hef")
    return fake, hb.HailoLLM(hb.shared_vdevice(fake), hef)


def test_the_llm_drafts_with_the_cpu_workers_settings_in_plain_text_messages(llm):
    fake, model = llm
    result = model.draft(EVENT)
    system, user = drafting.prompt(EVENT)
    system = f"{system} {drafting.DEVICE_FORMAT}"
    call, = fake.llms[0].calls
    assert call["prompt"] == [{"role": "system", "content": system}, {"role": "user", "content": user}]
    assert (call["do_sample"], call["seed"], call["max_generated_tokens"]) == (False, 19, 320)
    assert call["frequency_penalty"] == hb.LLM_FREQUENCY_PENALTY == 1.0 and call["temperature"] is None
    assert fake.llms[0].events == ["clear", "generate", "clear"]  # a fresh context for every draft
    assert result["prompt_sha256"] == hashlib.sha256((system + user).encode()).hexdigest()
    assert result["prompt_version"] == drafting.DEVICE_PROMPT_VERSION and result["stopped_at_object_end"] is True
    assert json.loads(result["text"])["evidence_ids"] == ["event-001-obs-1", "event-001-obs-2"]
    model.close()
    assert fake.llms[0].released and fake.genai == [("load", "llm"), ("release", "llm")]


def test_the_llm_leaves_the_stream_at_the_end_of_the_json_object(llm):
    fake, model = llm
    answer = {"title": "Person beside forklift", "summary": "x }", "review_questions": ["Is another camera view available?"],
              "evidence_ids": ["event-001-obs-1"]}
    reply = json.dumps(answer) + "\n\nNote: the boxes are predicted. {\"extra\": 1}"
    fake.reply = lambda prompt, frames: reply
    result = model.draft(EVENT)
    assert json.loads(result["text"]) == answer and result["stopped_at_object_end"] is True
    assert result["tokens"] < len(reply.split())
    assert fake.llms[0].events[-2:] == ["generate", "clear"]


def test_a_stricter_device_format_can_be_chosen_and_an_unknown_one_is_refused(fake_hailo, tmp_path):
    fake = fake_hailo()
    hef = tmp_path / "llm.hef"; hef.write_bytes(b"hef")
    vdevice = hb.shared_vdevice(fake)
    model = hb.HailoLLM(vdevice, hef, device_format="draft-v2-device1")
    assert model.draft(EVENT)["prompt_version"] == "draft-v2-device1"
    assert fake.llms[0].calls[0]["prompt"][0]["content"].endswith(drafting.DEVICE_FORMAT_1)
    with pytest.raises(hb.DeviceError, match="Unknown device draft format"):
        hb.HailoLLM(vdevice, hef, device_format="draft-v9")


def test_the_llm_takes_turns_with_the_detector(fake_hailo, tmp_path):
    np = pytest.importorskip("numpy")
    fake = fake_hailo(queue=8)
    fake.delay = 0.002
    hef = tmp_path / "model.hef"; hef.write_bytes(b"hef")
    vdevice, turns = hb.shared_vdevice(fake), hb.Turns()
    detector = hb.HailoDetector(vdevice, hef, batch_size=8, turns=turns)
    model = hb.HailoLLM(vdevice, hef, turns=turns)
    worker = threading.Thread(target=lambda: [model.draft(EVENT) for _ in range(3)])
    worker.start()
    while worker.is_alive():
        detector.detect_batch([(np.zeros((416, 416, 3), np.uint8), 320, 240)] * 16)
    worker.join()
    assert fake.overlaps == [] and turns.switches >= 1


def test_a_device_draft_by_the_llm_is_recorded_as_such(llm, tmp_path):
    fake, model = llm
    runtime = SimpleNamespace(draft_model="llm", llm=model, version="5.3.2",
                              models={"llm": {"file": "Qwen2.5-1.5B-Instruct.hef", "sha256": "ab" * 32}})
    audit = device_pipeline.generate(runtime, EVENT, tmp_path)
    assert audit["method"] == "ugen300-qwen2.5-draft" and audit["validation_errors"] == []
    raw = audit["model_output"]
    assert raw["model_file"] == "Qwen2.5-1.5B-Instruct.hef" and raw["model_sha256"] == "ab" * 32
    assert raw["frequency_penalty"] == 1.0 and raw["decoding"] == "greedy (do_sample=False)" and raw["seed"] == 19
    assert json.loads((tmp_path / "generation-event-001-input.json").read_text(encoding="utf-8"))["model"] == "Qwen2.5-1.5B-Instruct.hef"
    vlm_runtime = SimpleNamespace(vlm=SimpleNamespace(draft=lambda event: {"text": "no object", "seconds": 1.0}), version="5.3.2",
                                  models={"vlm": {"file": "Qwen2-VL-2B-Instruct.hef", "sha256": "cd" * 32}})
    audit = device_pipeline.generate(vlm_runtime, EVENT, tmp_path)  # a runtime without draft_model drafts with the VLM
    assert audit["method"] == "fixed-template-after-local-rejection" and audit["model_output"]["model_file"] == "Qwen2-VL-2B-Instruct.hef"
    assert "frequency_penalty" not in audit["model_output"]
