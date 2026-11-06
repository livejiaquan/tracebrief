"""The device draft uses the CPU worker's draft-v2 messages and the same source checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import types

import pytest

from tracebrief import device_pipeline, drafting

ROOT = Path(__file__).resolve().parents[1]
# SHA-256 of the draft-v2 system message the CPU worker has used since 2026-10-20.
SYSTEM_SHA256 = "9744b0d8540508ef8f88cc777f75cc28194af0e187338f6cef4bc3888a5fe0a8"


def event(with_check=True):
    value = {"id": "event-001", "uncertainties": ["No physical distance has been established."],
             "observations": [{"id": "event-001-obs-1", "time": 2.0, "text": "At 2s: Predicted person and forklift boxes overlap in the image."},
                              {"id": "event-001-obs-2", "time": 3.0, "text": "At 3s: Predicted person and forklift boxes are close but do not overlap in the image."}]}
    if with_check:
        value["visual_check"] = {"status": "confirmed", "time": 2.0, "description": "A worker walks beside a forklift in the aisle."}
    return value


def run_generate_local(tmp_path, monkeypatch, request_event):
    """scripts/generate_local.py with a stub llama_cpp that records what it is asked."""
    captured = {}

    class Llama:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def create_chat_completion(self, messages, grammar, **kwargs):
            captured["messages"] = messages
            return {"choices": [{"message": {"content": json.dumps({"title": "x", "summary": "x", "review_questions": ["x?"],
                                                                       "evidence_ids": ["event-001-obs-1"]})}}], "usage": {}}

    class LlamaGrammar:
        @staticmethod
        def from_json_schema(schema, verbose=False):
            captured["schema"] = json.loads(schema)

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=Llama, LlamaGrammar=LlamaGrammar))
    spec = importlib.util.spec_from_file_location("generate_local_under_test", ROOT / "scripts" / "generate_local.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.version = lambda name: "stub"
    model = tmp_path / "model.gguf"; model.write_bytes(b"stub")
    request = tmp_path / "input.json"; request.write_text(json.dumps({"event": request_event, "model": str(model)}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["generate_local.py", str(request), str(tmp_path / "output.json")])
    module.main()
    return captured, json.loads((tmp_path / "output.json").read_text(encoding="utf-8"))


@pytest.mark.skipif(sys.version_info < (3, 11), reason="generate_local.py uses hashlib.file_digest (Python 3.11+)")
@pytest.mark.parametrize("with_check", [True, False])
def test_device_messages_equal_the_cpu_worker_messages(tmp_path, monkeypatch, with_check):
    value = event(with_check)
    captured, output = run_generate_local(tmp_path, monkeypatch, value)
    system, user = drafting.prompt(value)
    assert captured["messages"] == [{"role": "system", "content": system}, {"role": "user", "content": user}]
    assert captured["schema"] == drafting.schema(value)
    assert output["prompt_version"] == drafting.PROMPT_VERSION
    assert output["prompt_sha256"] == hashlib.sha256((system + user).encode()).hexdigest()


class Runtime:
    """Just enough of HailoRuntime for device_pipeline.generate."""
    version = "5.3.0"
    models = {"vlm": {"file": "Qwen2-VL-2B-Instruct.hef", "sha256": "0" * 64}}

    def __init__(self, reply):
        self.vlm = types.SimpleNamespace(draft=lambda event: {"text": reply, "tokens": 9, "seconds": 1.0,
                                                              "ttft_s": 0.4, "tokens_per_s": 7.0, "prompt_sha256": "1" * 64})


def test_accepted_device_draft_is_composed_like_a_cpu_draft(tmp_path):
    reply = json.dumps({"title": "Person beside forklift in aisle", "summary": "A worker walks beside a forklift in the aisle.",
                        "review_questions": ["Is another camera view available?"], "evidence_ids": ["event-001-obs-1", "event-001-obs-2"]})
    audit = device_pipeline.generate(Runtime("Here is the draft: " + reply + "<|im_end|>"), event(), tmp_path)
    assert audit["method"] == "ugen300-qwen2-vl-draft" and audit["validation_errors"] == []
    assert audit["accepted"]["summary"].endswith("boxes are close at 3s and overlap at 2s.")
    saved = json.loads((tmp_path / "generation-event-001-output.json").read_text(encoding="utf-8"))
    assert saved["prompt_version"] == "draft-v2-device" and saved["model_sha256"] == "0" * 64


@pytest.mark.parametrize("reply, error", [
    ("The person is near the forklift.", "could not be read"),
    (json.dumps({"title": "x", "summary": "The forklift was 2.3 meters away.", "review_questions": ["Was it close?"],
                 "evidence_ids": ["event-001-obs-1"]}), "numbers absent"),
    (json.dumps({"title": "x", "summary": "x", "review_questions": ["Was it close?"], "evidence_ids": [{"id": "event-001-obs-1"}]}), "could not be read"),
])
def test_unusable_device_draft_falls_back_to_the_labeled_template(tmp_path, reply, error):
    audit = device_pipeline.generate(Runtime(reply), event(), tmp_path)
    assert audit["method"] == "fixed-template-after-local-rejection"
    assert audit["accepted"]["evidence_ids"] == ["event-001-obs-1", "event-001-obs-2"]
    assert any(error in e for e in audit["validation_errors"])


def test_device_format_names_every_schema_key():
    for key in drafting.SCHEMA["required"]:
        assert f'"{key}"' in drafting.DEVICE_FORMAT


@pytest.mark.parametrize("version", sorted(drafting.DEVICE_FORMATS))
def test_every_device_format_names_every_schema_key(version):
    assert drafting.DEVICE_FORMATS[drafting.DEVICE_PROMPT_VERSION] == drafting.DEVICE_FORMAT
    for key in drafting.SCHEMA["required"]:
        assert f'"{key}"' in drafting.DEVICE_FORMATS[version]


@pytest.mark.parametrize("version", sorted(set(drafting.DEVICE_FORMATS) - {"draft-v2-device"}))
def test_stricter_device_formats_hold_no_number_a_draft_could_copy(version):
    """The source checks reject a number that is not in the input, so the stricter hints write their limits in words."""
    assert not re.search(r"\d", drafting.DEVICE_FORMATS[version])


def test_the_cpu_prompt_is_unchanged_by_the_device_formats():
    system, _ = drafting.prompt(event())
    assert drafting.PROMPT_VERSION == "draft-v2" and system == drafting.SYSTEM
    assert hashlib.sha256(drafting.SYSTEM.encode()).hexdigest() == SYSTEM_SHA256


@pytest.mark.parametrize("version", sorted(drafting.DEVICE_FORMATS))
def test_the_device_draft_appends_the_chosen_format_and_records_its_version(fake_hailo, tmp_path, version):
    from tracebrief import hailo_backend as hb
    fake = fake_hailo()
    hef = tmp_path / "vlm.hef"; hef.write_bytes(b"hef")
    vlm = hb.HailoVLM(hb.shared_vdevice(fake), hef, device_format=version)
    runtime = types.SimpleNamespace(vlm=vlm, version="5.3.2", models={"vlm": {"file": "vlm.hef", "sha256": "0" * 64}})
    device_pipeline.generate(runtime, event(), tmp_path)
    system = fake.vlms[0].calls[-1]["prompt"][0]["content"][0]["text"]
    assert system == f"{drafting.prompt(event())[0]} {drafting.DEVICE_FORMATS[version]}"
    saved = json.loads((tmp_path / "generation-event-001-output.json").read_text(encoding="utf-8"))
    assert saved["prompt_version"] == version


def test_an_unknown_device_format_is_refused_before_the_vlm_loads(fake_hailo, tmp_path):
    from tracebrief import hailo_backend as hb
    fake = fake_hailo()
    with pytest.raises(hb.DeviceError, match="Unknown device draft format"):
        hb.HailoVLM(hb.shared_vdevice(fake), tmp_path / "vlm.hef", device_format="draft-v9")
    assert fake.vlms == []
