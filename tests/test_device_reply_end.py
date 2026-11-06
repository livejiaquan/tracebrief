"""The UGen300 VLM stops reading tokens once its JSON answer is complete.

On the device (HailoRT 5.3.2, Qwen2-VL-2B HEF) 15 of the 148 Stage I look-v2 answers went on after the closing
brace ("Please note that the description is based on ...") until the token limit: 38 s per crop instead of 14 s,
for text the parser never reads (data/evaluation/stage2/device/look-agreement). Leaving the stream there ended the
generation within half a second on the device, with the same answer, and the next check was unchanged.
"""
import json

import pytest

from tracebrief import hailo_backend as hb, look_prompt

ANSWER = {"forklift": "yes", "person": "yes", "description": "A person stands beside a forklift."}
TRAILER = " \n\nPlease note that the description is based on the visible elements in the image. It does not add anything."


@pytest.fixture
def vlm(fake_hailo, tmp_path):
    fake = fake_hailo()
    hef = tmp_path / "vlm.hef"; hef.write_bytes(b"hef")
    return fake, hb.HailoVLM(hb.shared_vdevice(fake), hef)


def test_the_check_leaves_the_stream_at_the_end_of_the_json_answer(vlm):
    Image = pytest.importorskip("PIL.Image")
    fake, model = vlm
    reply = json.dumps(ANSWER, indent=2) + TRAILER
    fake.reply = lambda prompt, frames: reply
    entry = model.check(Image.new("RGB", (64, 48)))
    assert entry["answer"] == look_prompt.parse(reply) == {**ANSWER}
    assert entry["raw_text"] == json.dumps(ANSWER, indent=2)
    assert entry["stopped_at_object_end"] is True
    assert entry["tokens"] < len(reply.split())
    assert fake.vlms[0].events[-2:] == ["generate", "clear"]  # the context is still cleared after an early stop


def test_a_brace_inside_a_string_does_not_end_the_answer(vlm):
    fake, model = vlm
    answer = {"title": "Odd } title { here", "summary": "Quote \\\" and } inside.", "review_questions": ["Is it {ok}?"],
              "evidence_ids": ["event-001-obs-1"]}
    reply = json.dumps(answer) + " Trailing words."
    fake.reply = lambda prompt, frames: reply
    result = model.generate([{"role": "user", "content": [{"type": "text", "text": "x"}]}], [], hb.DRAFT_TOKENS)
    assert json.loads(result["text"]) == answer and result["stopped_at_object_end"] is True


@pytest.mark.parametrize("reply", [
    '{"title": "x", "summary": "never closed", "review_questions": ["Is it? Is it? Is it?',  # a loop cut by the token limit
    "No JSON object at all.",
    '```json\n{"forklift": "yes", "person": "no", "description": "x"}\n```',  # the fence after the brace is not read
])
def test_a_reply_without_a_trailer_is_kept_up_to_its_answer(vlm, reply):
    fake, model = vlm
    fake.reply = lambda prompt, frames: reply
    result = model.generate([{"role": "user", "content": [{"type": "text", "text": "x"}]}], [], hb.DRAFT_TOKENS)
    end = hb.json_object_end(reply)
    assert result["text"] == (reply if end is None else reply[:end])
    assert result["stopped_at_object_end"] is (end is not None)


@pytest.mark.parametrize("text, end", [
    ('{"a": 1}', 8), ('x {"a": {"b": 2}} y', 17), ('{"a": "}"} z', 10), ('{"a": "\\"}"}', 12), ("{", None), ("none", None),
])
def test_json_object_end(text, end):
    assert hb.json_object_end(text) == end


def test_the_parsed_answer_is_the_full_replys_whenever_that_one_parses():
    """Cutting at the first complete object never changes an answer the frozen parser could read from the whole reply."""
    for reply in (json.dumps(ANSWER) + TRAILER, "```json\n" + json.dumps(ANSWER) + "\n```", json.dumps(ANSWER)):
        assert look_prompt.parse(reply[:hb.json_object_end(reply)]) == look_prompt.parse(reply)
