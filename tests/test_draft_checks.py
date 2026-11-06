"""Source checks and composition of drafts: numbers with units, CJK visual text, write failures."""
import pytest

from tracebrief import generation
from tracebrief.generation import baseline, compose, numbers, validate_draft


@pytest.fixture
def event():
    return {"id": "event-001", "uncertainties": ["No physical distance has been established."],
            "observations": [{"id": "obs-1", "time": 3.4, "text": "At 3.4s: Predicted person and forklift boxes overlap in the image."}],
            "visual_check": {"status": "confirmed", "time": 2.0, "description": "A worker walks beside a forklift in the aisle."}}


def draft(summary, questions=("Is another camera view available?",)):
    return {"title": "Person beside forklift", "summary": summary, "review_questions": list(questions), "evidence_ids": ["obs-1"]}


@pytest.mark.parametrize("summary", ["The forklift stopped for 97s.", "The person stood 12m from the forklift.",
                                     "Boxes overlap at 3.4s and again at 5s.", "The forklift moved .5 meters."])
def test_numbers_with_units_or_leading_points_are_checked(event, summary):
    assert "The draft introduced numbers absent from the observations." in validate_draft(draft(summary), event)


@pytest.mark.parametrize("summary", ["Boxes overlap at 3.4 seconds.", "Boxes overlap at 3.4s.", "At 3.40s the boxes overlap.",
                                     "The visual check at 2.0 seconds shows a worker beside a forklift."])
def test_supported_numbers_pass_with_or_without_units(event, summary):
    assert validate_draft(draft(summary), event) == []


def test_review_question_numbers_are_checked_with_units(event):
    errors = validate_draft(draft("Boxes overlap at 3.4s.", ["Was the person within 3m?"]), event)
    assert errors == ["A review question introduced numbers absent from the observations."]


def test_numbers_compare_by_value_and_ignore_version_like_tails():
    assert numbers("At 2s, 2.0s and 02 s") == {2.0}
    assert numbers("event-001-obs-12") == {1.0, 12.0}
    assert numbers("release 1.2.3") == {1.2}


def test_cjk_visual_text_is_added_when_the_draft_leaves_it_out(event):
    event["visual_check"]["description"] = "一名工人站在堆高機旁邊的走道上。"
    composed = compose(draft("The detector reports a candidate for review."), event)
    assert composed["summary"].startswith("Visual check at 2s: 一名工人站在堆高機旁邊的走道上。")


def test_cjk_visual_text_is_not_repeated_when_the_draft_covers_it(event):
    event["visual_check"]["description"] = "一名工人站在堆高機旁邊的走道上。"
    composed = compose(draft("第2秒的視覺檢查：工人站在堆高機旁邊的走道上。其他內容。"), event)
    assert composed["summary"].count("堆高機") == 1


def test_english_coverage_is_unchanged(event):
    covered = compose(draft("A worker walks beside a forklift in the aisle."), event)
    assert covered["summary"].count("A worker walks beside a forklift") == 1
    missing = compose(draft("A candidate needs review."), event)
    assert missing["summary"].startswith("Visual check at 2s: A worker walks beside a forklift in the aisle.")


def test_an_input_that_cannot_be_written_gives_the_template(event, tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"; model.write_bytes(b"x")
    monkeypatch.setenv("TRACEBRIEF_LLM_MODEL", str(model))
    event["observations"][0]["text"] = "At 3.4s: \ud800 unpaired surrogate"  # not encodable as UTF-8
    started = []
    audit = generation.generate(event, tmp_path, tmp_path, run=lambda *args: started.append(args))
    assert started == [] and audit["accepted"] == baseline(event) and audit["method"] == "fixed-template"
    assert "Local generation unavailable" in audit["warning"] and audit["validation_errors"]
