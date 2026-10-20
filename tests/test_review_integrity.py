"""Checks for failure handling and evidence integrity, not model accuracy."""
import importlib
import json
from pathlib import Path

import pytest

from tracebrief.generation import baseline, validate_draft
from tracebrief.server import export_html, safe_file


@pytest.fixture
def event():
    return {
        "id": "event-001", "start": 1.0, "end": 4.0, "status": "draft",
        "title": "Unreviewed interaction", "summary": "A candidate for review.",
        "observations": [{"id": "obs-1", "time": 2.0, "text": "Predicted person and forklift boxes are close in image space."}],
        "uncertainties": ["No physical distance has been established."], "history": [],
    }


def test_source_checks_reject_invented_citation_and_distance(event):
    draft = {"title": "Interaction", "summary": "The forklift was 2.3 meters away.",
             "review_questions": ["Was follow-up needed?"], "evidence_ids": ["nonexistent"]}
    errors = validate_draft(draft, event)
    assert len(errors) == 2


def test_template_remains_grounded_but_unapproved(event):
    draft = baseline(event)
    assert validate_draft(draft, event) == []
    assert draft["evidence_ids"] == ["obs-1"]
    assert event["status"] == "draft"


def test_declarative_review_questions_are_rejected(event):
    draft = baseline(event)
    draft["review_questions"] = ["Another camera view is available."]
    assert "not declarations" in validate_draft(draft, event)[0]


def test_review_questions_cannot_presume_harm_or_add_numbers(event):
    draft = baseline(event)
    draft["review_questions"] = ["Who caused the injury?"]
    assert any("unsupported consequence" in e for e in validate_draft(draft, event))
    draft["review_questions"] = ["Was the person within 1.5 meters?"]
    assert any("numbers absent" in e for e in validate_draft(draft, event))


def test_open_questions_about_contact_are_allowed(event):
    draft = baseline(event)
    draft["summary"] = "Boxes are close in image space; whether a collision or near miss occurred needs review."
    draft["review_questions"] = ["Does another view show whether a near miss occurred?"]
    assert validate_draft(draft, event) == []
    draft["summary"] = "The forklift collided with the person."
    assert validate_draft(draft, event)


def test_html_export_escapes_review_text_and_keeps_unknowns(event):
    event["summary"] = '<script>alert("x")</script>'
    result = {"title": "Sample", "events": [event], "source_label": "Synthetic",
              "generated_at": "2026-10-04", "source_sha256": "a"*64, "hardware_label": "CPU"}
    document = export_html(result)
    assert '<script>' not in document
    assert '&lt;script&gt;' in document
    assert "No physical distance has been established." in document
    assert "draft" in document
    assert "a"*64 in document


def test_static_files_cannot_escape_root(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    secret = tmp_path / "private.txt"
    secret.write_text("private")
    with pytest.raises(FileNotFoundError):
        safe_file(public, "../private.txt")
    (public / "escape").symlink_to(secret)
    with pytest.raises(FileNotFoundError):
        safe_file(public, "escape")


def test_empty_result_export_does_not_claim_safety():
    result = {"title": "Empty", "events": [], "source_label": "Synthetic",
              "generated_at": "2026-10-04", "source_sha256": "a"*64, "hardware_label": "CPU"}
    assert "does not establish absence of risk" in export_html(result)


def test_shipped_profiles_point_to_frozen_compact_weights():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    profiles = json.loads((root/'config'/'search_profiles.json').read_text())
    assert profiles['default'] == 'broader'
    assert [p['id'] for p in profiles['profiles']] == ['broader', 'fast', 'stricter']
    fast = next(p for p in profiles['profiles'] if p['id'] == 'fast')
    settings = json.loads((root/'config'/fast['settings_file']).read_text())
    assert settings['backend'] == 'compact-onnx'
    weights = root/settings['model_path']
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == settings['model_sha256']
