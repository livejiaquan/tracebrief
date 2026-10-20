"""Checks for the visual-check step: crop geometry, pair choice, verdicts and reporting."""
import pytest

from tracebrief.generation import baseline, validate_draft
from tracebrief.look import crop_region, key_pair, verdict, visual_check
from tracebrief.server import export_html


@pytest.fixture
def event():
    return {
        "id": "event-001", "start": 1.0, "end": 4.0, "peak": 3.0, "status": "draft",
        "title": "Unreviewed interaction", "summary": "A candidate for review.",
        "observations": [{"id": "obs-1", "time": 2.0, "text": "At 2s: Predicted person and forklift boxes overlap in the image."}],
        "uncertainties": ["No physical distance has been established."], "history": [],
        "vision_record": {"peak_s": 3.0, "evidence": [
            {"time_s": 1.0, "pairs": []},
            {"time_s": 2.0, "pairs": [
                {"person_bbox": [10, 10, 20, 40], "forklift_bbox": [30, 10, 60, 50], "gap_px": 10},
                {"person_bbox": [100, 10, 110, 40], "forklift_bbox": [112, 10, 150, 50], "gap_px": 2}]},
            {"time_s": 5.0, "pairs": [{"person_bbox": [0, 0, 5, 5], "forklift_bbox": [6, 0, 9, 5], "gap_px": 1}]},
        ]},
    }


def checked(status="confirmed"):
    return {"status": status, "time": 2.0, "forklift": "yes", "person": "yes" if status == "confirmed" else "no",
            "description": "A worker walks beside a forklift in the aisle.", "crop_url": None}


def test_crop_covers_both_boxes_and_stays_inside_the_frame():
    x, y, w, h = crop_region([[100, 50, 200, 300], [180, 120, 600, 500]], 640, 480)
    assert (x, y) == (0, 0)
    assert x + w == 640 and y + h == 480
    x, y, w, h = crop_region([[300, 200, 320, 260], [330, 210, 380, 270]], 1920, 1080)
    assert x <= 300 and y <= 200 and x + w >= 380 and y + h >= 270


def test_key_pair_is_nearest_peak_frame_and_closest_pair(event):
    frame, pair = key_pair(event)
    assert frame["time_s"] == 2.0
    assert pair["gap_px"] == 2


def test_key_pair_is_none_without_predicted_pairs(event):
    event["vision_record"]["evidence"] = [{"time_s": 1.0, "pairs": []}]
    assert key_pair(event) is None


def test_verdict_needs_both_objects():
    assert verdict({"forklift": "yes", "person": "yes"}) == "confirmed"
    assert verdict({"forklift": "no", "person": "yes"}) == "check-suggested"
    assert verdict({"forklift": "yes", "person": "no"}) == "check-suggested"


def test_visual_check_is_unavailable_without_a_model(event, tmp_path, monkeypatch):
    monkeypatch.delenv("TRACEBRIEF_VLM_MODEL", raising=False)
    result = visual_check(event, tmp_path / "missing.mp4", (1920, 1080), tmp_path, "/media/jobs/x/frames", tmp_path, tmp_path)
    assert result["status"] == "unavailable"


def test_template_and_checks_use_the_visual_description(event):
    event["visual_check"] = checked()
    draft = baseline(event)
    assert draft["summary"].startswith("Visual check at 2s: A worker walks")
    assert validate_draft(draft, event) == []


def test_unavailable_visual_check_adds_nothing_to_the_draft(event):
    event["visual_check"] = {"status": "unavailable", "reason": "Visual check model is not configured."}
    assert "Visual check" not in baseline(event)["summary"]


def test_export_reports_visual_check_and_suggested_false_candidate(event):
    event["visual_check"] = checked("check-suggested")
    result = {"title": "Clip", "source_label": "Synthetic", "generated_at": "now", "source_sha256": "0" * 64,
              "hardware_label": "Host CPU", "events": [event]}
    page = export_html(result)
    assert "Visual check" in page and "A worker walks beside a forklift" in page
    assert "Check whether this is a false candidate" in page


def test_export_states_when_visual_check_was_not_run(event):
    result = {"title": "Clip", "source_label": "Synthetic", "generated_at": "now", "source_sha256": "0" * 64,
              "hardware_label": "Host CPU", "events": [event]}
    assert "Unavailable: Not run for this record." in export_html(result)


def test_composed_draft_keeps_detector_timeline_and_rejects_box_titles(event):
    from tracebrief.generation import compose
    event["observations"].append({"id": "obs-2", "time": 3.0, "text": "At 3s: Predicted person and forklift boxes are close but do not overlap in the image."})
    event["visual_check"] = checked()
    draft = {"title": "Boxes overlap at 2s", "summary": "Predicted boxes overlap.", "review_questions": ["Is another view available?"],
             "evidence_ids": ["obs-1", "obs-2"]}
    composed = compose(draft, event)
    assert composed["title"] == baseline(event)["title"]
    assert composed["summary"].startswith("Visual check at 2s: A worker walks")
    assert composed["summary"].endswith("boxes are close at 3s and overlap at 2s.")
    assert validate_draft(composed, event) == []


def test_hailo_nms_output_maps_back_to_source_pixels():
    from tracebrief.hailo_backend import to_source_boxes
    # 1920x1080 letterboxed into 416: ratio 416/1920, so x and y scale by 1920 when normalized to the 416 input.
    person = [[0.10, 0.25, 0.20, 0.30, 0.9], [0.0, 0.0, 0.1, 0.1, 0.2]]
    forklift = [[0.20, 0.40, 0.40, 0.60, 0.8]]
    boxes = to_source_boxes([person, forklift], 1920, 1080)
    assert [b["class_name"] for b in boxes] == ["person", "forklift"]
    assert boxes[0]["bbox_xyxy"] == [480.0, 192.0, 576.0, 384.0]


def test_pad_to_square_keeps_the_whole_crop():
    from PIL import Image
    from tracebrief.hailo_backend import pad_to_square
    square = pad_to_square(Image.new("RGB", (448, 200), (255, 0, 0)))
    assert square.size == (448, 448)
    assert square.getpixel((224, 224)) == (255, 0, 0) and square.getpixel((0, 0)) == (114, 114, 114)
