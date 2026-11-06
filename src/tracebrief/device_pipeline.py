"""Job steps of the UGen300 backend: detection, visual check and drafting on the device.

Each step returns the schema of its CPU counterpart (compact_vision.analyze_video, look.visual_check,
generation.generate) and reuses their pure parts. A device failure is raised as DeviceError and fails
the job; no step falls back to a CPU model. A device draft that the source checks reject still gives
way to the labeled fixed template, exactly as on the CPU path. The server runs the steps in phases,
never interleaved: detection of the whole clip, then every visual check, then every draft (server.Phases).
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time

from . import compact_vision, drafting, generation, look, look_prompt, media
from .hailo_backend import DRAFT_TOKENS, INPUT_SIZE, LLM_FREQUENCY_PENALTY, SEED, VLM_SYSTEM, DeviceError, letterbox
from .vision import _pairs

WARNINGS = ['Synthetic-domain validation only; real warehouse generalization unverified.',
            'No detected candidates does not establish safety.', 'Sampling may miss short events.',
            'HEF agreement with the frozen ONNX detector was measured for one build on test frames only (docs/HARDWARE_PLAN.md); '
            'a rebuilt HEF needs its own comparison.']
# The generation_method of an accepted device draft, by the model that wrote it (runtime.draft_model).
DRAFT_METHODS = {"vlm": "ugen300-qwen2-vl-draft", "llm": "ugen300-qwen2.5-draft"}


@contextmanager
def device_step(step: str):
    """Report any failure inside a device call as a DeviceError naming the step."""
    try:
        yield
    except DeviceError:
        raise
    except Exception as exc:
        raise DeviceError(f"UGen300 {step} failed: {type(exc).__name__}: {exc}") from exc


def check_settings(runtime, settings: dict) -> dict:
    if (settings or {}).get('backend') != 'compact-onnx':
        raise ValueError('Only the compact detector (Fast search) runs on the UGen300; this search uses a host-only detector.')
    cfg = {**compact_vision.DEFAULTS, **(settings or {})}
    for field in ['sample_fps', 'confidence', 'gap_ratio', 'merge_gap_s', 'nms_iou']:
        cfg[field] = float(cfg[field])
        if not math.isfinite(cfg[field]):
            raise ValueError(f'Invalid {field}')
    if not 0 < cfg['sample_fps'] <= 30 or cfg['gap_ratio'] < 0 or cfg['merge_gap_s'] < 0:
        raise ValueError('Invalid sampling or gap settings')
    cfg['min_samples'] = int(cfg['min_samples'])
    if cfg['min_samples'] < 1 or int(cfg['input_size']) != INPUT_SIZE:
        raise ValueError('Invalid min_samples or input_size for the UGen300 detector')
    # Thresholds are fixed when the infer model is configured, once per process.
    detector = runtime.detector
    if (cfg['confidence'], cfg['nms_iou']) != (detector.confidence, detector.nms_iou):
        raise ValueError(f"The UGen300 detector is configured for confidence {detector.confidence} and NMS IoU "
                         f"{detector.nms_iou}; this search asks for {cfg['confidence']} and {cfg['nms_iou']}.")
    return cfg


def analyze_video(runtime, path, settings=None) -> dict:
    """compact_vision.analyze_video's observation/event schema, with the YOLOX HEF on the UGen300."""
    import cv2
    cfg = check_settings(runtime, settings)
    cfg['hef'] = runtime.models['detector']['file']
    detector = runtime.detector
    start = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError('Cannot decode video')
    fps = cap.get(cv2.CAP_PROP_FPS); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or total <= 0:
        cap.release()
        raise ValueError('Video has invalid duration')
    stride = max(1, round(fps / cfg['sample_fps']))
    output_dir = Path(cfg.get('output_dir', '.')).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    observations, candidates, pending = [], [], []
    timing = {'inference_s': 0.0}

    def flush():
        canvases = [(letterbox(image)[0], width, height) for _, image in pending]
        tick = time.perf_counter()
        with device_step('detection'):
            results = detector.detect_batch(canvases)
        timing['inference_s'] += time.perf_counter() - tick
        for (index, image), det in zip(pending, results):
            pairs = _pairs(det, cfg['gap_ratio'])
            obs = {'frame_index': index, 'time_s': round(index / fps, 3), 'detections': det, 'proximity_pairs': pairs}
            observations.append(obs)
            if pairs:
                image_path = output_dir / f'frame_{index:06d}.jpg'
                for d in det:
                    x1, y1, x2, y2 = map(int, d['bbox_xyxy']); color = (245, 175, 45) if d['class_name'] == 'person' else (50, 190, 100)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, f"{d['class_name']} {d['confidence']:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
                if not cv2.imwrite(str(image_path), image):
                    raise OSError(f'Cannot write evidence: {image_path}')
                obs['image_path'] = str(image_path); candidates.append(obs)
        pending.clear()

    index = 0
    try:
        while cap.grab():
            if index % stride:
                index += 1; continue
            ok, image = cap.retrieve()
            if not ok:
                raise ValueError(f'Cannot decode frame {index}')
            pending.append((index, image))
            if len(pending) == 2 * detector.group:  # both halves of the binding ring
                flush()
            index += 1
        if pending:
            flush()
    finally:
        cap.release()
    video_digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            video_digest.update(chunk)
    wall = time.perf_counter() - start
    inference_s = timing['inference_s']
    return {'schema_version': '1.0',
            'video': {'path': str(path), 'sha256': video_digest.hexdigest(), 'duration_s': total / fps, 'fps': fps, 'width': width, 'height': height},
            'model': {'id': cfg['model_id'], 'revision': cfg['model_revision'], 'backend': f'YOLOX HEF on Hailo-10H (HailoRT {runtime.version})',
                      'sha256': runtime.models['detector']['sha256'], 'hef': runtime.models['detector']['file'], 'onnx_sha256': cfg.get('model_sha256')},
            'settings': {k: v for k, v in cfg.items() if k != 'output_dir'}, 'observations': observations,
            'events': compact_vision.group_candidates(candidates, total / fps, cfg['merge_gap_s'], cfg['min_samples']),
            # The HEF stays loaded for the life of the process; its load time is in device.load_seconds.
            'timing': {'wall_s': round(wall, 3), 'sampled_frames': len(observations), 'model_load_s': 0.0,
                       'wall_without_model_load_s': round(wall, 6), 'inference_s': round(inference_s, 6),
                       'inference_ms_per_frame': round(inference_s * 1000 / len(observations), 3) if observations else None},
            'device': runtime.record(), 'warnings': list(WARNINGS)}


def visual_check(runtime, event: dict, video: Path, video_size: tuple[int, int], frames_dir: Path, frame_url: str,
                 make_crop=None) -> dict:
    """look.visual_check on the device: the same frame-exact pair crop, prompt, parser and verdict (make_crop as in cpu_pipeline)."""
    from PIL import Image
    chosen = look.key_pair(event)
    if chosen is None:
        return {"status": "unavailable", "reason": "No predicted person-forklift pair to check."}
    frame, pair = chosen
    seconds = float(frame.get("time_s", 0))
    region = look.crop_region([pair["person_bbox"], pair["forklift_bbox"]], *video_size)
    crop = (make_crop or media.crop_at)(video, seconds, region, frames_dir / f"{event['id']}-look.jpg")
    with Image.open(crop) as image, device_step("visual check"):
        result = runtime.vlm.check(image)
    vlm = runtime.models["vlm"]
    record = {"time": seconds, "crop_region": region, "crop_url": f"{frame_url}/{crop.name}",
              "seconds": result["seconds"], "model_file": vlm["file"], "model_sha256": vlm["sha256"],
              "prompt_version": look_prompt.PROMPT_VERSION, "prompt_sha256": hashlib.sha256(look_prompt.PROMPT.encode()).hexdigest(),
              "system_prompt": VLM_SYSTEM, "raw_text": result.get("raw_text"), "runtime": "hailo",
              "runtime_label": "UGen300 Qwen2-VL-2B", "ttft_s": result.get("ttft_s"), "tokens": result.get("tokens"),
              "tokens_per_s": result.get("tokens_per_s"), "stopped_at_object_end": result.get("stopped_at_object_end")}
    if "answer" not in result:
        return {**record, "status": "unavailable", "reason": f"Model answer could not be read: {result.get('error')}"}
    return {**record, **result["answer"], "status": look.verdict(result["answer"])}


def generate(runtime, event: dict, output_dir: Path) -> dict:
    """generation.generate on the device: the draft-v2 prompt answered by the device draft model, then the same checks.

    runtime.draft_model names the model ("vlm": the resident Qwen2-VL-2B in text-only mode, the default; "llm": the
    Qwen2.5-1.5B-Instruct HEF) and runtime.draft(event), when present, drafts with it; otherwise that model's draft().
    """
    started = time.monotonic()
    base = generation.baseline(event)
    audit = {"baseline": base, "model_output": None, "validation_errors": [], "method": "fixed-template", "seconds": 0.0}
    name = getattr(runtime, "draft_model", "vlm")
    model = runtime.models[name]
    (output_dir / f"generation-{event['id']}-input.json").write_text(
        json.dumps({"event": event, "model": model["file"], "runtime": "hailo"}, ensure_ascii=False, indent=2), encoding="utf-8")
    with device_step("drafting"):
        result = (getattr(runtime, "draft", None) or getattr(runtime, name).draft)(event)
    raw = {"raw_text": result["text"], "model_file": model["file"], "model_sha256": model["sha256"],
           "runtime": {"hailort": runtime.version}, "prompt_sha256": result.get("prompt_sha256"),
           "prompt_version": result.get("prompt_version", drafting.DEVICE_PROMPT_VERSION), "generation_seconds": result["seconds"],
           "ttft_s": result.get("ttft_s"), "tokens": result.get("tokens"), "tokens_per_s": result.get("tokens_per_s"),
           "stopped_at_object_end": result.get("stopped_at_object_end"),
           "max_generated_tokens": DRAFT_TOKENS, "seed": SEED, "decoding": "greedy (do_sample=False)"}
    if name == "llm":
        raw["frequency_penalty"] = LLM_FREQUENCY_PENALTY
    try:
        draft = drafting.parse_draft(result["text"])
        raw["draft"] = draft
        errors = generation.validate_draft(draft, event)
    except (ValueError, TypeError) as exc:  # no grammar on the device, so the reply may not be a usable object
        errors = [f"Device draft could not be read: {exc}"]
    (output_dir / f"generation-{event['id']}-output.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    audit.update(model_output=raw, validation_errors=errors)
    if errors:
        audit["warning"] = "Device draft failed automatic source checks. Fixed-template draft shown for review."
        audit["accepted"] = base
        audit["method"] = "fixed-template-after-local-rejection"
    else:
        audit["accepted"] = generation.compose(draft, event)
        audit["method"] = DRAFT_METHODS[name]
        audit["warning"] = "UGen300 draft. Citation and number checks do not establish factual correctness; review the footage."
    audit["seconds"] = round(time.monotonic() - started, 3)
    return audit
