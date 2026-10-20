# RGB candidate discovery

TraceBrief reads a full input video locally, samples frames at 1 Hz, detects people and forklifts from pixels, and groups image-space proximity observations into review candidates. The predictor has no annotation input. Simulator boxes are used only by a separate evaluation script.

This is an **offline batch prototype**. It does not establish physical distance, collision, a near miss, unsafe conduct, or a safe scene. No candidate means no candidate was found, never a safety clearance. A reviewer must inspect the original video.

## Run locally

From the `tracebrief/` project directory, use Python 3.11 or newer in a virtual environment:

```bash
python -m venv .venv-vision
.venv-vision/bin/pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.venv-vision/bin/pip install -r scripts/vision_requirements.txt
.venv-vision/bin/python scripts/vision_download.py --output models/grounding-dino-tiny
export TRACEBRIEF_MODEL_DIR="$PWD/models/grounding-dino-tiny"
.venv-vision/bin/python scripts/vision_analyze.py data/samples/YOUR_VIDEO.mp4 --output .runtime/analysis.json
```

The download is explicit and requires network access. Subsequent inference uses `local_files_only=True` and needs no network. CPU execution is limited to at most four PyTorch and OpenCV threads. The server can set `TRACEBRIEF_VISION_PYTHON` to this environment's Python executable.

Python interface:

```python
import json
from pathlib import Path
from tracebrief.vision import analyze_video
settings = json.loads(Path("config/vision.json").read_text())
result = analyze_video("clip.mp4", {**settings, "output_dir": "output/frames"})
```

The returned schema includes video SHA-256 and duration; model ID/revision; effective settings; every sampled observation including no detections; all candidate groups; timestamped JPEG evidence; wall time and explicit limitations. Boxes are absolute pixel `xyxy`. Exported JPEGs contain predicted boxes, never simulator ground truth.

## Detector choice

The default is [IDEA Research Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny), pinned to revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`. The official model card identifies Apache-2.0. Its text prompt is exactly `a person. a forklift.`. Model weights are downloaded separately with a SHA-256 manifest.

A development-only comparison used 12 frames from three previously explored NVIDIA synthetic runs. OWLv2 ensemble took 8.27 seconds per frame on this CPU probe, missed the forklift on two frames at confidence 0.10, and produced shelf-edge false detections. Grounding DINO took 7.12 seconds per frame in the same probe and produced person and forklift detections on all 12 frames, with one additional detection at threshold 0.20. These tiny development results selected a model; they are not final accuracy claims. COCO-only models do not contain a forklift class. Dedicated keremberke forklift weights were considered but not downloaded or selected because their public model card did not state a license.

Grounding DINO CPU execution is a Stage I feasibility path, not an asserted Hailo-compatible detector. Stage II requires a separate compatible detector/compilation strategy and accelerator measurements. No UGen300 performance is measured here.

## Fixed search profiles

| Profile | Confidence | Processor resize | Tradeoff |
|---|---:|---|---|
| **Broader search — default (original v1)** | 0.35 | Default 800/1333 | Retains more candidates; original routine evaluation exposed shelf errors and more review work. |
| Stricter search (frozen v2) | 0.65 | 512/853 | Faster batch processing and fewer routine candidates in the observed cohort; materially lower primary recall. |

Broader search is the application and CLI default, selected after the explicit same-input comparison because it retained higher recall with similar pair precision. This post-result product decision is not a new heldout validation.

These are two existing fixed configurations, not a continuously tuned risk score. Neither provides safety clearance. Profile-specific evidence and the post-result comparison are reported in `VISION_RESULTS.md`. The application offers both profiles; CLI settings can be selected with `--settings config/vision-broader.json` or `--settings config/vision-stricter.json`.

## Shared candidate rule and stricter settings

- Full video sequential decoding; 1 sampled frame per second.
- Grounding DINO box confidence 0.65; text threshold 0.25.
- Processor resize request: shortest edge 512, longest edge 853. A 1920×1080 input becomes 853×480 because the longest-edge bound applies.
- Per-class NMS at IoU 0.45.
- A person/forklift pair is a candidate observation when the edge-to-edge 2D box gap is at most 0.30 times the predicted person's box height.
- Group observations separated by at most 2.1 seconds; require at least two sampled observations. Export a one-second margin around each group.
- Person/forklift identities are not tracked across frames; multiple nearby pairs can form a shared review interval. Inspect evidence to distinguish actors.

The CLI default is `config/vision.json` (Broader); the application uses the same fixed Broader profile as its default. The frozen Stricter configuration remains `config/vision-stricter.json` and is archived with its original hash in the prospective evaluation bundle. Direct Python callers must pass that configuration explicitly; omitted settings preserve the original baseline defaults for backward compatibility.

These are image-plane review prioritization rules. Perspective, occlusion, small people, object confusion and 1 Hz sampling can cause errors. The threshold is not a distance in meters and is not a safety rule.

## Evaluation protocol

Three previously inspected runs are development only. The original configuration (confidence 0.35, default 800/1333 resize) was frozen before evaluating 24 independent ceiling-camera runs, followed by ten routine box-pickup videos. Routine shelf errors motivated a revision. These original 34 runs subsequently became calibration evidence; their original frozen results remain separately reported.

The revised configuration uses confidence 0.65 and resize 512/853, selected using original development frames and previously evaluated routine frames. The resize initially reintroduced shelf errors at 0.60; the 0.65 adjustment removed those sampled errors but also lost one routine person detection. This history is retained in `data/evaluation/calibration/`.

A prospective second evaluation selects ten new ceiling-camera and ten new routine runs, excluding all previously inspected runs. Exact settings, implementation/configuration hashes and run IDs were frozen before any new-run inference or frame inspection. No settings are tuned against the second evaluation. Primary image-space metrics and routine candidate rates are reported separately; missing stationary-forklift annotations prevent routine specificity claims.

A separate evaluator maps `Worker` annotations to `person` and `forklift` annotations to `forklift`. It measures class localization at IoU 0.50 and sampled-frame candidate agreement against the same 2D proximity rule computed from annotation boxes. This is an explicitly limited proxy, not a physical near-miss benchmark. Ground-truth boxes can be conservative or occlusion-dependent. Same-scenario far-apart frames are hard negatives within a run; they are not independent negative videos. The previous requirement of ten positive plus ten independent hard-negative videos remains unproven.

Dataset origin: [NVIDIA PhysicalAI WorldModel Synthetic Warehouse Operations Scenes](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes), synthetic RGB and annotations under OpenMDW-1.1. Model selection and evaluation are entirely local. No human-subject data is collected.

Frozen-run metrics are in [VISION_RESULTS.md](VISION_RESULTS.md).
