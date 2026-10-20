# Compact detector v2 (not adopted)

**v2 did not fix the parked-forklift problem in routine footage, so v1 remains the Fast search model.** Pair recall on the primary clips rose slightly (0.8644 to 0.8983) and pair precision fell (0.9808 to 0.9464); in several routine clips v2 boxed shelving or pillars as forklifts.

v2 is a revision made after we had seen the v1 evaluation and routine results. It reuses the same evaluation clips, so it is a regression comparison, not a new independent test. The confidence threshold and best epoch were chosen on the run-separated validation split only and frozen before the v2 evaluation; nothing was tuned on the evaluation results. No HEF was compiled.

## Same scoring scope as the other profiles

Same 9 scorable runs / 90 sampled frames, same scorer, boxes, IoU and pair definition. The coverage exclusion is unchanged (`052961d2a2e7eeaf2652_run_5_seed_1856081553`: 300 video frames but 198 annotation rows).

| Model | Pair TP / FP / FN | Pair P / R | Person P / R | Forklift P / R |
|---|---|---|---|---|
| Broader (Grounding DINO) | 55 / 6 / 4 | 0.9016 / 0.9322 | 0.9884 / 0.9551 | 0.9677 / 1.0000 |
| Stricter (Grounding DINO) | 37 / 4 / 22 | 0.9024 / 0.6271 | 0.9881 / 0.9326 | 1.0000 / 0.8000 |
| YOLOX-tiny v1, 1 fps | 51 / 1 / 8 | 0.9808 / 0.8644 | 1.0000 / 0.9326 | 1.0000 / 1.0000 |
| YOLOX-tiny v2, 1 fps | 53 / 3 / 6 | 0.9464 / 0.8983 | 0.9655 / 0.9438 | 1.0000 / 1.0000 |

## Training data, validation and freeze

We removed the 21 routine box-pickup training runs and kept the original near-miss allocation of 34 training and 9 validation runs. One validation run failed the annotation integrity check, leaving 34 training and 8 validation runs. Each run contributes its `ceiling_00` view plus two eye-level views; no new runs were added. That gives 2,040 training frames and 480 validation frames. Overlap with any evaluation run: 0. Details: `annotation_integrity.json`, `camera_selection.json`.

v2 was retrained from the official Apache-2.0 YOLOX-tiny COCO weights, not from the v1 checkpoint, with the same settings as v1: 416 input, AdamW, batch 8, 8 CPU threads, fixed seed, no mosaic or mixup. Frozen 2026-10-12T09:33:26Z; best epoch 10; confidence 0.20; aggregate validation recall ≥ 0.9 for both classes.

| Validation view | Class | TP / FP / FN | Precision | Recall |
|---|---|---|---|---|
| ceiling | person | 152 / 2 / 7 | 0.9870 | 0.9560 |
| ceiling | forklift | 160 / 0 / 0 | 1.0000 | 1.0000 |
| non-ceiling | person | 242 / 23 / 36 | 0.9132 | 0.8705 |
| non-ceiling | forklift | 293 / 16 / 12 | 0.9482 | 0.9607 |

Per-view results use the single frozen threshold; no per-view thresholds were set.

## Routine clips: first-frame check

The ten routine clips produced 11 retained candidate groups. Routine annotations omit parked forklifts, so no routine false-alarm rate or forklift recall can be computed. The notes below are a qualitative check of the first frame of each clip after the freeze.

| Run prefix | Groups | Pair-candidate frames | First-frame observation |
|---|---|---|---|
| 01644a25617e0dd8ebb3 | 0 | 2 / 14 | Several person boxes on shelving and barrels, no person visible |
| 0179f236dd63ebbdf964 | 2 | 9 / 14 | Parked forklift boxed as person (0.72); forklift boxes (0.30, 0.31) on a wide shelving area, not on the vehicle |
| 017e84541eeadcaf7ba1 | 1 | 13 / 14 | Person boxed; two forklift boxes on shelving or pillars, no vehicle under either box |
| 018c5222a72e2bff8e86 | 1 | 2 / 14 | Person boxed; foreground shelving on the left boxed as forklift |
| 018cd2d26ab89c5ad9b2 | 1 | 13 / 14 | Person boxed; shelving on both sides boxed as forklift |
| 018ce6edabcf479d76d2 | 1 | 14 / 14 | Parked forklift boxed (0.46), but a pillar boxed as forklift (0.89) and the vehicle's rear boxed as person (0.66) |
| 01a8b535022f10dbf3b8 | 1 | 4 / 14 | A bottle boxed as person (0.86); several forklift boxes on shelving or pillars |
| 01db36b56c4daa26452e | 2 | 10 / 14 | Parked forklift still not boxed; shelving on the left and right boxed as forklift (0.52, 0.67); person boxed |
| 01fb00c3126ef31816d1 | 1 | 14 / 14 | Person boxed; forklift box on shelving at left; a partly out-of-frame object at right boxed, identity unclear |
| 02204d8ac901f56538dc | 1 | 9 / 14 | Several forklift boxes on stacked cartons or shelving; no person box |

## CPU timing (median, seconds)

| Cohort | Wall incl. model load | Wall excl. load | Model load | ONNX Runtime ms per frame |
|---|---|---|---|---|
| Primary | 0.728 | 0.619 | 0.112 | 20.168 |
| Routine | 0.962 | 0.856 | 0.107 | 19.901 |

## Files

`FREEZE.json` (weights and ONNX hashes, settings), `best_validation.json`, `camera_selection.json`, `annotation_integrity.json`, `comparison.json`, `evaluation_fps1.json` and `predictions/`. Data come from NVIDIA PhysicalAI SDG-Warehouse, dataset revision `d5b88d3abcf659f304a107f4336b71b4e2159133`, under OpenMDW-1.1. Architecture, preprocessing and the raw two-class output match v1, so the Hailo comparison in `../v1/HAILO_COMPATIBILITY.md` also applies.
