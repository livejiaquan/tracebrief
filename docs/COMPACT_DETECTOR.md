# Compact detector (Fast search)

**Summary.** Since 2026-11-05 Fast search runs **detector v4**, a YOLOX-s trained on synthetic and real warehouse images that passed all <!-- fact: v4.criteria -->7 of 7 criteria registered before its training ([Detector v4](#detector-v4)). The sections before it describe the **Stage I model**, which stays selectable as **Fast search (Stage I detector)** so the Stage I numbers can be reproduced. That purpose-trained YOLOX-tiny for people and forklifts showed about <!-- fact: fast.speedup -->90 times lower observed vision wall time than Grounding DINO on the same shared CPU host (medians of end-to-end runs that each include loading the model; not a controlled benchmark), with higher pair precision and somewhat lower recall, on the same synthetic test clips. Its forklift-labeled training views were overhead (ceiling) views, and it misses parked forklifts seen from ground-level cameras. Both are optional profiles; **Broader search** remains the default.

## Why

Grounding DINO has no verified Hailo deployment path in this prototype and took <!-- fact: stricter.wall_s -->37–<!-- fact: broader.wall_s -->71 seconds (median) of vision wall time on CPU per 10-second clip. YOLOX-tiny is published for Hailo-10H (the chip in ASUS UGen300) in the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_object_detection.rst), so a person/forklift YOLOX-tiny has a direct Stage II compile path. This Stage I model was only parsed for Hailo, never compiled; see `data/evaluation/compact/v1/HAILO_COMPATIBILITY.md`. Detector v4, its successor, was compiled and run on the UGen300 ([HARDWARE_PLAN.md](HARDWARE_PLAN.md)).

## How the Stage I model was built

- Official [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) tiny COCO checkpoint, fine-tuned on CPU at 416×416.
- Training data: 64 new NVIDIA synthetic runs (43 near-miss ceiling views, 21 routine box-pickup ground views) that do not overlap any evaluation run; split 51 train / 13 validation **by run**. Overlap audit: 0.
- Epoch and confidence (0.40) were chosen on validation only and frozen (`FREEZE.json`, with weight hashes) before any evaluation clip was processed.

## Stage I results on the same test clips

Same 9 scorable runs / 90 sampled frames, same scorer and coverage exclusion as the Broader/Stricter comparison. A pair counts only if both predicted boxes match annotated boxes (IoU ≥ 0.5).

| Profile | Pair TP / FP / FN | Pair precision | Pair recall | Median vision time per 10 s clip |
|---|---|---|---|---|
| Broader (Grounding DINO) | <!-- fact: broader.pair_counts -->55 / 6 / 4 | <!-- fact: broader.pair_precision -->90.2% | <!-- fact: broader.pair_recall -->93.2% | <!-- fact: broader.wall_s -->71.3 s |
| Stricter (Grounding DINO) | <!-- fact: stricter.pair_counts -->37 / 4 / 22 | <!-- fact: stricter.pair_precision -->90.2% | <!-- fact: stricter.pair_recall -->62.7% | <!-- fact: stricter.wall_s -->36.6 s |
| **Fast, Stage I model (YOLOX-tiny v1)** | <!-- fact: fast.pair_counts -->**51 / 1 / 8** | <!-- fact: fast.pair_precision -->**98.1%** | <!-- fact: fast.pair_recall -->**86.4%** | <!-- fact: fast.wall_s -->**0.79 s** end to end (inference <!-- fact: fast.inference_ms -->≈20 ms per frame) |

Times are medians over all 10 test clips (including the one excluded from scoring) on the same shared host, end to end; Grounding DINO timings include model loading, the compact timing includes its <!-- fact: fast.load_s -->0.1 s load. These are batch CPU measurements, not UGen300 results. All profiles sample <!-- fact: config.sample_fps -->1 frame per second (`sample_fps` in `config/vision*.json`, as frozen). For the Stage I Fast model, inference is only <!-- fact: fast.inference_s -->0.2 s of the 0.79 s (about <!-- fact: fast.inference_fps -->49 frames/s on <!-- fact: config.threads -->4 threads, inference only); <!-- fact: fast.other_s -->0.48 s is spent outside the model, mostly decoding every frame of the 30 fps clip to sample one per second. The freeze also planned a 5 fps run as a sensitivity check (`compact/v1/evaluation_fps5.json`): scored on the same gold frames it gave the same pair counts (<!-- fact: fast.fps5_pair_counts -->51 / 1 / 8) and took <!-- fact: fast.fps5_wall_s -->1.95 s per clip. Sampling at 5 fps would make brief interactions between 1 Hz samples less likely to be missed, but that benefit is not measured here and 5 fps is not a shipped setting.

End to end in the app with the Stage I model (recorded demonstration job `f1a1dc63da35460e`, `warehouse-1` sample, 2026-10-20; summary in `data/evaluation/timing-jobs/`): 30.4 s from video to draft, of which detection took 0.8 s, the Qwen2-VL-2B visual check 13.4 s on 8 threads and local drafting most of the rest. Before the visual check was added (job `3ee3d1fe1f11435f`, 2026-10-12) the same clip took 19.0 s. On CPU the language models, not the detector, are the bottleneck; Stage II moves the detector, the visual check and the draft onto UGen300 ([HARDWARE_PLAN.md](HARDWARE_PLAN.md)).

## Limits found by inspection (Stage I model)

On ten routine ground-level clips the detector produced no candidate groups, but visual inspection showed it **did not box clearly visible parked forklifts in at least three clips** and produced two low-confidence person boxes on objects. Likely causes: forklift labels came only from overhead near-miss views, while the ground-level routine training clips supplied person labels but omit parked forklifts, which taught the model to treat them as background.

A revision (v2) removed those routine clips and added two ground-level views per near-miss run. It raised test recall to 89.8% (53 / 3 / 6) but reintroduced shelves and pillars boxed as forklifts in routine clips, so it was **not adopted**. v2 was a revision made after seeing v1 results, not an independent test. Records: `data/evaluation/compact/v2/`.

Deployment implication: a site pilot should include a short calibration step on the site's own camera view (a few dozen labeled clips) before relying on Fast search; until then Broader search remains the default for uncalibrated camera views; real-site accuracy is still to be tested. Detector v4 below is the first test on real images; it does not replace that pilot.

## Detector v4

Records: [`data/evaluation/stage2/detector-v4/`](../data/evaluation/stage2/detector-v4/README.md). Settings `config/vision-compact-v4.json`, weights `weights/yolox-s-warehouse-v4.onnx` (SHA-256 `fdcc92d1…f05be2`).

**What changed.**

- **Real images in training.** Besides the SDG synthetic runs: NVIDIA's synthetic SmartSpaces training buildings and two real-image sets, the LOCO training sites and the keremberke training split (Roboflow forklift-dsitv). LOCO has no person class, so its persons were pseudo-labeled by the official YOLOX-L COCO model, with uncertain boxes masked.
- **YOLOX-s instead of YOLOX-tiny**, still at 416×416. Epoch and confidence (0.25, the lowest allowed) were chosen on validation only. A YOLOX-tiny arm, v4-T (confidence 0.45), was trained alongside under the same rules.
- **Same input and output** as the Stage I model (`images` 1×3×416×416 → `raw_predictions` 1×3549×7, opset 11), so the app runs it through the unchanged, frozen `compact_vision.py`.

**Protocol.** Data, splits, criteria and the decision rule were registered in `PREREG.json` (2026-11-04 18:59:55 UTC) before any training step and before any new test-set inference. Each model was scored once per test set after its freeze, and no threshold, epoch or rule was changed after a test result.

| Pre-registered criterion | v4-T (YOLOX-tiny, 0.45) | v4-S (YOLOX-s, 0.25) |
|---|---|---|
| C1a SDG Stage I frames: pair precision ≥ 93.08%, recall ≥ 81.44% | pass | pass |
| C1b SDG held-out events: ≥ 24 of 33 found (strict) | pass | pass |
| C2a LOCO held-out site: forklift AP50 and recall ≥ Stage I model + 0.20, precision ≥ 0.50 | **fail**: AP50 / recall / precision <!-- fact: v4.tiny_loco -->0.271 / 0.154 / 0.744 | pass |
| C2b keremberke test: forklift AP50 + 0.20, person AP50 + 0.10, pair recall + 0.20 over the Stage I model | pass | pass |
| C3 SmartSpaces (re-used test): within registered tolerances of v3 (forklifts found −0.03, person precision and recall −0.05, strict alerts −2, at most 12 false alerts against v3's 0) | pass (1 false alert) | pass (8 false alerts) |
| C4 LOCO held-out site: false forklifts on pallet trucks and stillages ≤ Stage I model + 3 points | pass | pass |
| Export: ONNX input/output, PyTorch parity, Hailo end-node check | pass | pass |
| **Verdict** | not adopted (6 of 7) | **adopted** (<!-- fact: v4.criteria -->7 of 7) |

v4-T's recall interval reaches just above the margin, but the registered test is on the point estimate. Its frozen confidence (0.45) is probably too high for real forklifts; choosing a lower one now would use test data.

**Stage I model against v4-S.** Each model at its frozen confidence (Stage I 0.40, v4-S 0.25). The Stage I model reproduces its published SDG numbers exactly in this harness.

| Test set and measure | Stage I model (YOLOX-tiny) | v4-S (YOLOX-s), Fast search |
|---|---|---|
| SDG Stage I test frames: pair precision / recall (TP / FP / FN) | <!-- fact: fast.pair_precision -->98.1% / <!-- fact: fast.pair_recall -->86.4% (<!-- fact: fast.pair_counts -->51 / 1 / 8) | <!-- fact: v4.pair_precision -->96.4% / <!-- fact: v4.pair_recall -->89.8% (<!-- fact: v4.pair_counts -->53 / 2 / 6) |
| SDG held-out events found, strict | <!-- fact: v4.old_events_strict -->26 of 33 | <!-- fact: v4.events_strict -->27 of 33 |
| LOCO held-out real site (873 images, 188 forklifts): forklift AP50; precision / recall | <!-- fact: v4.old_loco -->0.0001; 0.00 / 0.00 | <!-- fact: v4.loco_ap50 -->**0.337**; <!-- fact: v4.loco_pr -->0.67 / **0.34** |
| keremberke test (42 images): forklift / person AP50; pair recall (25 images) | <!-- fact: v4.old_kb -->0.000 / 0.087; 0.00 | <!-- fact: v4.kb_ap50 -->**0.894 / 0.511**; <!-- fact: v4.kb_pair_recall -->**0.68** (17 of 25) |
| SmartSpaces, **re-used test, 3rd use**: forklift found per camera | <!-- fact: v4.old_ss_found -->0.122 | <!-- fact: v4.ss_found -->0.332 (v3: <!-- fact: v4.v3_ss_found -->0.273) |
| SmartSpaces, re-used test, 3rd use: person precision / recall | <!-- fact: v4.old_ss_person -->0.675 / 0.254 | <!-- fact: v4.ss_person -->0.821 / 0.565 |
| SmartSpaces, re-used test, 3rd use: strict / false alerts (61 events) | <!-- fact: v4.old_ss_alerts -->8 / 1 | <!-- fact: v4.ss_alerts -->21 / 8 |
| SmartSpaces, re-used test, 3rd use: false person boxes on humanoid robots | <!-- fact: v4.old_ss_humanoid_fp -->18047 | <!-- fact: v4.ss_humanoid_fp -->4997 (v3: <!-- fact: v4.v3_ss_humanoid_fp -->507) |
| CPU, ONNX Runtime 4 threads: ms per frame (two passes) | <!-- fact: v4.old_cpu_ms -->22.4 / 25.9 | <!-- fact: v4.cpu_ms -->40.5 / 37.3 |
| App, bundled 10 s clips: end to end; inference per frame (median of 6 runs) | <!-- fact: v4.app_stage1_wall_s -->0.94 s; 26.9 ms | <!-- fact: v4.app_wall_s -->1.09 s; 40.9 ms |

The SmartSpaces scenes were used by the pilot, by v3 and now by v4, so they are not fresh evidence; v3 was a study model, never shipped. In a full CPU job on `warehouse-1` (visual check and draft included, `app_check.json`) both Fast profiles gave one candidate, confirmed by the visual check, in 23–30 s once the models were in the disk cache (the very first job, with a cold cache, took 75 s); the language models take almost all of that time.

**Weaknesses.**

- **Real-forklift recall is low.** At the unseen LOCO site v4-S finds about a third of the forklifts (v4-T 15%), and on keremberke it finds half of the persons (recall 0.50); persons there are mostly drivers, partly hidden by the cab and mast.
- **Small and partly used real tests.** The keremberke test has 42 images, 25 with both classes; the pair-recall 95% interval is 0.48–0.87. LOCO subset 3 had been scored once before, by v3, without tuning. These are the only real-image tests.
- **More false persons than v3.** False person boxes on humanoid robots are about ten times v3's in SmartSpaces (far fewer than the Stage I model's), false persons on pallet trucks rose too, and v4-S raises 8 false alerts (the registered limit is 12) and 27 false app-native candidates against v3's 2.
- **Cost and device.** About 1.6 times the Stage I model's CPU time per frame (about 25 frames/s at 4 threads, ample for 1 frame/s sampling). Its HEF on the UGen300 kept the ONNX pair precision and lost one pair of recall on the Stage I frames (<!-- fact: device.pair_precision -->96.3% / <!-- fact: device.pair_recall -->88.1%) and ran at <!-- fact: device.detector_fps_b8 -->202–205 FPS at batch 8, measured on a USB port that supplied only 0.9 A (reduced-performance mode); see [HARDWARE_PLAN.md](HARDWARE_PLAN.md).
- **Frozen warning text.** Every compact detection record still carries the frozen `compact_vision.py` warning "Synthetic-domain validation only; real warehouse generalization unverified." The file is frozen, so the text was not changed; for v4 read it with the real-image results above.

**Deviations from the protocol** (`RESULTS.md` §8):

- Before registration: the first person-teacher run used PyTorch's BatchNorm eps (1e-5) instead of YOLOX's (1e-3) and was discarded and re-run; a code review also fixed the training defaults (14 epochs and 6.0 h to 12 epochs and 6.5 h), an epoch-ranking tie-break and a key error in `tables.py`.
- After registration, 4.5 h before the first v4 test result: `tables.py` was fixed for a crash found in a dry run on the Stage I and v3 baselines. The criteria logic is unchanged, but its SHA-256 differs from `PREREG.json`.
- After registration: the SmartSpaces runner's self-check now skips boxes whose score is rounded onto the threshold (stored as 0.2500), and the scorer counts them as detections: 26 of 202,603 v4-S and 13 of 171,589 v4-T SmartSpaces detections, none on LOCO or keremberke. Only the interrupted video was re-run.
- Speed only: the v4-S SmartSpaces run used 2 workers × 4 threads instead of 1 × 4; the Stage I model and v3 ran with 1 thread on LOCO and keremberke while training ran; training used 11 threads and the teacher and the final SmartSpaces evaluation 12, on 8 physical cores, so the arms slowed each other.
- Code added after registration was for presentation only (`report_tables.py`) plus one exploratory latency check outside the criteria; two stray `.pyc` files from a read-only review's imports were never used.
- `server.py` changed after v3 but not during the study; all four models ran with the same file. When v4-S was adopted, `server.py` gained only the check that keeps the Stage I profile out of device mode.
- Data limits seen during the work: LOCO leaves some forklifts unlabeled (false negatives in training), LOCO test persons are not scored, and the selection rule let an early epoch with only 5 LOCO validation forklift detections block later ones (it did not change the final pick).

**UGen300.** `scripts/hailo/compile_yolox_hef.py` defaults to `weights/yolox-s-warehouse-v4.onnx` and `nms_yolox_warehouse.json` with `nms_scores_th` 0.25. The ONNX end-node check passes with the same nine end nodes for both models, whose graphs differ only in channel width. Detector v4 has since been parsed, optimized and compiled with Dataflow Compiler 5.3.0 and run on the UGen300; the build we use, o2s, passed the acceptance rule ([HARDWARE_PLAN.md, Detector: ONNX to HEF](HARDWARE_PLAN.md#detector-onnx-to-hef); records in [stage2/device/hef-o2s](../data/evaluation/stage2/device/hef-o2s/)). `config/hailo-manifest.json` names `models/hailo/yolox_s_warehouse_v4.hef` without a hash, because a rebuild hashes differently; record your build's hash in a local manifest. The Stage I config is kept as `scripts/hailo/nms_yolox_tiny_stage1.json`. Hailo's published `yolox_tiny` speed does not carry over to YOLOX-s.

**Reproducing Stage I.** `config/vision-compact.json` and `weights/yolox-tiny-warehouse.onnx` are unchanged and still match `compact/v1/FREEZE.json` (checked by `tests/test_facts.py`). In the app choose **Fast search (Stage I detector)**, or send `"search_profile": "stage1"` to `/api/jobs`; `scripts/compact/run_evaluation.py` reads the frozen Stage I settings directly. The Stage I profile is CPU only: device mode does not offer it, because the detector HEF is compiled from v4-S.

Training data licenses and the CC BY 4.0 attribution lines: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
