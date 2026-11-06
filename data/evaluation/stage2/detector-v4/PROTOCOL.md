# Detector v4 (real-image-capable person/forklift detector): pre-registered protocol

Status: written before any v4 training step and before any new test-set inference by any model. The machine-readable copy with timestamps, code hashes and input hashes is `PREREG.json` (its `written_utc` is the registration time). Everything is in `work/stage2/detector-v4/`. Files in `work/stage2/detector-v3/`, `work/stage2/smartspaces/`, `work/stage2/eval-events/` and `work/stage1/` are read, never written. The app (`tracebrief/`) is not modified and its weights are not replaced; adoption is a later decision.

## 1. Question

Can a YOLOX detector in the app's output format (416×416 letterbox, two classes person/forklift, raw YOLOX grids, ONNX opset 11) be trained so that it detects **real** forklifts and persons, without losing accuracy on our SDG-Warehouse held-out data and without becoming worse than v3 on the SmartSpaces scenes? Lessons carried over from v3 (`detector-v3/RESULTS.md`): synthetic-only training gives LOCO forklift AP50 ≈ 0 for old, v3 and v3-A; family-A SmartSpaces forklifts transfer to all test buildings; the v3 selection rule (SDG recall ≥ 0.90) forced v3-A to confidence 0.10 and wrecked precision.

## 2. Models compared

| Label | What | Confidence |
|---|---|---|
| `old` | App model `tracebrief/weights/yolox-tiny-warehouse.onnx` (sha256 `535cbfda…bf1d`), Stage I v1 export, epoch 4 | 0.40 (app Fast profile) |
| `v3` | `detector-v3/freeze/v3/yolox-tiny-warehouse-v3.onnx` (sha256 `0c867580…5afbdf`) | 0.35 (its frozen value) |
| `v4-T` (arm T) | YOLOX-tiny, 416×416, initialised from the official YOLOX-tiny COCO checkpoint (Apache-2.0, sha256 `9de513de…0f8`). Same architecture and ONNX I/O as the app model, so the app loads it unchanged (new `model_sha256` / `confidence`). | chosen on validation |
| `v4-S` (arm S) | YOLOX-s at **416×416** (chosen in advance over 640: it keeps the app's input size and the same 3549-anchor output, so `compact_vision`, the worker and the Hailo NMS layout are unchanged; 640 would need a new input size, NMS JSON and about 2.4× the compute). Initialised from the official YOLOX-s COCO checkpoint (Apache-2.0, sha256 `f55ded71…a30`). | chosen on validation |

Both arms keep the two-class head (COCO person row → person, COCO truck row → forklift at initialisation, as v1–v3) and are exported exactly like v3 (opset 11, input `images` 1×3×416×416, output `raw_predictions` 1×3549×7, `decode_in_inference=False`). For arm S the app needs only a settings file pointing at the new ONNX; the Hailo compile script (`tracebrief/scripts/hailo/compile_yolox_hef.py`) would need its `--onnx` argument, and its HAR names (`yolox_tiny_warehouse`) and `nms_scores_th` (0.40) should be updated to arm S's frozen confidence; whether its parsed HAR layer names match `nms_yolox_warehouse.json` is only known after a DFC parse, which is not run here. Its CPU latency is measured (§7.6) for the CPU fallback.

## 3. Data

### 3.1 SDG-Warehouse (NVIDIA PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes, OpenMDW-1.1)

Unchanged from v3: the v2 COCO frames, read from `detector-v3/data/sdg_coco/` (JSON sha256 identical to `work/stage1/compact-detector/v2/coco/`). 34 train runs / 2,040 frames (1,910 person, 1,970 forklift boxes); 8 val runs / 480 frames (437 person, 465 forklift boxes); `ceiling_00` + eye views, 640×360. The overlap audit is re-run (`code/audit_overlap.py` → `out/overlap_audit.json`) against every held-out, test, routine and development list (`tracebrief/data/evaluation/{sources,freeze,gold}.json`, `phase2/{sources,gold,coverage_exclusions}.json`, `eval-events/inputs/{heldout_runs,dev_runs,inventory}.json` incl. all non-training inventory roles). Training requires overlap 0 and train ∩ val = ∅.

### 3.2 SmartSpaces (`nvidia/PhysicalAI-SmartSpaces`, `MTMC_Tracking_2026`, CC BY 4.0)

Frames are the v3 frames (`detector-v3/data/ss_frames/`, 640×360, same label rule: GT Person → person, GT Forklift → forklift, every other type unlabelled; boxes < 24 px high or < 8 px wide at 1080p ignored and grey-filled). Building families are from v3's metadata check (`detector-v3/out/scene_similarity.json`: map, map transform, camera poses).

| Role | Scenes | Building family | Why |
|---|---|---|---|
| training | 000, 002, 004, 006 (v3 `ss_familyA_train`, every other camera, 3,900 frames, 15,827 person / 3,642 forklift boxes) | A | the only forklift-rich independent-of-test pool; v3-A showed these forklifts transfer to all four test buildings |
| training | 008, 009 (from v3 `ss_strict_train`, 2,400 frames, 11,959 person boxes) | B | persons, humanoids and robots as hard negatives |
| **selection** | 012, 013, 014, 015, 016 (every 5th v3 stored frame per camera = every 450th source frame; 1,000 frames, 6,406 person / 160 forklift boxes) | D | a building in **neither training nor test**; it has persons, humanoids, Transporters and two forklifts (W016) |
| re-used test | 010, 018, 019, 022 | C, E, own | scored with the pilot scorer (§7.3) |
| not used | 001/003/005/007 (re-renders of A, not downloaded), 011 (building of test 010), 017 (building of 018/019), 020/021 (building A; v3's selection scene, now a training building), 023–027 (test split) | | |

Choice of the selection building: family A must be in training (forklift variety), so W020/021 (family A) cannot be the selection scene. Family D is the only remaining building with labelled forklifts that shares no building or camera with a training or test scene (audited by `audit_overlap.py`). Its forklift cell is small (two instances), so selection also relies on SDG, keremberke and LOCO forklifts.

### 3.3 LOCO real images (github.com/tum-fml/loco, CC0-1.0)

The 5,097 annotated images v3 fetched (`detector-v3/data/loco/`, CRC-checked), five subsets. The LOCO authors describe five logistics environments; each subset has its own recording date and camera set, so **subset = site** here.

| Role | Images | Forklift / pallet-truck / stillage boxes |
|---|---|---|
| **test site: subset 3** (2019-12-17; fixed cameras Cam1–3 at 1280×800 plus Kinect/RealSense walks) | 873 | 188 / 308 / 402 |
| validation (selection only): whole sessions 2019-11-25_0 (subset 2), 2020-01-16_16 (subset 4), 2020-03-05_1 (subset 5) | 578 | 80 / 461 / 50 |
| training: everything else in subsets 1, 2, 4, 5 | 3,646 | 330 / 2,058 / 4,955 |

The test site was chosen from GT counts only (most forklifts, and both pallet trucks and stillages for criterion C4). **Prior use of subset 3:** v3 scored old, v3 and v3-A once on all five subsets, including subset 3; no threshold, epoch or rule of v3 was tuned on it, and per-subset LOCO results of v3 were not looked at for this protocol. Subset 3 is therefore "previously evaluated, never trained or tuned on" rather than never seen; no image of it is used for v4 training, selection or teacher pseudo-labelling.

Labels: LOCO forklift → forklift; pallets, pallet trucks, stillages and small load carriers stay unlabelled (hard negatives). LOCO has no person class, so persons are pseudo-labelled by the official **YOLOX-L COCO** checkpoint (Apache-2.0, sha256 `1e6b7fa6…2917`) built as the official YOLOX model builder (BatchNorm eps 1e-3) and run on the full-resolution training- and validation-site images (letterbox long side 960, NMS 0.45): score ≥ **0.60** and ≥ 24 px high / 8 px wide at 1080p scale → person label; score in **[0.25, 0.60)**, or ≥ 0.60 but below the size floor → **ignore region**; < 0.25 → background. Ignore regions are grey-filled in training frames unless the fill would cover > 20% of a kept positive box (then only no label), and listed (not painted) in validation frames. LOCO frames are stored 640 px wide (640×360 for the 1920×1080 images) for training and validation; the test site is scored on the original images.

Audit (`out/overlap_audit.json`): no test-site image in any training/validation pool or teacher input, no session with mixed roles. A 64-bit dHash check flags 97 test images within Hamming distance 8 of some training/validation image; the closest pairs (distance 3) and one at distance 8 are different places with similar aisle layouts (`out/loco_hash_flag_visual_check.jpg`), i.e. hash collisions between different sites, so nothing is removed.

Teacher check on keremberke train / valid GT persons (`out/teacher_check.json`, lower-bound precision because unlabelled persons count as false): at 0.60 precision 0.93 / 0.90, recall 0.74 / 0.69. On the LOCO training and validation sites the teacher gives 1,133 persons ≥ 0.60 and 607 in the ignore band.

Pre-registration note: a first teacher run built YOLOX-L with PyTorch's default BatchNorm eps (1e-5 instead of YOLOX's 1e-3), which shifts its scores; a code review caught it after about half the images, before any use. That output is kept only as `logs/teacher_persons_bn_eps_1e-5_discarded.jsonl`; the teacher was re-run from scratch with the fix (two halves of the item list, 6 threads each, merged by `code/merge_teacher.py`).

### 3.4 keremberke/forklift-object-detection (Hugging Face, revision `2c0f4e8e…`; Roboflow forklift-dsitv v1, CC BY 4.0)

421 web images (classes forklift, person): **train 295** (319 forklift / 296 person boxes) for training, **valid 84** (96 / 75) for selection, **test 42** (43 / 44; 25 images with both classes) untouched until the freeze. Near-duplicate audit (`code/dup_audit.py`, 64-bit dHash incl. flip, Hamming ≤ 8): 0 pairs across splits (minimum test-to-train/valid distance 14); a flagged pair would have removed the train/valid copy. Possible missing person labels: a teacher person ≥ 0.25 matching no GT person (IoU < 0.5) is an ignore region in train (grey fill rule) and valid (listed); test GT is used exactly as published. Images are kept at their own size (mostly 500×375).

### 3.5 Licences

See `DATA_LICENSES.md` (SDG OpenMDW-1.1, SmartSpaces CC BY 4.0, LOCO CC0-1.0, keremberke/Roboflow CC BY 4.0 with attribution, YOLOX code and checkpoints Apache-2.0).

## 4. Training (both arms identical except architecture, initialisation and threads)

Code: `code/train_v4.py`, derived from `detector-v3/code/train_v3.py`.

- An epoch = all 2,040 SDG train frames + 1,500 SmartSpaces training frames (seeded draw without replacement from the 6,300) + 2,000 LOCO training images (all 228 training images that contain a forklift, plus a seeded draw of the 3,418 others) + every keremberke train image twice (590) = 6,130 images. Seeds: 20261105 (draws use 20261105·1000 + epoch).
- AdamW lr 5e-4, weight decay 5e-4, batch 8, one-epoch linear warm-up, cosine to 0.1× over the maximum epochs, EMA 0.99, gradient clip 10, L1 loss from epoch max(5, epochs − 5). Augmentation as v3: scale jitter (p 0.5, s ∈ [0.6, 1.5]) then YOLOX `TrainTransform` (flip 0.5, HSV 0.5); no mosaic or mixup.
- At most 12 epochs; minimum 8; stop after 4 epochs without improvement of the selection rank, or when the next epoch would end after 6.5 h of training wall time.
- CPU only, both arms concurrently: arm T 4 torch threads, arm S 7 threads (11 in total; data loading in each main thread).

## 5. Selection (validation only; precision-aware)

After each epoch the EMA weights are evaluated (PyTorch, objectness × class score, class-wise NMS 0.45, top 200, greedy IoU ≥ 0.5 matching per class, ignore boxes neither TP nor FP) on: SDG v2 val (480), SmartSpaces family D (1,000), keremberke valid (84), LOCO validation sessions (578). For every confidence t ∈ {0.25, 0.30, …, 0.80} (**0.25 is the minimum**):

S(e, t) = mean F1 of seven cells: SDG person, SDG forklift, family-D person, family-D forklift, keremberke person, keremberke forklift, LOCO-val forklift (LOCO-val person uses pseudo-labels and is logged only).

Constraints (all must hold for (e, t) to be feasible):

| Source | Floor |
|---|---|
| SDG val | precision pooled over both classes ≥ 0.90, and recall ≥ 0.85 for each class |
| SmartSpaces family D | precision pooled over both classes ≥ 0.60 |
| keremberke valid | precision pooled over both classes ≥ 0.60 |
| LOCO validation sessions | forklift precision ≥ 0.50 |

Rank of (e, t): (feasible, number of constraints met, S, t); the frozen model is the epoch with the best rank at its best confidence (ties: higher t, then earlier epoch). If nothing is feasible this falls back to the most constraints met, and the shortfall is reported. No test data is touched before the freeze.

## 6. Freeze and export

`code/freeze_export.py` (from v3): checkpoint and ONNX sha256, onnx.checker, I/O identical to the app model, PyTorch/ORT parity (atol = rtol = 1e-3) on 8 validation frames (2 per validation source), app `compact_vision` decode of the exported ONNX, and the Hailo script's `check_onnx_end_nodes` imported read-only (no DFC). Settings files `freeze/<name>/vision-compact-<name>.json` keep every non-model Fast-profile value of the app. `FREEZE.json` records `test_inference_before_freeze: false`.

## 7. Test sets (each model scored once per set after its freeze)

1. **SDG held-out** (C1): app pipeline unmodified (`code/run_app_eval.py`, copied from v3): Stage I frame set (phase-2 primary, 9 runs / 90 frames, `vision_evaluate.py` pair evidence) and 33 event runs (`eval-events/scripts/score_a1.py`, primary matching, called unmodified). Because `tracebrief/src/tracebrief/server.py` changed after v3 (sha256 now `a7a2ce9a…`; only `normalize_vision` is used), old is re-run here before registration as a harness check (must reproduce 51/1/8 and 26/33), and v3 and both v4 arms are run with this same harness after registration.
2. **LOCO held-out site, subset 3** (C2a, C4): `code/run_images.py` (app preprocess/decode, cache ≥ 0.05) and `code/score_images.py` (definitions identical to v3's `score_loco.py`: forklift AP50 VOC all-point at IoU 0.5, P/R at the frozen confidence, false-forklift rate = share of GT pallet-truck / stillage boxes hit at IoU ≥ 0.3 by an unmatched forklift detection ≥ frozen confidence; 2,000-image paired bootstrap vs old). Old and v3 are re-run on these images with the same code.
3. **keremberke test** (C2b): same runner and scorer: person and forklift AP50, P/R at the frozen confidence, pair level (25 images with both classes; detected when ≥ 1 GT person and ≥ 1 GT forklift are matched at the frozen confidence; pair recall gating, pair precision secondary).
4. **SmartSpaces W010/018/019/022 — re-used test (3rd use), not fresh** (C3): `code/run_ss_detector.py` + `code/score_ss.py` (copied from v3; pilot scorer imported read-only after its sha256 check, frozen-confidence constant set to the model's). Old and v3 numbers are v3's stored results from the same frames and scorer (`detector-v3/out/{old,v3}/ss_results.json`). No pass/fail weight except "not worse than v3".
5. **Export gate** (§6).
6. **CPU latency** (`code/bench_cpu.py`): ONNX Runtime CPU, 4 intra-op threads, 300 timed runs on 30 fixed validation frames, mean/median/p95 ms per frame (session only, and preprocess + run + decode), for old, v3, v4-T and v4-S, nothing else heavy running.

## 8. Pre-registered success criteria (per v4 arm)

| Id | Test set | Pass if |
|---|---|---|
| C1a | Stage I frame set | pair precision ≥ 0.9308 **and** pair recall ≥ 0.8144 (old 0.981 / 0.864 − 0.05) |
| C1b | 33 event runs | strict recall ≥ 24/33 |
| C2a | LOCO subset 3 | forklift AP50 ≥ old + 0.20 **and** forklift recall at frozen confidence ≥ old + 0.20 **and** forklift precision at frozen confidence ≥ 0.50 |
| C2b | keremberke test | forklift AP50 ≥ old + 0.20 **and** person AP50 ≥ old + 0.10 **and** pair recall ≥ old + 0.20 |
| C3 | SmartSpaces (re-used, 3rd use) | not worse than v3: pooled per-camera forklift found ≥ 0.2428 (v3 0.2728 − 0.03), pooled person recall ≥ 0.5141 and precision ≥ 0.7436 (v3 0.5641 / 0.7936 − 0.05), per-camera-cooldown alerts strict ≥ 12 (v3 14 − 2) and false alerts ≤ 12 (the GT-box oracle's count) |
| C4 | LOCO subset 3 | false-forklift rate on pallet trucks ≤ old + 0.03 **and** on stillages ≤ old + 0.03 |
| Export | — | as §6: opset 11, `images` 1×3×416×416 → `raw_predictions` 1×3549×7, parity 1e-3, `check_onnx_end_nodes` reports no problem |

Secondary (reported, never gating): AP50 and P/R per class and scene, bootstrap CIs, LOCO large/small recall and FP confusion, SDG event precision, frozen-truth event score, GDINO-24 frame set, SmartSpaces app-native candidates, humanoid / pallet-truck false persons, recall by box height, keremberke pair precision, CPU latency.

## 9. Decision rule

1. An arm passes when C1a, C1b, C2a, C2b, C3, C4 and the export gate all pass.
2. Both pass: **CPU** recommendation v4-T. **UGen300 (Hailo-10H)** recommendation v4-S only if it beats v4-T by ≥ 0.05 forklift AP50 on both LOCO subset 3 and keremberke test; otherwise v4-T.
3. Only v4-T passes: v4-T for both.
4. Only v4-S passes: v4-S for UGen300; for CPU only if its 4-thread mean session time is ≤ 150 ms per frame, otherwise keep the current model on CPU.
5. Neither passes: keep the current app model and report partial gains plainly. An arm failing C1 is reported as failing whatever else it passes.

## 10. Failure examples

`code/render_failures.py`, fixed rule, ≤ 12 JPEGs, for the recommended arm (v4-T if none is recommended): per SmartSpaces test scene the forklift miss with the tallest GT box (≤ 4); the first missed SDG event in run order (else the first false candidate) (≤ 1); on LOCO subset 3 the 2 largest missed GT forklifts and the 2 most confident unmatched forklift detections (≤ 4); on keremberke test the largest missed GT forklift, the largest missed GT person and the most confident unmatched detection (≤ 3).

## 11. Known limitations, stated in advance

- keremberke test has 42 images (25 pair images): CIs are wide; the gain margins are set large for this reason.
- LOCO subset 3 was evaluated (not tuned) by v3; person pseudo-labels on LOCO come from a COCO teacher and may miss or invent persons; LOCO person output is not scored on the test site.
- The SmartSpaces selection building has only two forklift instances; SmartSpaces test scenes are a 3rd use and are not fresh evidence.
- CPU training limits epochs (≤ 12) and excludes larger inputs or heavier augmentation.
- Small n on SDG (59 GT pairs, 33 events): the 0.05 / 2-event margins are about the noise level.
