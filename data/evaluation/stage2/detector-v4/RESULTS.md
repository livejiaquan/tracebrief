# Detector v4 (real-image person/forklift detector): results

Internal report, 2026-11-05. Protocol: `PROTOCOL.md` and `PREREG.json`, registered 2026-11-04T18:59:55Z. Registration came before any training step and before any new test-set inference by any model; training started at 19:00:06Z. Each model was scored once per test set after its freeze. Machine-readable summary: `out/summary.json`. Tables: `out/tables.md`.

## Verdict

**v4-S passes every pre-registered criterion. v4-T passes all of them except C2a.** v4-S is YOLOX-s at 416, frozen at epoch 11, confidence 0.25. v4-T is YOLOX-tiny, epoch 11, confidence 0.45. On the held-out LOCO site, v4-T's forklift recall at its frozen confidence is 0.154, below the 0.20 required. Its AP50 (0.271) and precision (0.744) pass.

Under the decision rule case "only v4-S passes", **v4-S is the recommended model for UGen300 and also for the CPU fallback**, since its 4-thread time is 37–41 ms per frame, below the 150 ms limit. Nothing in `tracebrief/` was changed; replacing the app weights is a separate decision.

What changed against old and v3:

- **Real forklifts.** For the first time, our models detect real forklifts. On LOCO subset 3, a site never trained or tuned on, forklift AP50 is 0.27 (T) and 0.34 (S), against ≈ 0 for old and v3. On the keremberke test images it is 0.91 (T) and 0.89 (S), against 0.00.
- **Real persons.** Person AP50 on keremberke is 0.51, against 0.09 (old). 17 of the 25 person-and-forklift images are pair-detected, against 0.
- **False forklifts.** No false forklift lands on any LOCO pallet truck or stillage.
- **SDG.** The SDG gate holds. Stage I pair P/R is 0.964 / 0.898 for both arms (old 0.981 / 0.864, v3 0.914 / 0.898); events are 25/33 (T) and 27/33 (S).
- **SmartSpaces (re-used test, not fresh).** Both arms are at least as good as v3. Forklifts are found per camera in 28% (T) and 33% (S) of cases, against 27% for v3, now in all four test buildings. Strict alerts are 20 and 21, against 14.

What is still weak:

- **Real-forklift recall.** v4-S finds 34% of the forklifts at the held-out LOCO site; v4-T finds 15%.
- **Small person test.** keremberke has only 42 test images, and person recall there is about 0.5.
- **False persons on robots and pallet trucks.** v4 produces more of them in SmartSpaces than v3, though far fewer than old (v4-S 4,997 / 2,570, against v3 507 / 8 and old 18,047 / 1,940; secondary). v4-S raises 8 false alerts, within the limit of 12.

## Key table

| Metric (frozen confidence) | old (0.40) | v3 (0.35) | v4-T (0.45) | v4-S (0.25) |
|---|---|---|---|---|
| C1a Stage I pair P / R (TP/FP/FN) | 0.981 / 0.864 (51/1/8) | 0.914 / 0.898 (53/5/6) | 0.964 / 0.898 (53/2/6) | 0.964 / 0.898 (53/2/6) |
| C1b events strict (of 33) | 26 | 26 | 25 | 27 |
| C2a LOCO subset 3 forklift AP50 | 0.0001 | 0.005 | **0.271** | **0.337** |
| C2a LOCO subset 3 forklift P / R at conf | 0.00 / 0.000 | 0.07 / 0.005 | 0.74 / **0.154** | 0.67 / **0.340** |
| C2b keremberke test forklift AP50 | 0.000 | 0.000 | **0.913** | **0.894** |
| C2b keremberke test person AP50 | 0.087 | 0.033 | **0.514** | **0.511** |
| C2b keremberke pair recall (of 25) | 0.00 | 0.00 | 0.68 | 0.68 |
| C3 SmartSpaces forklift found / person P / R (re-used test) | 0.122 / 0.675 / 0.254 | 0.273 / 0.794 / 0.564 | 0.276 / 0.920 / 0.539 | 0.332 / 0.821 / 0.565 |
| C3 SmartSpaces alerts strict / covered / false | 8 / 25 / 1 | 14 / 36 / 0 | 20 / 50 / 1 | 21 / 52 / 8 |
| C4 LOCO false forklift on pallet trucks / stillages | 0.0% / 0.50% | 0.65% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| CPU ms/frame, ORT 4 threads (mean, two runs) | 22.4 / 25.9 | 22.5 / 25.0 | 27.8 / 27.3 | 40.5 / 37.3 |
| **Pre-registered verdict** | — | — | **FAIL (C2a)** | **PASS** |

## 1. Data used

| Source (licence) | Training | Selection (validation) | Test |
|---|---|---|---|
| SDG-Warehouse v2 frames (OpenMDW-1.1) | 34 runs, 2,040 frames, 1,910 person / 1,970 forklift boxes | 8 runs, 480 frames, 437 / 465 | Stage I frame set (9 runs / 90 frames) and 33 event runs (unchanged held-out sets) |
| SmartSpaces MTMC 2026 (CC BY 4.0) | building A (000/002/004/006) + building B (008/009): 6,300 frames, 27,786 person / 3,642 forklift boxes | building D (012–016): 1,000 frames, 6,406 / 160 boxes | 010/018/019/022, 63,360 frames (re-used test, 3rd use) |
| LOCO (CC0-1.0) | subsets 1, 2, 4, 5 except three sessions: 3,646 images, 330 forklifts (in 228 images), 976 teacher persons, 517 ignore regions | sessions 2019-11-25_0, 2020-01-16_16, 2020-03-05_1: 578 images, 80 forklifts, 157 teacher persons, 90 ignore regions | **subset 3 (held-out site)**: 873 images, 188 forklifts, 308 pallet trucks, 402 stillages |
| keremberke/forklift-object-detection (Roboflow forklift-dsitv, CC BY 4.0) | train: 295 images, 319 forklift / 296 person boxes, 67 teacher-only person ignore regions | valid: 84 images, 96 / 75 boxes, 19 ignore regions | **test**: 42 images, 43 / 44 boxes, 25 images with both classes |

Each training epoch used 6,130 images: 2,040 SDG, 1,500 SmartSpaces, 2,000 LOCO (all 228 forklift images plus 1,772 drawn at random) and the 295 keremberke images twice (590). New data on disk: real-image frames 399 MB, keremberke 22 MB (zips 20 MB), YOLOX-s and YOLOX-L checkpoints 483 MB. The SmartSpaces frames (751 MB), LOCO images (684 MB) and SDG frames (103 MB) were read from detector-v3. Licences and attribution: `DATA_LICENSES.md`. CC BY 4.0 attribution is needed for SmartSpaces and keremberke if weights are released.

LOCO has no person class, so persons were pseudo-labelled by the official YOLOX-L COCO checkpoint: score ≥ 0.60 → person, [0.25, 0.60) → ignore region (grey-filled in training frames). Against keremberke GT persons the teacher reached precision 0.93 / 0.90 (train / valid) at 0.60, a lower bound because unlabelled persons count as false. Recall was 0.74 / 0.69. A first teacher run used the wrong BatchNorm eps (PyTorch's 1e-5 instead of YOLOX's 1e-3). A code review caught this before any use, and the output was discarded and re-run (`PROTOCOL.md` §3.3).

## 2. Splits and overlap audits (`out/overlap_audit.json`, `out/kb_dup_audit.json`)

- **SDG.** The 42 run ids were checked against every held-out, test, routine and development list: overlap 0, and train ∩ val = ∅. The frames are identical to the v2 split and to the stage1 v2 COCO files.
- **SmartSpaces.** Training (A, B), selection (D) and test scenes share no building or camera; the audit found no cross-role conflicts. Building A also contains 020/021 (v3's selection scene) and 023/024. None of them is used.
- **LOCO.** Test site subset 3 is in no training or selection pool and was never passed to the teacher. No session mixes roles. A 64-bit dHash check flags 97 test images within Hamming distance 8 of a training image. The closest pairs show different places with similar aisles (`out/loco_hash_flag_visual_check.jpg`), so these are hash collisions, not copies. Subset 3 is not "never seen": v3 had scored old, v3 and v3-A on it once, without tuning anything. Our new old and v3 caches on subset 3 are identical to v3's.
- **keremberke.** No near-duplicate pairs across splits; the smallest test-to-train/valid dHash distance is 14. The test split, like LOCO subset 3, was first used after registration: old and v3 baselines ran while training was under way, and the v4 arms ran only after their freezes. No v4 decision used either set.

## 3. Training, selection, freeze

| | v4-T (YOLOX-tiny 416) | v4-S (YOLOX-s 416) |
|---|---|---|
| Init (official COCO, Apache-2.0) | `yolox_tiny.pth` `9de513de…` | `yolox_s.pth` `f55ded71…` |
| Epochs run / selected | 12 / **11** | 12 / **11** |
| Frozen confidence | **0.45** | **0.25** (the floor) |
| Selection score (mean F1 of 7 cells) | 0.706 (all 5 floors met) | 0.725 (all 5 floors met) |
| Val SDG person / forklift P/R | 0.970/0.897, 0.996/0.989 | 0.939/0.911, 0.989/0.989 |
| Val SmartSpaces D person / forklift P/R | 0.909/0.684, 0.642/0.212 | 0.848/0.706, 0.340/0.412 |
| Val keremberke person / forklift P/R | 0.760/0.507, 0.918/0.812 | 0.688/0.587, 0.888/0.823 |
| Val LOCO forklift P/R (person, pseudo-labels) | 0.675/0.338 (0.811/0.573) | 0.554/0.512 (0.678/0.618) |
| Checkpoint sha256 | `a7c01fd7…d89d4` | `75d7feef…ded0c0` |
| ONNX sha256 / size | `88de60de…f4ea4b` / 20.2 MB | `fdcc92d1…f05be2` / 35.8 MB |
| Frozen (UTC) | 2026-11-05T00:55:30Z | 2026-11-05T01:06:29Z |
| Training wall time (threads) | 5.92 h (4) | 6.10 h (7) |

The export gate passed for both arms: opset 11, `images` 1×3×416×416 → `raw_predictions` 1×3549×7, identical to the app model. PyTorch/ORT parity max abs error was 1.2e-5 (T) and 5.1e-6 (S). The Hailo script's `check_onnx_end_nodes`, imported read-only, reported no problem for either. Both graphs have the same operator counts (83 Conv, 80 Sigmoid, …), because YOLOX-s differs from tiny only in width. The app can load either file unchanged; it only needs a settings file (`freeze/<arm>/vision-compact-<arm>.json`).

For UGen300, arm S's HEF compile would need these changes:
- `--onnx` pointing to the new file;
- `nms_scores_th` in `nms_yolox_warehouse.json` set to 0.25;
- optionally, a new HAR/HEF name.

Whether the parsed HAR layer names (conv54…conv83) match the NMS JSON can only be checked by a DFC parse, which was not run here.

Validation trends (not test results):
- Both arms peaked at epoch 11 of 12 (score 0.706 / 0.725), and epoch 12 was slightly lower, so a longer schedule would probably gain little.
- The SmartSpaces-D forklift cell (2 instances) was noisy, between 0.0 and 0.6 F1.
- Arm S epoch 2 met every floor with only 5 LOCO-val forklift detections, and epochs 3–5 then did not count as improvements. Feasibility-first ranking with a small cell is fragile, but it did not decide the final pick.

## 4. Test results

### 4.1 SDG held-out (app pipeline, 1 fps; criterion C1)

| Model (conf) | Stage I pair TP/FP/FN | Pair P / R | Events strict (primary) | Event precision | Interval-overlap variant | Frozen-truth score | GDINO-24 pair P / R (secondary) |
|---|---|---|---|---|---|---|---|
| old (0.40) | 51/1/8 | 0.981 / 0.864 | 26/33 | 27/31 | 30/33 | 26 | 0.912 / 0.852 |
| v3 (0.35) | 53/5/6 | 0.914 / 0.898 | 26/33 | 26/31 | 31/33 | 25 | 0.966 / 0.934 |
| **v4-T (0.45)** | 53/2/6 | **0.964 / 0.898** | **25/33** | 25/30 | 29/33 | 25 | 0.958 / 0.926 |
| **v4-S (0.25)** | 53/2/6 | **0.964 / 0.898** | **27/33** | 27/32 | 31/33 | 26 | 0.958 / 0.943 |

All models were run through the current app pipeline (server.py `a7a2ce9a…`). Old and v3 reproduce their published numbers exactly. Event misses are the same kind as before: peak before the event window (4), a single 1 Hz sample (1–2) and an image gap above the rule (1). v4-T also has one candidate whose person location is more than 2.5 m away.

### 4.2 Real images: LOCO held-out site, subset 3 (C2a, C4)

| Model (conf) | Forklift AP50 | TP / FP at conf | Recall | Precision | FP / 100 images | False forklift on pallet trucks (of 308) | on stillages (of 402) |
|---|---|---|---|---|---|---|---|
| old (0.40) | 0.0001 | 0 / 30 | 0.000 | 0.000 | 3.4 | 0 (0.0%) | 2 (0.50%) |
| v3 (0.35) | 0.005 | 1 / 13 | 0.005 | 0.071 | 1.5 | 2 (0.65%) | 0 |
| **v4-T (0.45)** | **0.271** | 29 / 10 | **0.154** | 0.744 | 1.1 | 0 | 0 |
| **v4-S (0.25)** | **0.337** | 64 / 32 | **0.340** | 0.667 | 3.7 | 0 | 0 |

The table below shows the paired image bootstrap of each v4 arm minus old (95% CI). The upper end of v4-T's recall interval sits just above the +0.20 margin, but the pre-registered test is on the point estimate (0.154).

| | AP50 | Recall |
|---|---|---|
| v4-T − old | +0.20 to +0.34 | +0.11 to +0.21 |
| v4-S − old | +0.26 to +0.42 | +0.27 to +0.41 |

187 of the 188 GT forklifts are large (≥ 96/1080 of the image height). Every unmatched v4 forklift detection falls on background: none overlaps a pallet truck, stillage, pallet or small load carrier.

### 4.3 Real images: keremberke test (C2b; 42 images)

| Model (conf) | Forklift AP50 | Forklift P / R | Person AP50 | Person P / R | Pair recall (of 25) | Pair precision |
|---|---|---|---|---|---|---|
| old (0.40) | 0.000 | 0.00 / 0.00 (1 FP) | 0.087 | 0.36 / 0.09 | 0.00 (0) | — |
| v3 (0.35) | 0.000 | — / 0.00 | 0.033 | 0.05 / 0.07 | 0.00 (0) | — |
| **v4-T (0.45)** | **0.913** | 0.86 / 0.88 | **0.514** | 0.72 / 0.48 | **0.68 (17)** | 0.74 |
| **v4-S (0.25)** | **0.894** | 0.83 / 0.91 | **0.511** | 0.63 / 0.50 | **0.68 (17)** | 0.68 |

Bootstrap 95% CIs vs old:

| | Forklift AP50 | Person AP50 | Pair recall |
|---|---|---|---|
| v4-T | +0.82 to +0.99 | +0.33 to +0.55 | +0.50 to +0.87 |
| v4-S | +0.80 to +0.97 | +0.23 to +0.64 | +0.48 to +0.87 |

Persons on this set are mostly forklift drivers, partly hidden by the cab and mast. That limits person recall for every model.

### 4.4 SmartSpaces W010/018/019/022: re-used test (3rd use), not fresh (C3)

These scenes were used by the pilot, by v3 and now by v4. They are **not fresh evidence**, and the only pre-registered weight is "not worse than v3". The same frames and the pilot scorer were used; old and v3 numbers are v3's stored results.

| Scene | Forklift found per camera: old / v3 / v4-T / v4-S | Any-camera found: old / v3 / v4-T / v4-S | Person P/R: old / v3 / v4-T / v4-S | AP50 person: old / v3 / T / S | AP50 forklift: old / v3 / T / S |
|---|---|---|---|---|---|
| W010 | 20.1% / 48.1% / 36.5% / 41.4% | 67 / 100 / 91 / 100% | .64/.24 / .80/.59 / .92/.49 / .84/.51 | .21 / .60 / .54 / .54 | .24 / .52 / .51 / .45 |
| W018 | 2.1% / 1.4% / 13.9% / 20.9% | 6 / 3 / 19 / 26% | .43/.32 / .70/.73 / .98/.80 / .95/.94 | .35 / .76 / .91 / .95 | .12 / .03 / .20 / .22 |
| W019 | 2.9% / 0.1% / 19.6% / 25.7% | 6 / 0 / 24 / 33% | .82/.28 / .83/.59 / .83/.67 / .76/.71 | .30 / .61 / .70 / .72 | .05 / .00 / .25 / .26 |
| W022 | 0.0% / 0.0% / 45.2% / 41.8% | 0 / 0 / 45 / 42% | .93/.28 / .79/.43 / .95/.67 / .77/.68 | .30 / .47 / .71 / .69 | .00 / .00 / .64 / .51 |
| **Pooled** | 12.2% / 27.3% / **27.6%** / **33.2%** | | .675/.254 / .794/.564 / **.920/.539** / **.821/.565** | | |

| Secondary | old | v3 | v4-T | v4-S |
|---|---|---|---|---|
| Alerts (61 GT events, per-camera cooldown): alerts / strict / covered / false | 9 / 8 / 25 / 1 | 14 / 14 / 36 / 0 | 26 / 20 / 50 / 1 | 36 / 21 / 52 / 8 |
| App-native 1 fps candidates: strict events / candidates / false | 27 / 29 / 2 | 39 / 41 / 2 | 44 / 62 / 9 | 51 / 91 / 27 |
| False person boxes on humanoids (FourierGR1T2 + AgilityDigit) | 18,047 | 507 | 2,089 | 4,997 |
| False person boxes on pallet trucks | 1,940 | 8 | 1,752 | 2,570 |

v4 finds forklifts in all four buildings; v3 found them only in W010. The pallet-truck and humanoid false persons rose again compared with v3, though they stay far below old. Missing negatives do not explain this: building B contains many humanoid robots and building A contains pallet trucks, both unlabelled in training. One untested possibility is the broader real-image person class, since pseudo-labelled LOCO workers often stand on or next to pallet trucks.

### 4.5 CPU latency (ONNX Runtime, 4 intra-op threads, i7-10700, 300 runs on 30 validation frames)

| Model | ONNX size | Run 1: session.run mean / median / p95 ms | Run 2 (reverse order): mean / median / p95 ms | Preprocess + run + decode, mean (run 1 / 2) |
|---|---|---|---|---|
| old (YOLOX-tiny) | 20.2 MB | 22.4 / 17.9 / 31.3 | 25.9 / 22.6 / 33.9 | 23.9 / 27.6 |
| v3 (YOLOX-tiny) | 20.2 MB | 22.5 / 18.4 / 30.9 | 25.0 / 20.8 / 32.2 | 24.0 / 26.6 |
| v4-T (YOLOX-tiny) | 20.2 MB | 27.8 / 31.0 / 33.9 | 27.3 / 31.0 / 33.1 | 29.4 / 29.0 |
| v4-S (YOLOX-s, 416) | 35.8 MB | 40.5 / 33.0 / 54.1 | 37.3 / 31.1 / 52.1 | 42.2 / 38.9 |

The two passes are in `out/cpu_latency.json` and `out/cpu_latency_run2.json`. old, v3 and v4-T have identical graphs (same operators and shapes, no denormal weights). Their 3–5 ms spread is run-to-run noise on this WSL machine: an extra check measured v4-T at 21.0 ms against old at 22.4 ms. YOLOX-s costs about 1.5× YOLOX-tiny on CPU, about 25 fps at 4 threads, which is ample for the app's 1 fps sampling. UGen300 (Hailo-10H) latency was not measured; it needs a compiled HEF.

## 5. Pre-registered criteria

| Id | Pass if (pre-registered) | v4-T | v4-S |
|---|---|---|---|
| C1a | Stage I pair P ≥ 0.9308 and R ≥ 0.8144 | 0.964 / 0.898 pass | 0.964 / 0.898 pass |
| C1b | events strict ≥ 24/33 | 25 pass | 27 pass |
| C2a | LOCO s3 forklift AP50 ≥ old + 0.20 (≥ 0.2001), recall at conf ≥ old + 0.20 (≥ 0.20), precision ≥ 0.50 | 0.271 / **0.154** / 0.744 **FAIL** | 0.337 / 0.340 / 0.667 pass |
| C2b | keremberke forklift AP50 ≥ 0.20, person AP50 ≥ 0.187 (old 0.087 + 0.10), pair recall ≥ 0.20 | 0.913 / 0.514 / 0.68 pass | 0.894 / 0.511 / 0.68 pass |
| C3 | SmartSpaces (re-used): found ≥ 0.2428, person R ≥ 0.5141, P ≥ 0.7436, strict alerts ≥ 12, false ≤ 12 | 0.276 / 0.539 / 0.920 / 20 / 1 pass | 0.332 / 0.565 / 0.821 / 21 / 8 pass |
| C4 | LOCO s3 false-forklift rate pallet trucks ≤ 0.030, stillages ≤ 0.035 (old + 0.03) | 0 / 0 pass | 0 / 0 pass |
| Export | opset 11, 1×3×416×416 → 1×3549×7, parity 1e-3, Hailo end-node check | pass | pass |
| **All** | | **FAIL** | **PASS** |

The decision rule gives "only v4-S passes": v4-S for UGen300, and for CPU because 37–41 ms ≤ 150 ms. No threshold, epoch or rule was changed after any test result.

## 6. Failure examples (`out/v4-S/failures/`, fixed rule, 12 JPEGs)

The images were rendered for the recommended arm, v4-S at confidence 0.25. White boxes are GT person or forklift, grey boxes are other GT objects, orange is a predicted person and green a predicted forklift.

- **01–04: SmartSpaces forklift misses (tallest GT box per scene).**
  - 01 (W010): a stand-on order picker at the frame edge.
  - 02 (W018): a near-field counterbalance truck. A forklift box is predicted on the body, but it does not match the larger GT box (which includes the mast) under the scorer's rule.
  - 03 (W019): a walk-behind stacker in the foreground.
  - 04 (W022): a forklift far down a racking aisle.
- **05: missed SDG event** (`00e0991f…`). The candidate interval overlaps the event, but its peak lies outside the event window. This is the same failure as old and v3.
- **06–07: LOCO subset 3 forklift misses (largest).** Both are forklifts passing very close to the walking camera: motion-blurred and cut by the frame.
- **08–09: LOCO subset 3 false forklifts (most confident).** Two consecutive frames of a large blue high-speed roll-up door. Every v4-S false forklift on this site falls on background, none on a pallet truck or stillage.
- **10–11: keremberke test image 36.** Two counterbalance trucks carrying timber. They are detected, but the GT boxes include the tall load and the predicted boxes do not, so IoU is below 0.5. The drivers behind the cab glass are missed.
- **12: keremberke test image 9, "false person".** The standing worker is detected, but his GT box includes the outstretched arm, so IoU is below 0.5.

Real-image errors are mostly localisation-convention mismatches (load or arm inside the GT box) and close, blurred views. Confusion with pallet trucks or stillages does not appear.

## 7. Cost

| Step | Wall time | Threads |
|---|---|---|
| Downloads (keremberke 20 MB, YOLOX-s 72 MB, YOLOX-L 434 MB) | about 2 min | — |
| Teacher pseudo-labels, YOLOX-L on 4,603 images | 40 min (2 × 6 threads), plus a discarded first run of about 21 min (10 threads) | 12 |
| Frame build, audits, harness check, smoke tests | about 15 min | ≤ 9 |
| Training arm T, 12 epochs (2.2 s per batch of 8; validation 40 min total) | 5.92 h | 4 |
| Training arm S, 12 epochs, concurrently with T | 6.10 h | 7 |
| SDG app pipeline (34 videos), per model | 100–110 s | 4 |
| LOCO subset 3, 873 images, per model | 48 s (4 threads); 136 s (old/v3, 1 thread) | 1–4 |
| keremberke test, 42 images, per model | 2–6 s | 1–4 |
| SmartSpaces 63,360 frames: v4-T / v4-S | 68 min (1 × 4) / about 53 min (2 × 4, including a 4-min single-video rerun) | 4–8 |
| CPU latency (2 passes) and failure rendering | about 8 min | 4 |
| **Total, start of work to final report** | **about 8.5 h** (01:47–10:15 local), within the 12 h budget | ≤ 12 |

Memory was not a constraint on the 20 GB machine (not measured per process). Disk used: runs 700 MB (all per-epoch checkpoints), freeze 54 MB, outputs 43 MB, data 0.93 GB.

## 8. Deviations and notes

- **Before registration.** The first teacher run used the wrong BatchNorm eps and was discarded (see §1). An independent code review also found:
  - the training defaults still said 14 epochs and a 6.0 h budget (fixed to 12 epochs and 6.5 h);
  - a tie-break mismatch in epoch ranking (confidence added to the rank);
  - a `tables.py` key error.

  All were fixed before `PREREG.json` was written.
- **`tables.py` changed after registration.** The model label `m` was overwritten by the events-metrics dict, so the script crashed. This was found in a dry run on the old/v3 baseline outputs at 04:30 local, 4.5 h before the first v4 test result. The criteria logic is unchanged, but the file's sha256 differs from `PREREG.json`.
- **`run_ss_detector.py` self-check changed after registration.** The copied v3 runner asserts that the ≥ 0.05 cache, filtered at the frozen confidence, equals a decode at that confidence. The v4-S run stopped at Warehouse_010 Camera_0002 frame 8700. The cause is that the app decode rounds confidences to 4 decimals, so a box with a true score just below 0.25 is stored as 0.2500. The check now skips such boundary boxes; the caches and the pilot scorer are unchanged.
  - Effect: the pilot scorer counts boundary boxes as detections at the frozen confidence. That is 26 of 202,603 v4-S detections and 13 of 171,589 v4-T detections in the SmartSpaces caches, and 0 in every LOCO and keremberke cache.
  - Only the interrupted video was re-run; the other 20 cached videos were kept.
- **v4-S SmartSpaces run used 2 workers.** It ran 2 workers × 4 threads (new wrapper `code/ss_eval.sh`, same commands) instead of `eval_model.sh`'s 1 × 4, to fit the time budget. Only speed differs.
- **Old and v3 run with 1 thread.** They were run on LOCO subset 3 and keremberke test with 1 thread while training ran. Their subset-3 caches are identical to v3's all-subset caches.
- **Code added after registration, presentation only.** `code/report_tables.py`, plus one exploratory latency check (denormal flush) that is not part of any criterion.
- **Threads.** Training used 11 threads (4 + 7); the teacher and the final SmartSpaces evaluation used 12 (6 + 6, then 4 + 8). The machine has 8 physical cores, so the arms slowed each other: both ran at about 2.2 s per batch.
- **Stray bytecode.** `code/__pycache__/` holds two `.pyc` files written at 02:09–02:15 by the read-only review's imports. No run used them (all runs set `PYTHONDONTWRITEBYTECODE`), and the folder can be deleted.
- **Changes in the app since v3.** `tracebrief/src/tracebrief/server.py` changed after v3, but not during this study (sha256 `a7a2ce9a…` at registration and at the end). All four models were run with it, and old and v3 reproduce their published SDG numbers exactly.
- **Nothing written outside this folder.** No file outside `work/stage2/detector-v4/` was written, the app weights were not replaced, and nothing was committed or uploaded.
- **Limitations seen during the work.** LOCO leaves some forklifts unlabelled; one was seen at an image edge in a subset-1 training frame, so they act as false negatives in training. LOCO-test persons are not scored.
- **Selection-rule fragility (validation only).** Arm S's epoch 2 met every floor with only 5 LOCO-val forklift detections, which kept better epochs 3–5 from counting as improvements. It did not change the final choice; a future rule should require a minimum number of detections behind each precision floor.

## 9. Recommendation

1. **Adopt v4-S, as the pre-registered decision rule gives, for UGen300 and for the CPU fallback.**
   - **CPU.** The CPU path is drop-in: it uses the same input and output as the app model. Point a settings file at `freeze/v4-S/yolox-s-warehouse-v4-S.onnx` with sha256 `fdcc92d1…` and confidence 0.25, as in `freeze/v4-S/vision-compact-v4-S.json`. Cost is about 38–42 ms per frame at 4 threads, against about 24 ms for tiny.
   - **UGen300 (Hailo-10H).** Before switching:
     - compile with `compile_yolox_hef.py --onnx …v4-S.onnx`;
     - set `nms_scores_th` to 0.25 in `nms_yolox_warehouse.json`;
     - check after the DFC parse that the HAR output layer names still match the NMS JSON (the ONNX end-node check passed);
     - re-check accuracy on the device after quantisation.
2. **v4-T is a near miss.** It is better than old and v3 on every real-image metric, has higher precision (LOCO 0.74, SmartSpaces persons 0.92) and passes everything except C2a (LOCO recall 0.154 < 0.20). It is not adopted. Its frozen confidence of 0.45 is probably too high for real forklifts, but choosing a lower one now would use test data. That would need a new protocol and a fresh real-image test set.
3. **Before relying on v4-S in production, note the following:**
   - It still misses about two thirds of forklifts at an unseen real site, and half of real persons on keremberke.
   - Its SmartSpaces false persons on humanoid robots and pallet trucks rose compared with v3 (still far below old), and it produces more false app-native candidates (27 vs 2).
   - keremberke test (42 images) and LOCO subset 3 are the only real-image tests; subset 3 had been scored once before by v3, and SmartSpaces is a 3rd use.
4. **Next steps.**
   - Add real forklift images from more sites (for example the unannotated LOCO data, or other permissively licensed sets) while keeping a fresh real site for testing.
   - Reduce false persons on humanoids and pallet trucks, for example with hard-negative mining on SmartSpaces training buildings, and check whether real-image person labels cause them.
   - Require a minimum detection count behind each precision floor in the selection rule.

## Files

`PROTOCOL.md`, `PREREG.json` (copy at registration: `logs/PREREG.at_registration.json`), `DATA_LICENSES.md`; `code/` (all scripts; registered hashes in `PREREG.json`); `data/` (keremberke, teacher output, real-image frames, SmartSpaces pool lists); `runs/{T,S}/` (config, history, per-epoch and best checkpoints, logs); `freeze/{v4-T,v4-S}/` (FREEZE.json, ONNX, app settings); `out/{old,v3,v4-T,v4-S}/` (SDG app-pipeline results, LOCO subset 3 and keremberke caches and results, SmartSpaces caches and results for v4); `out/overlap_audit.json`, `out/kb_dup_audit.json`, `out/teacher_check.json`, `out/harness_check_old.json`, `out/cpu_latency.json`, `out/summary.json`, `out/tables.md`.
