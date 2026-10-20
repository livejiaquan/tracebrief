# RGB validation results

**Broader search is the product default. On the same nine scorable new synthetic runs, it achieved 90.2% localized image-space pair precision and 93.2% recall; Stricter search achieved 90.2% precision and 62.7% recall. The Broader comparison and default selection were made after seeing the prospective Stricter results. These are not near-miss or collision detection scores.**

## Fixed-profile comparison and default decision

| Profile | Pair precision / recall | Pair counts | Median CPU time, all ten inputs |
|---|---|---|---|
| **Broader search — default**: 0.35, 800/1333 | **90.2% / 93.2%** | 55 TP, 6 FP, 4 FN | 71.280 s per 10 s video |
| Stricter search: 0.65, 512/853 | 90.2% / 62.7% | 37 TP, 4 FP, 22 FN | 36.590 s per 10 s video |

Both profiles processed the same ten videos and retained all 100 sampled predictions. The same annotation-integrity exclusion leaves nine scorable runs and 90 frames. The exact original v1 source/configuration was used for the Broader comparison after the frozen Stricter evaluation. This is an explicit **post-result paired comparison and product-default selection**, not a fresh heldout validation of the selected default. The original baseline results remain separate below.

Broader recovered 17 positive sampled frames missed by Stricter while retaining all 38 of Stricter's true-positive frames; four positive frames were missed by both. It added two binary false-positive frames. Broader binary counts were 55 TP, 5 FP, 4 FN and 26 TN. The combined configuration change materially reduced sensitivity; this comparison cannot isolate confidence from resolution effects.

The default prioritizes finding more review candidates. Broader remains vulnerable to shelf errors documented in the original routine cohort and creates more review work. Stricter remains available as a faster, less sensitive option. Neither profile can establish safety. Their routine cohorts differ, so the original nine groups in seven clips versus the new one group in one clip cannot establish a paired false-alarm reduction.

All-ten-video Broader CPU times ranged 65.087–95.581 seconds, versus 27.852–48.034 seconds for Stricter. Sequential batches ran on a shared workstation; these are descriptive runtime observations, not a controlled speedup benchmark. See the [paired comparison bundle](../data/evaluation/phase2/comparator_v1/README.md).

## Stricter profile and prospective evaluation

Two fixed profiles are available: Broader search retains the original v1 configuration (0.35, default 800/1333); Stricter search uses v2 (0.65, 512/853). This is a sensitivity, review-work and runtime tradeoff. V2 is not an unqualified accuracy improvement.

V2 uses Grounding DINO tiny at confidence 0.65 and processor resize 512/853, with the same 1 Hz sampling, image-plane gap rule and grouping settings. Selection used original development and calibration material; the complete configuration and new 20-run selection were frozen before new-video inference or frame inspection. No tuning followed these results.

The original 34 evaluated runs became calibration evidence. A faster resize initially reintroduced known shelf errors at confidence 0.60. Filtering the saved calibration outputs at 0.65 retained the original development matches and sampled parked forklift, but removed one routine person detection. All calibration history and failures are retained separately.

### Primary results and annotation coverage

Ten new primary runs produced 100 sampled predictions. One video has 300 RGB frames but only 198 annotation rows, no explicit frame IDs/timestamps, and visibly misaligned late-row boxes. The entire run is unscorable; all its predictions remain available. There is no replacement run, rescaling, or outcome-based exclusion. A visual check of early and late frames found no evident drift in the other nine runs. **The reported denominator is nine runs and 90 frames**, containing 59 image-space-positive and 31 negative annotated frames.

| Measure | Precision | Recall | Counts |
|---|---:|---:|---|
| Prespecified localized proximity-pair evidence | **90.2%** | **62.7%** | 37 TP, 4 FP, 22 FN |
| Binary sampled-frame proximity agreement | 92.7% | 64.4% | 38 TP, 3 FP, 21 FN, 28 TN |
| Person localization, IoU ≥ 0.50 | 98.8% | 93.3% | 83 TP, 1 FP, 6 FN |
| Forklift localization, IoU ≥ 0.50 | 100.0% | 80.0% | 72 TP, 0 FP, 18 FN |

The localized metric requires both predicted actor boxes to match a gold proximity pair at IoU ≥ 0.50. A 2,000-replicate run-cluster bootstrap gives 95% intervals of 81.8–98.1% for pair precision and 43.1–82.7% for recall. These wide intervals describe this small synthetic cohort only.

All 21 binary false-negative frames were inspected. Five lack a retained person detection under severe forklift occlusion. Sixteen lack a retained forklift detection: six are dark-scene frames; many others show frontal/top-truncated forklifts, and one has a large visible side view. Several missed forklifts remain clearly visible. Saved outputs do not include rejected logits, so these observations cannot establish whether confidence rejection or absent proposals caused each miss. The reduced recall is a material limitation; no-candidate output cannot establish safety.

### Separate routine cohort

Ten new routine videos produced 140 sampled frames and **one candidate group in one clip**. Every one of its 14 evidence frames was inspected and shows a person and a real parked forklift. No shelf-as-forklift event was observed in this small cohort. This does not demonstrate a zero false-alarm rate.

A three-frame-per-video actor audit found three scenes with a clearly visible person and parked forklift, four with a visible person but no clearly visible forklift, and three with neither actor clearly visible in the inspected frames. Nine annotation streams were retrieved; one selected routine annotation was not found in the expected or adjacent archive shards. Retrieved streams omit stationary forklifts. **Missing labels prevent a specificity estimate**, and neither actor absence in three inspected frames nor a lack of candidates proves safety.

### CPU runtime

Across all ten primary videos, median end-to-end time was **36.590 seconds per 10-second video**, range 27.852–48.034 seconds. Routine clips took a median 41.199 seconds for approximately 14 seconds of input. Model loading and evidence export are included; workstation concurrency varied. These are batch observations, not controlled speedup or real-time claims. No UGen300 measurement is included.

### Reproducibility

The [prospective v2 bundle](../data/evaluation/phase2/README.md) retains all 20 predictions, frozen settings/source hashes, coverage exclusions, sampled gold, raw source provenance, metrics, uncertainty and audits. Rescoring from the compact gold and from the original annotations gives identical counts. The exact original source and unchanged baseline evidence remain available below. Frame/pair recall is not event recall.

---

# Original baseline (v1)

**TraceBrief discovers review candidates from raw RGB. On 24 heldout synthetic runs, the stricter actor-localized image-space pair metric achieved 92.1% precision and 95.9% recall. These are not near-miss or collision detection scores.**

## What was frozen and measured

Three previously explored runs were development only. The final detector, thresholds, 1 Hz sampling and 24 independent run IDs were frozen before heldout inference. All 24 generator seeds differ. The filenames in artifact archives add an extra hash prefix: an initial split mistake was caught and corrected during development, before any heldout prediction. Normalized run IDs exclude all development runs.

Grounding DINO tiny, revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, processes every sampled RGB frame with the fixed prompt `a person. a forklift.`. Inference never reads annotations. It uses confidence 0.35, text threshold 0.25, per-class NMS 0.45, and the image-plane box-gap rule described in [VISION.md](VISION.md). No settings changed after heldout inference began.

The 240 sampled frames contain 122 annotated image-space-positive frames and 118 far-apart negative frames; both actors are annotated in all 118 negatives. They come from one synthetic generator/scene family. Twenty-three videos contain at least one positive sampled frame; only one video contains none. This is not a set of ten independent positive and ten independent negative safety videos.

## Original baseline results

| Measure | Precision | Recall | Counts |
|---|---:|---:|---|
| Prespecified binary sampled-frame proximity agreement | 92.9% | 96.7% | 118 TP, 9 FP, 4 FN, 109 TN |
| Supplementary localized proximity-pair evidence | **92.1%** | **95.9%** | 117 TP, 10 FP, 5 FN |
| Person localization, IoU ≥ 0.50 | 99.6% | 98.3% | 236 TP, 1 FP, 4 FN |
| Forklift localization, IoU ≥ 0.50 | 99.6% | 99.6% | 239 TP, 1 FP, 1 FN |

The stricter pair metric requires BOTH predicted actor boxes to match the corresponding gold proximity pair at IoU ≥ 0.50. It was added as a supplementary evaluator audit after predictor freeze, because binary frame agreement alone can conceal incorrect evidence localization. The prespecified result is retained.

A 2,000-replicate percentile bootstrap resampling entire runs gives a 95% interval of 86.4–96.8% for localized-pair precision and 91.5–99.2% for recall. These intervals reflect variation within this small synthetic cohort, not real warehouse generalization.

## Errors and abstention behavior

All errors are preserved in [results.json](../data/evaluation/results.json), together with every prediction. No failed run was removed.

- The nine binary false-positive frames crossed the arbitrary 0.30 threshold because predicted and annotated box boundaries differ: their annotated gap ratios were 0.308–0.407.
- Three binary false-negative frames missed an annotated person; inspected examples show heavy occlusion and red illumination. Another false-negative frame had a box-boundary mismatch. Annotation visibility is itself imperfect, so these are annotation-agreement errors, not a complete account of what a human can see.
- The grouping rule emitted 22 review groups. It intentionally requires two positive sampled frames. One run had a correctly detected single positive frame but no group; another run had a single false-positive frame but no group. Thus grouping can suppress both brief true observations and isolated false positives. **Frame/pair recall is not event recall.**
- One-hertz sampling can miss shorter interactions. Person/forklift identities are not tracked across frames.

## Runtime and hardware scope

The primary cohort used an Intel Core i7-10700 host under WSL, with at most four PyTorch/OpenCV threads. Median end-to-end time was **71.786 seconds per 10-second video**, with a range of 65.491–153.976 seconds. Timing includes model loading and JPEG export and reflects concurrent workstation workloads. It is not a controlled throughput benchmark or a real-time capability claim.

No UGen300 speed, power, memory or accuracy measurement is included. The CPU Grounding DINO path is not asserted to compile to Hailo. A compatible deployment path and the actual benefit of accelerator-backed generation remain to be measured in Stage II.

## Separate routine cohort

A further ten preselected box-pickup videos use the unchanged predictor and a separate freeze. They have 418 frames each except one with 419, at 30 fps. All ten annotation streams contain a Worker but omit stationary forklifts and all other object annotations. Therefore missing forklift labels cannot be converted into negative truth.

A raw-frame visual audit at indices 0, 150 and 299 found clearly visible person/forklift pairs in two runs; the other eight had no clearly visible forklift in those inspected frames. This is not ten verified hard-negative videos, and the three-frame inspection does not establish whole-video safety.

All ten runs completed: 140 sampled frames produced nine review groups in seven videos; median end-to-end time was 90.569 seconds. The first routine clip produced two candidates for a correctly localized person near a parked forklift in image space. Other clips exposed shelf-as-forklift errors, retained in the bundle. These results motivated a separate calibration and prospective second evaluation; the original results remain unchanged. Incomplete forklift annotations prevent a specificity estimate.

## Reproduce and inspect

The [portable evaluation bundle](../data/evaluation/README.md) contains sanitized per-run predictions, sampled gold boxes, exact video/annotation hashes, pinned source URLs, frozen settings and aggregate results. Re-scoring from the portable compact gold exactly matched re-scoring from original annotations. A separate download of a pinned video reproduced its SHA-256 under a 32 MiB transfer cap.

All original RGB and annotation material derives from NVIDIA's public synthetic warehouse dataset, with OpenMDW-1.1 origin/license notices. Model weights are not included in the source archive. See the bundle and model documentation for rights and retrieval instructions.
