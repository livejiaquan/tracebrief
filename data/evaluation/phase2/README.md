# Prospective revised-configuration evaluation

This cohort contains ten new near-scenario videos and ten new routine box-pickup videos. It excludes the three original development runs and all 34 videos used in the original evaluation and subsequent calibration. Selection uses the next eligible lexicographically sorted run IDs; no failed run is dropped.

The configuration was frozen before new-video inference or frame inspection: Grounding DINO tiny, confidence 0.65, processor shortest edge 512 / longest edge 853, 1 Hz sampling, image-space gap ratio 0.30, minimum two observations per group. The model revision, exact settings and implementation/configuration hashes are in `freeze.json`; exact source is in `vision.py.txt`. The exact frozen configuration is `vision.json` beside this README; the application exposes it as the Stricter search profile (`config/vision-stricter.json`). Broader search uses the original v1 configuration and is reported separately.

## Product selection after this evaluation

The application and CLI default is now **Broader search**, selected after an explicit original-v1 comparison on these same ten primary inputs. That post-result comparison is stored separately in [comparator_v1/README.md](comparator_v1/README.md); it does not change this frozen Stricter cohort, settings or results. Broader pair precision/recall was 90.2%/93.2%, versus Stricter 90.2%/62.7% on the same nine scorable runs. Both profiles remain available.

## Annotation coverage

All ten primary videos were inferred (100 sampled frames). One run has only 198 annotation rows for a 300-frame video and visibly misaligned late-row boxes, without frame IDs or timestamps. The complete run remains in predictions but is unscorable; no replacement or time rescaling was introduced. Reported primary metrics therefore cover **nine runs and 90 frames**. `coverage_exclusions.json` records the reason, hashes and retained coverage. This exclusion follows annotation integrity, not detector accuracy.

## Scope

Primary results measure agreement with synthetic 2D annotations, including a localized pair metric requiring both actor boxes at IoU ≥ 0.50. They do not measure physical distance, near misses, collision risk or event recall. Routine annotations omit stationary forklifts, so routine candidate counts and qualitative visual audits cannot establish specificity. All results are CPU batch measurements; no UGen300 measurements are claimed.

The original 24-run metrics in the parent directory describe the original frozen configuration. Those 24 runs plus ten original routine runs subsequently informed calibration and are not heldout evidence for this revised configuration. Calibration history, including the failed 0.60 resize and the lost person detection after changing to 0.65, remains in `../calibration/`.

## Re-score saved outputs

From the `tracebrief/` directory:

```bash
python scripts/vision_evaluate.py --predictions data/evaluation/phase2/primary/predictions --compact-gold data/evaluation/phase2/gold.json --coverage-exclusions data/evaluation/phase2/coverage_exclusions.json --output /tmp/phase2-rescore.json
python scripts/vision_summarize.py /tmp/phase2-rescore.json --output /tmp/phase2-intervals.json
```

`evaluation.json` and `uncertainty.json` retain reference aggregates and run-cluster bootstrap intervals. `routine_summary.json` and `visual_audit.json` retain the separate routine analysis. `sources.json` records pinned URLs, archive members, byte offsets and SHA-256 hashes for original videos and annotations. Model weights and source videos are fetched separately; neither is embedded here.

Retrieve exact videos with the bounded downloader:

```bash
python scripts/vision_fetch_eval.py --manifest data/evaluation/phase2/sources.json --cohort primary --output data/phase2-primary --max-mib 512
python scripts/vision_fetch_eval.py --manifest data/evaluation/phase2/sources.json --cohort routine --output data/phase2-routine --max-mib 512
```

Origin: NVIDIA PhysicalAI WorldModel Synthetic Warehouse Operations Scenes, revision recorded in `sources.json`, OpenMDW-1.1. See `docs/licenses/` for complete license and origin notices.

Retrieve available primary annotations and reproduce the alignment audit (the excluded source file remains in the manifest):

```bash
python scripts/vision_fetch_eval.py --manifest data/evaluation/phase2/sources.json --cohort primary --asset annotation --output data/phase2-annotations --max-mib 64
python scripts/vision_alignment_audit.py --video data/phase2-primary/052961d2a2e7eeaf2652_run_5_seed_1856081553.ceiling_00.rgb.mp4 --annotations data/phase2-annotations/7c7bf9256cc1113c3285_052961d2a2e7eeaf2652_run_5_seed_1856081553.ceiling_00.object_detection.jsonl --output /tmp/alignment-audit.jpg
```
