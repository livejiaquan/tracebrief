> **Version scope:** This directory retains the original v1 frozen evaluation. Its 24 primary and ten routine runs subsequently informed calibration. Current prospective v2 results are separate in [phase2/README.md](phase2/README.md); the original score is not current v2 performance. Exact v1 source/configuration are in `baseline-source/`, and failed calibration probes remain in `calibration/`.

# Inspectable synthetic evaluation

This bundle supports re-scoring the reported results without downloading videos, weights or private workspace files. The original model outputs are preserved, with local paths and JPEG references removed for portability.

- `freeze.json`: model/configuration and 24 primary run IDs fixed before heldout inference; source implementation SHA-256.
- `predictions/`: all primary per-frame predictions, including misses and no-candidate runs.
- `gold.json`: sampled simulator 2D boxes derived from public annotation files, retaining original frame indices. Mapping: Worker→person, forklift→forklift; boxes clipped to 1920×1080. These are NOT physical safety labels.
- `sources.json`: exact source URLs pinned to dataset revision, tar members, file sizes and SHA-256 hashes; separate primary and routine cohorts.
- `results.json`: primary sampled-frame agreement, class localization and supplementary pair-evidence scoring; all per-run results and frame failures.
- `uncertainty.json`: bootstrap of entire runs, not individual correlated frames; observed shared-workload CPU timings.
- `routine/`: separate preselected routine cohort outputs and visual/annotation audit. Missing parked-forklift annotations are never converted into negative ground truth.
- `model_manifest.json`, `environment.json`: model-file hashes and measured runtime versions.

From the project root, cached re-scoring uses only the Python standard library:

```bash
python scripts/vision_evaluate.py --predictions data/evaluation/predictions --compact-gold data/evaluation/gold.json --output /tmp/tracebrief-rescored.json
python scripts/vision_summarize.py /tmp/tracebrief-rescored.json --output /tmp/tracebrief-uncertainty.json
```

To reproduce inference, first install the vision environment and pinned model as described in `docs/VISION.md`. Then explicitly retrieve individual videos (roughly 400 MB primary cohort; roughly 223 MB routine cohort). The downloader reads tar headers and selected byte ranges; a 1 GiB default aggregate response-body cap prevents whole-shard downloads:

```bash
python scripts/vision_fetch_eval.py --cohort primary --output data/evaluation-videos --max-mib 1024
python scripts/vision_analyze.py data/evaluation-videos/CHOSEN_VIDEO.rgb.mp4 --output .runtime/reproduced/analysis.json
```

The 24 primary clips use a frozen 1 Hz RGB predictor, never simulator annotations at inference. Scores measure a 2D box-gap proxy; they do not measure near-miss, collision or unsafe-action classification. Same generator/assets create shared structure across runs. Real warehouse generalization, useful review-time savings, independent human acceptance and UGen300 speed/power are unverified. Stage I CPU timings reflect a shared workstation workload.

The stricter pair-evidence metric was added during the evaluator audit after predictor freeze: both predicted actor boxes must match a gold proximity pair at IoU≥0.50. It is supplementary; the prespecified binary frame agreement remains reported separately. Temporal candidate groups are not independently labeled event truth.

Dataset-derived boxes and any redistributed NVIDIA clips are provided with the original [OpenMDW-1.1 license](../../docs/licenses/OpenMDW-1.1.txt). Origin: NVIDIA, *PhysicalAI WorldModel Synthetic Warehouse Operations Scenes*. Preserve this notice and the license on redistribution. Grounding DINO model rights are documented separately; weights are not in this bundle.
