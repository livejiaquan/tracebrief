# Post-result original-profile comparison

After observing the prospective v2 results, the exact original v1 predictor/configuration was run on the **same ten new primary videos**. This comparison examines the combined effect of confidence and resize; it does not isolate their individual effects. It is explicitly post-result analysis, not a new heldout model-selection result. No third configuration or threshold grid was run.

Both profiles retain all ten runs and 100 sampled predictions. The same annotation-integrity exclusion leaves nine scorable runs and 90 frames. No outcome-based filtering, replacement runs or annotation time rescaling was used.

`freeze.json` identifies the exact original source and settings. Original source/configuration are in `../../baseline-source/`; v2 source/configuration and shared gold/source provenance are in the parent directory. `paired_comparison.json` retains both aggregates and all-ten-video runtime summaries. Workstation load varied between sequential batches, so runtime differences are descriptive observations, not a controlled speedup benchmark.

From `tracebrief/`:

```bash
python scripts/vision_evaluate.py --predictions data/evaluation/phase2/comparator_v1/predictions --compact-gold data/evaluation/phase2/gold.json --coverage-exclusions data/evaluation/phase2/comparator_v1/coverage_exclusions.json --output /tmp/comparator-rescore.json
```

Routine cohorts are different between the original and prospective evaluations. Their candidate counts cannot establish a paired reduction in routine false alarms. Neither profile measures physical safety.
