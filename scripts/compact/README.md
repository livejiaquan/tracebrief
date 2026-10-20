# Compact detector experiment scripts (as run)

These scripts record how the YOLOX-tiny person/forklift detector in `weights/` was built on 2026-10-12. They are kept for provenance. Each script reads and writes its working files (checkpoints, COCO frames, logs) in its own directory, as in the original run; input locations that were specific to our machine are command-line options (`--metadata`, `--videos`, `--python`, `--model-path`), so a rerun needs those inputs prepared first. Some paths are relative to the current directory (for example `tracebrief/data/evaluation/sources.json`), so start the scripts from the directory that contains the `tracebrief` checkout.

Log messages were translated to English after the run, and the machine-specific input paths were replaced by these options; SHA-256 values in `data/evaluation/compact/v1/experiment_code_sha256.json` refer to the as-run files.

1. `prepare_data.py` — selects new NVIDIA synthetic runs that do not overlap any evaluation run, fetches annotations and video by byte range, and checks frame/annotation alignment.
2. `build_coco.py` — samples frames at 2 fps and writes COCO-format person/forklift boxes, split by simulation run.
3. `train_cpu.py` — fine-tunes the official Apache-2.0 YOLOX-tiny COCO checkpoint on CPU at 416×416 with early stopping on validation recall.
4. `freeze_export.py` — selects the epoch and confidence on validation only, writes the freeze record and exports ONNX (opset 11).
5. `run_evaluation.py` — runs the frozen ONNX model through `scripts/vision_worker.py` on the same evaluation clips and scores it with `scripts/vision_evaluate.py`.

Results and freeze records: `data/evaluation/compact/`. Summary: `docs/COMPACT_DETECTOR.md`.
