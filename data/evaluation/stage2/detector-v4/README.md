# Detector v4: pre-registered evaluation records

Detector v4 is the person/forklift detector behind **Fast search** since 2026-11-05: v4-S, YOLOX-s at 416×416, confidence 0.25 (`config/vision-compact-v4.json`, `weights/yolox-s-warehouse-v4.onnx`, SHA-256 `fdcc92d1…f05be2`). The Stage I YOLOX-tiny remains selectable as **Fast search (Stage I detector)** (`config/vision-compact.json`); its records are in `../../compact/v1/`. Summary and weaknesses: [docs/COMPACT_DETECTOR.md](../../../../docs/COMPACT_DETECTOR.md#detector-v4).

The protocol, criteria and decision rule were registered (`PREREG.json`, 2026-11-04T18:59:55Z) before any training step and before any new test-set inference. Each model was scored once per test set after its freeze. v4-S passed every criterion; v4-T (YOLOX-tiny) failed C2a, forklift recall at the held-out real site.

| File | Contents |
|---|---|
| `PROTOCOL.md`, `PREREG.json` | Question, data, splits, training, selection rule, criteria and decision rule; `PREREG.json` adds timestamps and the SHA-256 of every script and input |
| `DATA_LICENSES.md` | Licenses and attribution of every dataset and checkpoint used |
| `RESULTS.md` | The study report: verdict, all test tables, failure examples, cost and every deviation from the protocol (§8) |
| `freeze/v4-S/`, `freeze/v4-T/` | Freeze records (epoch, confidence, validation cells, ONNX I/O and parity, Hailo end-node check) and the app settings of each arm |
| `out/summary.json`, `out/tables.md` | All test results and criteria verdicts, for the Stage I model ("old"), v3, v4-T and v4-S |
| `out/<model>/*.json` | Per-model results: SDG Stage I frames, SDG events, the 24-run v1 baseline frames, LOCO subset 3, keremberke test, SmartSpaces (v4 arms; old and v3 come from the detector-v3 study) |
| `out/cpu_latency*.json` | CPU latency, two passes in opposite order |
| `out/overlap_audit.json`, `out/kb_dup_audit.json`, `out/teacher_check.json`, `out/harness_check_old.json`, `out/oracle_selftest_*.json` | Split overlap and near-duplicate audits, the YOLOX-L person teacher check, the app-harness reproduction check and the scorer self-tests |
| `app_check.json` | Check made when v4-S was adopted: both settings files through `scripts/vision_worker.py` on the three bundled samples, and four CPU jobs in the app (Fast search and the Stage I profile) |
| `COPY_MANIFEST.json` | SHA-256 of each file here and of the original it was copied from |

Not included: training and scoring code, checkpoints and per-epoch runs, per-image detection caches, SmartSpaces caches, the rendered failure images and any downloaded image or video. The training and scoring scripts, whose hashes are in `PREREG.json`, stay in our study folder, so this bundle supports reading and checking the reported numbers, not re-scoring them. Local absolute paths were shortened to workspace-relative ones (`work/…`, `tracebrief/…`), so some copies differ from their originals' hashes; `COPY_MANIFEST.json` lists both. `RESULTS.md` was written before adoption and says that nothing in `tracebrief/` had been changed; that was true at the time.

The records hold scores, identifiers and box statistics derived from NVIDIA SDG-Warehouse (OpenMDW-1.1), NVIDIA PhysicalAI-SmartSpaces (CC BY 4.0), LOCO (CC0-1.0) and keremberke/forklift-object-detection (Roboflow forklift-dsitv, CC BY 4.0); attribution: [docs/THIRD_PARTY_NOTICES.md](../../../../docs/THIRD_PARTY_NOTICES.md).
