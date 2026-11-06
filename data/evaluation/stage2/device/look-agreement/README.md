# UGen300 visual check and drafting against the CPU path (2026-11-05/06)

**Conditions: UGen300 over USB at 0.9 A (reduced-performance mode).** As for every record in [`../`](../README.md), the module sat on a USB-A to USB-C cable whose source advertised 0.9 A, and HailoRT warned at every start that it may run in reduced-performance mode. Every time here is what this setup gave, not the device's specification; re-measure on a 3 A USB-C port before quoting any of them.

**Not the same model files.** The device runs `Qwen2-VL-2B-Instruct.hef`, Hailo's compiled and quantised build of Qwen2-VL-2B-Instruct for HailoRT 5.3.0 (SHA-256 `7b2a8ee15269172a6b5e63690ac88d3e59ff9eeaacb3d99d6e1049bc59fbc407`), which takes one fixed 336×336 frame: TraceBrief pads each crop to a square and resizes it. The published CPU answers came from the GGUF build in llama.cpp (Qwen2-VL-2B-Instruct Q4_K_M with the Q8_0 projector, which reads the crop at its own resolution), so the two differ in quantisation and image input, not only in hardware. For drafting the difference is larger: the CPU path drafts with a different model, Qwen2.5-1.5B-Instruct Q4_K_M, under a JSON grammar; the device drafts with the same Qwen2-VL-2B HEF in text-only mode and no grammar. Prompt, parser, source checks and decoding (greedy, seed 19) are the app's on both sides.

## What was compared

1. **Visual check.** The frozen look-v2 check (`HailoVLM.check`: the look-v2 prompt and parser of `scripts/look_local.py`, Hailo's system message, 160 tokens) on the 148 Stage I test crops, the exact published files: each crop's SHA-256 was checked against `data/evaluation/look/FREEZE.json` before the device was opened. One resident VLM, context cleared before and after every crop, the detector not loaded. Compared with the published CPU answers in `data/evaluation/look/results.json`, whose re-run on frame-exact crops gave the same 148 answers (`../../frame-timing/`).
2. **Drafting.** 20 candidates fixed in `draft_events.json` before any device draft was made : the 7 distinct candidates with a visual-check description in earlier device job records of the bundled sample clips (checked on the device; those job records are not part of this repository) and the stored local jobs (one CPU-checked job; the other stored jobs repeat its prompt or have no visual check), then 13 of the 27 distinct Stage I candidate groups as the app forms them, with the published CPU look-v2 description as their visual check. Identical prompts were dropped because greedy decoding gives identical drafts. Each candidate was drafted by `device_pipeline.generate` on the device and by `cpu_pipeline.generate` through the resident CPU draft worker (4 threads), and both went through `generation.validate_draft` and `generation.compose`.

## Results

### Visual check (148 crops)

| | Device | CPU (published) |
|---|---|---|
| Same verdict as the other path | **144 / 148** (97.3%) | |
| `forklift` / `person` agreement, both answers readable | 145 / 147, 146 / 147 | |
| Answers that could not be read | 0 / 148 | 1 / 148 |
| Annotation-matched (gold-TP) pairs confirmed | **103 / 106** | 104 / 106 |
| Unmatched (gold-FP) pairs flagged | **1 / 7** | 1 / 7 |
| Routine crops whose "forklift" box is a rack, flagged | **0 / 3** | 2 / 3 (third unreadable) |
| Routine crops with a real parked forklift, confirmed | 14 / 14 | 14 / 14 |
| Agreement with the visual labels, both fields | **141 / 148** | 144 / 148 |
| Candidate groups as the app checks them (one key pair each), confirmed | 33 / 33 | 33 / 33 |

The four different verdicts: on one gold-TP crop (`t-054b0c8f-f120-b`, a person on the forklift) the device answered `person: no`; on the three routine crops where the detector boxed a rack upright as a forklift, the device answered `forklift: yes` and so confirmed them (the CPU flagged two; its third answer was the unreadable one). The test has only these three non-forklift boxes, so it hardly measures false-candidate detection on either path, but the device caught none of them.

Descriptions: no device description is word for word the CPU's. They are longer (median 20 words against 18; 71 of 147 over the prompt's 20-word limit, against 37), and 70 of 147 repeat the prompt's phrase "a powered industrial truck" ("A person is standing next to a forklift, which is a powered industrial truck, in a warehouse setting."); no CPU description does. They were not audited for factual errors as the Stage I sample was.

Timing (device, generation only, VLM resident; at 0.9 A):

| | Device, first run (read to the token limit) | Device, after the fix below | CPU (published, 12 threads) |
|---|---|---|---|
| Seconds per crop, median / p95 (range) | 14.9 / 38.2 (12.6–42.5) | 14.4 / 20.3 (12.6–21.7) | 8.3 / 10.2 (5.0–12.2) |
| Time to first token | 2.55 s (2.549–2.562) | | |
| Tokens per second | 3.98 (3.976–4.007) | | |
| Tokens per answer, median / p95 | 50 / 143 | 47 / 71 | 48 / 61 |

Token counts are the stream items HailoRT returned, including the end token (so 161 at the 160-token limit). The CPU token counts come from the Stage I run's raw output, which is not in this repository.

### Device-path fix: the VLM now stops reading at the end of its JSON answer

In the first run 15 of the 148 device answers went on after the closing brace of the JSON ("Please note that the description is based on the visible elements in the image …"), 4 of them to the 160-token limit: those crops took 38.2 s (median) instead of 14.4 s, for text that neither parser reads. No CPU answer did this. The parsers read from the first `{` to the last `}`, so a `}` in such a trailer makes a readable answer unreadable; one device draft in the hint trials failed exactly so (`Extra data`, a second JSON object after the first). `HailoVLM` now leaves the token stream as soon as the reply holds a complete JSON object (braces inside strings ignored) and keeps the reply up to it (`hailo_backend.ObjectEnd`, `tests/test_device_reply_end.py`). Cutting there cannot change an answer the parser could read from the whole reply; on these 15 replies the cut reply parses to the same answer (15 / 15). `probe_early_stop.json`: on 4 such crops, leaving the stream ended the generation in 0.49 s, the answer was the same, and the next full check gave the same raw reply as in the first run; 34–42 s became 13–20 s. The full re-run with the fix (`device_answers_fixed.jsonl`) gave the same answer on all 148 crops, each reply a prefix of the first run's, in 14.4 s per crop (median; 95th percentile 20.3 s instead of 38.2 s) and 2,260 s in all instead of 2,551 s.

### Drafting (20 candidates)

| | Device, draft-v2-device | CPU, draft-v2 + grammar |
|---|---|---|
| Passed the source checks (model draft shown) | **17 / 20** (selection 9 / 10, held-out 8 / 10) | **20 / 20** |
| Labeled template shown instead | 3 / 20 | 0 / 20 |
| Why rejected | a repetition loop to the 320-token limit (no JSON object); an invented evidence ID ("Unknown"); numbers absent from the input in the summary and a question | — |
| Raw reply asks whether another camera view is available / which site procedure applies | 2 / 20, 0 / 20 | 20 / 20, 20 / 20 |
| Raw reply cites every evidence ID | 15 / 20 | 20 / 20 |
| Seconds per draft, median / p95 | 43.1 / 65.7 (first run); 41.9 / 63.8 after the fix (same 20 replies) | 11.6 / 13.1 |
| Tokens per draft, median | 161 | 118.5 |
| Summary as shown, words (median) | 39.5 | 38 |

The source checks accept a device draft whose title is generic ("Review of Warehouse Key Frame" for 9 of the 20 candidates) and whose questions ask something else ("Is the forklift a powered industrial truck?"); they check citations, numbers and asserted consequences, not whether the draft follows the prompt. The device drafts were not audited by hand. The CPU drafts ran beside the first device run (which uses almost no host CPU), at 4 threads.

Reproducibility: the six device-checked candidates gave byte-identical device replies to the drafts first made for them on 2026-11-05, and the stored CPU job's draft was reproduced byte for byte by the CPU worker.

**Stricter device format hints (rejected).** Because 3 of 20 device drafts fell back to the template (none of the CPU's), two stricter device-only hints were tried, written as versioned constants in `src/tracebrief/drafting.py` (`draft-v2-device1`: spelled-out questions, every listed ID and no other, nothing after the object; `draft-v2-device2`: the same as a JSON skeleton), with `draft-v2` unchanged. The rule for choosing one (`SELECTION.json`) was written after the draft-v2-device run and before either hint ran, and used only the 10 selection candidates. Both did worse: 4 / 10 and 1 / 10 drafts passed (draft-v2-device: 9 / 10). device1 dropped `evidence_ids` or answered the questions instead of asking them; device2 dropped the questions or looped to the token limit. Neither was eligible, so the device keeps draft-v2-device, and the held-out comparison is draft-v2-device against the CPU on the other 10: 8 / 10 against 10 / 10.

## Files

| File | Contents |
|---|---|
| `device_answers.jsonl`, `device_meta.json` | First device run: per crop the crop's SHA-256, the SHA-256 of the 336×336 frame the VLM saw, raw reply, parsed answer or error, seconds, time to first token, tokens, tokens/s, verdict; run metadata with device identity, HailoRT version, both HEF hashes (from the local manifest), load time, prompt hash and decoding |
| `device_answers_fixed.jsonl`, `device_meta_fixed.json` | The same after the fix |
| `probe_early_stop.json` | The probe that left the stream at the end of the answer on 4 crops |
| `draft_events.json` | The 20 candidates (observations, unknowns, visual check), their source, prompt hash and split |
| `cpu_drafts.json` | CPU audits (`generation.generate` records: model output, checks, accepted brief, seconds) |
| `device_drafts.json`, `device_drafts_fixed.json` | Device audits with draft-v2-device (`device_pipeline.generate` records), before and after the fix |
| `device_drafts_device1_selection.json`, `device_drafts_device2_selection.json` | The two stricter hints on the selection half, with the hint text |
| `SELECTION.json` | The hint-selection rule, written before the hints ran |
| `look_agreement.json`, `draft_comparison.json` | Every number above |
| `MANIFEST.json` | SHA-256 of every file here |

## How they were made

The device runs used the Windows interpreter of the device host (HailoRT 5.3.2, Python 3.10) with TraceBrief imported from this repository and a local manifest holding both HEF hashes; one process held the device at a time. Each driver loaded `HailoRuntime` once, then called `HailoVLM.check` on every crop (after checking its SHA-256 against `data/evaluation/look/FREEZE.json`) or `device_pipeline.generate` on every candidate of `draft_events.json`; the hint trials passed `--format draft-v2-device1` / `draft-v2-device2` on the selection split, and the runs after the stream fix repeated the same commands. The CPU drafts ran `cpu_pipeline.generate` through the resident draft worker (Qwen2.5-1.5B Q4_K_M, 4 threads). The driver scripts ran outside the repository and are not part of it; host paths in the records are shortened to `C:\tb-device`.

## Limits

- One device, one power condition (0.9 A), synthetic warehouse footage. The visual labels are one labeling pass.
- Only three test crops have a non-forklift box; the device's 0 of 3 says little about how often it would confirm a false candidate, but it is worse than the CPU's 2 of 3.
- 20 draft candidates, many from the same few clips; acceptance by the source checks is not a quality audit.
- The device and CPU draft with different models, and only the CPU has grammar-constrained decoding.
