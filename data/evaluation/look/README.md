# Look step: visual check of candidate pairs

For each predicted person–forklift pair, the Look step crops the source frame around both boxes (`src/tracebrief/look.py`: margin 0.6, longest side ≤ 448 px) and asks a local vision-language model, Qwen2-VL-2B-Instruct (GGUF Q4_K_M + mmproj Q8_0, llama.cpp on CPU), whether a forklift and a person are in the crop, plus one sentence of description (`scripts/look_local.py`, prompt `look-v2`, temperature 0, seed 19). If the answer is `forklift: no` or `person: no`, the app suggests the reviewer check for a false candidate. It never dismisses anything automatically.

## Protocol and timeline (UTC, 2026-10-20)

1. **Development (05:43–07:09).** We used 39 crops from cohorts already used for calibration: 26 from the original routine clips (22 shelf/rack boxes predicted as forklifts in R02–R09, including 2 built from R09 detections; 4 with the parked forklift in R01) and 13 from the original 24-run test (10 gold-TP, 3 gold-FP). We recorded visual labels before reading any model output.
   - `look-v2` flagged only 6 of the 22 shelf/rack crops. On 9 of them the model left out the `forklift` key, so the answer could not be read. It confirmed 17 of 17 crops with a real forklift.
   - We made one prompt-only revision, `look-v3`, after writing down the rule for keeping it: v3 needed more exact matches than v2 (23 of 39) and had to confirm at least 15 of 17 real-forklift crops. v3 flagged all 22 shelf/rack crops and reached 32 exact matches, but confirmed only 11 of 17 real-forklift crops: it answered "no forklift" on 4 overhead reach trucks and gave 2 answers that could not be read. Under the rule we kept `look-v2` and restored `scripts/look_local.py` byte for byte. Crop logic, model, decoding and parse rules were never changed. Details are in `dev.json`.
2. **Freeze (07:10:20).** `FREEZE.json` records the SHA-256 of `look_local.py`, `look.py`, the model, the projector, the prompt, and every test item id with its crop hash. `look.py` gained app-integration functions before the freeze; the crop functions themselves did not change.
3. **Visual labels (07:12:21).** We labeled all 148 test crops from contact sheets that showed only the crop, thin box outlines and the item id. These are visual labels recorded before the model ran: `test_labels.json`, SHA-256 `8771363878b2…cfb2124`, which also cites the freeze hash.
4. **Test run (07:12:31–07:33:00).** The frozen step ran once on all 148 crops, with 12 threads.

## Test set (prospective; never used for tuning)

The test set is every predicted pair (one per run, frame and pair) from the phase2 cohort. Broader search contributed 70 primary pairs. Stricter search contributed 66 (50 primary, 16 routine) and Fast search 62 (61 primary, 1 routine), for 198 pairs in total. Pairs on the same run and frame whose boxes matched at IoU ≥ 0.9 share one crop, which leaves **148 crops** (131 primary, 17 routine). We labeled the primary pairs with the existing rule (both boxes at IoU ≥ 0.50 against `phase2/gold.json`): 106 gold-TP crops, 7 gold-FP crops and 18 from the annotation-excluded run, which we ran but did not score. The visual labels show both a forklift and a person in 145 crops. The 3 crops with no forklift are all routine crops where a rack upright or rack guard was boxed as a forklift.

## Results

| Measure | Result |
|---|---|
| Gold-TP pairs confirmed (forklift = yes and person = yes) | 104 / 106 crops (98.1%); Broader 53/55, Stricter 37/37, Fast 51/51 |
| Gold-FP pairs flagged | 1 / 7 crops; Broader 1/6, Stricter 1/4, Fast 0/1 |
| Routine pairs whose forklift box is not a forklift, flagged | 2 / 3 (the third answer could not be read) |
| Routine pairs with a real parked forklift, confirmed | 14 / 14 |
| Agreement with visual labels | forklift 147/147, person 144/147 (readable answers); both fields 144/148 |
| Answers that could not be read | 1 / 148 (`forklift` key missing on a shelf crop) |
| Descriptions with a clear factual error (20 random crops) | 5 / 20 |
| Seconds per crop | median 8.3, p90 10.0 (range 5.0–12.2); model load 1.1 s |

- **All 5 suggestions:** 2 were correct (rack crops answered `forklift: no`). 3 were wrong: the model answered `person: no` for a worker standing on the forklift's operator platform (one run, 05601faa), and one of those 3 happened to be a gold-FP pair. All 7 gold-FP crops do show a real forklift and person. They are box-localization misses, which this check is not designed to catch. See `examples_flagged.jpg` and `examples_confirmed.jpg`.
- **If the reviewer accepted every suggestion (hypothetical; 9 scorable runs):** Broader pair precision/recall would go from 90.2%/93.2% to 91.4%/89.8% (TP 55→53, FP 6→5). Stricter would go from 90.2%/62.7% to 92.5%/62.7% (FP 4→3). Fast would stay at 98.1%/86.4%.
- **As the app runs it** (one key pair per candidate group): all 33 groups (11 per profile) were confirmed and none were flagged. The routine shelf pairs never formed a candidate group.
- **Descriptions:** the 5 errors are a forklift mentioned in a rack crop the model itself answered `forklift: no` for, a box "held" that was on the floor, "sitting" for a standing or crouching person (twice), and the forklift's blue body called "a blue box". `summary.json` lists all 20 with notes.
- **Machine:** Intel Core i7-10700 (8 cores / 16 threads), WSL2, CPU only. The 1-minute load average was 1.6 before the run and 11–14 during it; the Look process itself accounted for about 11.6 cores. The development v2 run used 8 threads.

## Limits

- The test clips are synthetic, and only 3 test crops had a non-forklift box, so the test hardly measures false-candidate detection. On development crops, `look-v2` caught only 6 of 22 shelf/rack boxes.
- The visual labels come from one labeling pass over contact sheets, not independent ground truth. Gold-FP pairs here are localization errors that a presence check cannot catch.
- The model misses people standing on a forklift's operator platform, and 5 of 20 sampled descriptions contain a clear factual error. Descriptions should be shown as unverified notes.
- Timings are CPU batch measurements on a shared workstation. No UGen300 measurement is claimed.

## Files

`FREEZE.json` (freeze record), `test_labels.json` (visual labels), `results.json` (per crop: run, frame, profiles, gold status, visual label, model answer, description, raw output, seconds), `summary.json` (every metric with numerator and denominator, the description audit, timing and the development summary), `dev.json` (development crops, labels, v2/v3 answers, decision rule), and two captioned contact sheets.
