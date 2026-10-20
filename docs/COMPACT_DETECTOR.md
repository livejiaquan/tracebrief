# Compact detector (Fast search)

**Summary.** A purpose-trained YOLOX-tiny detector for people and forklifts showed about 90 times lower observed vision wall time than Grounding DINO on the same shared CPU host (not a controlled benchmark), with higher pair precision and somewhat lower recall, on the same synthetic test clips. Its forklift-labeled training views were overhead (ceiling) views, and it misses parked forklifts seen from ground-level cameras. It is offered as the optional **Fast search** profile; **Broader search** remains the default.

## Why

Grounding DINO has no verified Hailo deployment path in this prototype and took 37–71 seconds (median) of vision wall time on CPU per 10-second clip. YOLOX-tiny is published for Hailo-10H (the chip in ASUS UGen300) in the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_object_detection.rst), so a person/forklift YOLOX-tiny has a direct Stage II compile path. No HEF has been compiled yet; see `data/evaluation/compact/v1/HAILO_COMPATIBILITY.md`.

## How it was built

- Official [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) tiny COCO checkpoint, fine-tuned on CPU at 416×416.
- Training data: 64 new NVIDIA synthetic runs (43 near-miss ceiling views, 21 routine box-pickup ground views) that do not overlap any evaluation run; split 51 train / 13 validation **by run**. Overlap audit: 0.
- Epoch and confidence (0.40) were chosen on validation only and frozen (`FREEZE.json`, with weight hashes) before any evaluation clip was processed.

## Results on the same test clips

Same 9 scorable runs / 90 sampled frames, same scorer and coverage exclusion as the Broader/Stricter comparison. A pair counts only if both predicted boxes match annotated boxes (IoU ≥ 0.5).

| Profile | Pair TP / FP / FN | Pair precision | Pair recall | Median vision time per 10 s clip |
|---|---|---|---|---|
| Broader (Grounding DINO) | 55 / 6 / 4 | 90.2% | 93.2% | 71.3 s |
| Stricter (Grounding DINO) | 37 / 4 / 22 | 90.2% | 62.7% | 36.6 s |
| **Fast (YOLOX-tiny v1)** | **51 / 1 / 8** | **98.1%** | **86.4%** | **0.79 s** (≈20 ms per frame) |

Times are medians over all 10 test clips (including the one excluded from scoring) on the same shared host; Grounding DINO timings include model loading, the compact timing includes its 0.1 s load. These are batch CPU measurements, not UGen300 results. At 5 fps the compact detector processes a 10-second clip in about 2 s, which reduces the risk of missing brief interactions between 1 Hz samples.

End to end in the app (recorded demonstration job `3ee3d1fe1f11435f`, `warehouse-1` sample, 2026-10-12): 19.0 s from video to draft, of which local Qwen drafting took 15.8 s (4.2 s model load, 11.3 s generation). An earlier run the same day took 17.4 s (15.3 s drafting). On CPU, language generation is now the bottleneck, which is the part the Hailo GenAI path for Qwen2.5-1.5B targets.

## Limits found by inspection

On ten routine ground-level clips the detector produced no candidate groups, but visual inspection showed it **did not box clearly visible parked forklifts in at least three clips** and produced two low-confidence person boxes on objects. Likely causes: forklift labels came only from overhead near-miss views, while the ground-level routine training clips supplied person labels but omit parked forklifts, which taught the model to treat them as background.

A revision (v2) removed those routine clips and added two ground-level views per near-miss run. It raised test recall to 89.8% (53 / 3 / 6) but reintroduced shelves and pillars boxed as forklifts in routine clips, so it was **not adopted**. v2 was a revision made after seeing v1 results, not an independent test. Records: `data/evaluation/compact/v2/`.

Deployment implication: a site pilot should include a short calibration step on the site's own camera view (a few dozen labeled clips) before relying on Fast search; until then Broader search remains the default for uncalibrated camera views; real-site accuracy is still to be tested.
