# Frequently asked questions

**Who is TraceBrief for?**
The warehouse supervisor or EHS lead who has to answer "what happened?" after a forklift close call is reported: find the right seconds in the recordings, check what is visible and hand over a written record.

**How is this different from a safety-analytics platform such as Protex AI?**
Those platforms monitor sites continuously and are sold as site-wide systems. TraceBrief starts with the job every site already does by hand: reviewing a recording and writing it up. It runs on the PC that already stores the NVR exports, links every sentence of the report to a frame, and keeps the full edit history. Footage never leaves the site.

**Why does it need a vision-language model?**
The detector only knows where boxes are. In our tests it also boxed shelving as forklifts and flagged parked forklifts. The visual check looks at a crop around each detected pair and says whether a person and a forklift are actually there and what they are doing. That turns "boxes overlap at 3 s" into a sentence a supervisor can check against the frame. On held-out crops it confirmed <!-- fact: look.tp_confirmed -->104 of 106 real pairs; it does not yet reject shelving reliably, and about one description in four contained a factual error, so its output is shown as a model note. Nothing is dismissed automatically.

**Why UGen300 rather than the CPU alone?**
On a CPU the prototype handles one clip at a time: the visual check takes a median <!-- fact: look.median_s -->8.3 s per candidate on 12 threads, and every model shares the CPU with everything else on the site PC. On the UGen300 our detector v4 HEF ran at <!-- fact: device.detector_fps_b8 -->202–205 FPS (batch 8) and Qwen2-VL-2B at <!-- fact: device.vlm_speed -->2.54–2.56 s to first token and 4.0–4.1 tokens/s, measured on a USB port that supplied only 0.9 A, so the module ran in reduced-performance mode and these numbers are a floor to re-measure at 3 A. That is the split we use: the detector, the visual check and the draft (the same Qwen2-VL-2B in text-only mode) on the UGen300, taking turns in phases; decoding, the source checks, review and export on the host. Details and caveats: [HARDWARE_PLAN.md](HARDWARE_PLAN.md).

**Are these real incidents?**
No. The bundled clips and the Stage I tests are NVIDIA's synthetic warehouse dataset, labeled as synthetic everywhere it appears. Detector v4 was also tested on public real-image sets (LOCO and keremberke), which show warehouses and forklifts, not incidents. A one-camera site pilot is the next step after device integration.

**Can the tool decide whether a collision or a violation happened?**
No. A candidate means a predicted person box and forklift box were close in the image. Distance, contact and responsibility are decided by the reviewer from the footage.

**What happens when the detector misses someone?**
A missed detection can hide a candidate, so the default search profile favors recall, and every miss in our test set is listed in [VISION_RESULTS.md](VISION_RESULTS.md).

**How were the numbers measured?**
Training, tuning and test footage come from separate simulation runs. Settings were frozen before the test clips were processed, every prediction is kept, and the scripts in `scripts/` re-score each published number from the raw records in `data/evaluation/`. [FACTS.md](FACTS.md) lists every headline number with its source file and frozen setting, and a test checks the documents against it. The visual-check prompt was frozen on development crops before the test crops were labeled and run; the labels come from a single pass over contact sheets with no independent second labeler, and an independent re-label is part of Stage II. See `data/evaluation/look/`.
