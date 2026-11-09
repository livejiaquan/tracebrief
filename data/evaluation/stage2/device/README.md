# UGen300 device records (Stage II)

**Conditions: UGen300 over USB at 0.9 A (reduced-performance mode).** Every record here was made with the UGen300 on a USB-A to USB-C cable whose source advertised 0.9 A. HailoRT warned at every start: "USB-C source electrical current advertised: 0.9A. The module may run in reduced-performance mode. For full performance, use a USB-C port that advertises 3.0A." The speeds are therefore what this setup gave, not the device's specification: Hailo publishes 1.02 s to first token and 7.04 tokens/s for the same Qwen2-VL-2B HEF, against our 2.5 s and 4.1 tokens/s. Re-measure on a 3 A USB-C port before quoting any of them.

| | |
|---|---|
| Device | ASUS UGen300 (Hailo-10H), USB, device `usb/001:024`, firmware 5.3.2 (release,app) |
| Host | Windows 11 on an Intel Core i7-10700, driven from WSL; HailoRT 5.3.2 (driver, CLI and `hailo_platform`) on Python 3.10.22, with opencv-python-headless 5.0.0 and ffmpeg 6.1.2 (conda-forge build); TraceBrief imported from this repository; each run held to four CPUs |
| Detector HEF | build o2s of detector v4: Dataflow Compiler 5.3.0, optimization level 2, 1,024 calibration frames in a seeded shuffle (seed 20261105); 13.4 MB, SHA-256 `87597f869386091b5bcdeddf0b2cb43e32ae3914be07969b3b8e96c72128053e`. Benchmark `bench-o0.json` used build o0 (level 0), SHA-256 `4a0979dad643e90ee8359bf9fe077ad019d8351a621f1d59b45bcfa67892626a` |
| VLM HEF | `Qwen2-VL-2B-Instruct.hef` for HailoRT 5.3.0 from Hailo's resource server, SHA-256 `7b2a8ee15269172a6b5e63690ac88d3e59ff9eeaacb3d99d6e1049bc59fbc407` |
| Manifest | a local manifest with both hashes (`TRACEBRIEF_HAILO_MANIFEST`); `config/hailo-manifest.json` pins the VLM's hash and records none for the detector, see below |

## Files

| File | Contents |
|---|---|
| `bench/bench-o0.json`, `bench/bench-o2s.json` | `scripts/hailo/bench_ugen300.py` on 2026-11-05 with HEFs o0 and o2s: device identity, load times, detector FPS and latency at batch 1 and 8, Qwen2-VL-2B time to first token and tokens/s for a visual check and a draft, the detector while the VLM generates in the same process. These predate the phase scheduling; their second-process run is the failure string the benchmark wrote then |
| `bench/bench-o2s-phases.json` | The same benchmark after it, with the second-process run reported as a record: not measurable, because the device admits one process |
| `hef-o2s/acceptance.json` | The build's acceptance check: per-head SNR, full precision against INT8, on the 64 validation frames |
| `hef-o2s/stage1_device_eval.json`, `kb_test_device_results.json`, `loco_test_device_results.json` | The HEF's detections on the device, scored by the same scripts as the ONNX model: the 9 Stage I runs (pair evidence), keremberke test and LOCO subset 3 |
| `hef-o2s/scores/` | The raw detections behind those three files, for the device (`device_hailo/`) and the ONNX model through the app pipeline (`onnx_app/`) on the same frames: per image for keremberke and LOCO (`*_dets.jsonl`), per run for Stage I (`stage1/primary/*/analysis.json`); `scores/MANIFEST.json` has their SHA-256 and origin |
| `hef-o2s/calibration_frames.json` | The 1,024 calibration frames in the order the compiler received them and the 64 acceptance frames: dataset, source image and its SHA-256 |
| `hef-o2s/device_agreement.json`, `device_run.json` | Device boxes against Hailo's emulators and ONNX on the same letterboxed frames; `HailoDetector` throughput and the batch-8 / batch-1 and raw-buffer parser checks |
| `clip/clip-summary.json`, `clip/clip-job.json` | One end-to-end job through the server |
| `held-guard.json` | A second process asking for the device while the first holds it: `probe_held.py` |
| `probe_held.py` | Its driver, run on the device host (`--manifest`, `--runtime`, `--out`) |
| `COPY_MANIFEST.json` | SHA-256 of every file here and, for copies, where on the device host it came from |

## What they show

Every figure at 0.9 A (reduced-performance mode). `docs/FACTS.md` recomputes the ones the README and `docs/` quote.

- **Detector HEF.** 72.1–74.2 FPS at batch 1 and 202.1–204.7 FPS at batch 8 through TraceBrief's `HailoDetector`, host NMS included (three benchmark runs); 70–71 and 183–207 FPS on the 1,015 test frames (`device_run.json`). The batch-8 path returned exactly the batch-1 detections, and TraceBrief's parser read every raw NMS buffer as HailoRT converts it.
- **Accuracy against the ONNX model.** On the 9 scorable Stage I runs the device found 52 pairs with 2 false and 7 missed, against ONNX's 53 / 2 / 6: one boundary pair lost, as in Hailo's INT8 emulator, inside the two-point acceptance rule. keremberke forklift / person AP50 0.891 / 0.481 (ONNX 0.894 / 0.511); LOCO forklift AP50 0.326 (ONNX 0.337), with no false forklift on a pallet truck or stillage. Lowest per-head SNR at build time 20.1 dB.
- **Qwen2-VL-2B.** 10.5–14.0 s to load (plus about 7 s to hash the 3.26 GB HEF), 2.54–2.56 s to first token, 4.0–4.1 tokens/s.
- **The two models together.** In one process the detector ran at 0.37–0.43 FPS while the VLM generated; a second process could not open the device, so `bench-o2s-phases.json` reports that run as not measurable.
- **One clip** (`warehouse-1`, 10 sampled frames, one candidate, models loaded first): detection with evidence frames 5.2 s, visual check 14.9 s, draft 37.2 s, 57.4 s in all; one switch from the detector to the VLM. The job-status endpoint answered in about 3 ms during detection and in up to 7.2 s while the VLM generated, because the server shares the process and the GIL with it.
- **Second process.** While one process held the device, a second was refused in 0.29 s (`HailoRuntime`), a hailo-mode job failed in 0.19 s with the same message, and the benchmark exited in 0.04 s; no HEF was hashed and the first process kept detecting.

## How they were made

On the device host's Windows interpreter (HailoRT 5.3.2, Python 3.10), with a Windows working folder for the HailoRT log, the local manifest and TraceBrief imported from this repository; ffmpeg and ffprobe named by `TRACEBRIEF_FFMPEG` / `TRACEBRIEF_FFPROBE`. The benchmark is `scripts/hailo/bench_ugen300.py --manifest <local manifest> --video warehouse-1.mp4 --output bench-o2s-phases.json`.

The clip run starts the server's HTTP handler in its own process, loads the device as the server does at start-up, submits `warehouse-1` as a Fast search job through `POST /api/jobs` and polls the job until it completes. Its `trace` comes from that polling, which waits while the VLM holds the GIL, so its times lag; `schedule.phases` has the phase times. The held-device probe keeps one `HailoRuntime` detecting while a second process tries `HailoRuntime`, a hailo-mode server job and the benchmark. `probe_held.py` is the held-device driver (`python probe_held.py --manifest <local manifest> --runtime <folder> --out held-guard.json`); the clip driver is not part of this release. Host paths in the records are shortened to `C:\tb-device`.

## The HEF is not in the repository

The detector HEF cannot be rebuilt from this repository alone, so `config/hailo-manifest.json` records no hash for it. Its 1,024 calibration frames are training images of third-party datasets that we do not redistribute (NVIDIA SDG warehouse runs 342, NVIDIA PhysicalAI-SmartSpaces 250, LOCO 334, keremberke/forklift-object-detection 98; licences in `../detector-v4/DATA_LICENSES.md`), and Hailo's Dataflow Compiler and its outputs are kept out of the repository under Hailo's licence terms. To rebuild: obtain those datasets, draw the calibration frames from the detector v4 training pools only and 64 acceptance frames from the validation pools, letterbox them with `compact_vision.preprocess`, and run `scripts/hailo/compile_yolox_hef.py --calibration-dir … --eval-dir … --optimization-level 2` with DFC 5.3.0 (default `--calibration-seed 20261105`). `hef-o2s/calibration_frames.json` names every frame we used, with its SHA-256. A rebuilt HEF will hash differently: repeat the acceptance and ONNX comparison before using it, and record its SHA-256 in a local manifest. Steps and acceptance rule: [docs/HARDWARE_PLAN.md](../../../../docs/HARDWARE_PLAN.md#detector-onnx-to-hef).
