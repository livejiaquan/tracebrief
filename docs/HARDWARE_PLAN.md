# UGen300 integration plan

Stage I runs every model on the host CPU (Intel Core i7-10700). Stage II moves the two neural workloads onto one ASUS UGen300 (Hailo-10H, 8 GB) and leaves decoding, drafting, review and export on the host. Nothing in this repository has run on a UGen300 yet; the device code in `src/tracebrief/hailo_backend.py` and `scripts/hailo/` is written against the HailoRT 5.4 Python API and the Hailo Dataflow Compiler and is untested on hardware. Source IDs refer to [SOURCES.md](SOURCES.md).

## Target split

| Work | Stage I (now) | Stage II target | Why |
|---|---|---|---|
| Decode and sample frames | Host, ffmpeg | Host | The site PC already decodes its NVR exports. |
| Person/forklift detection | YOLOX-tiny ONNX on CPU, ≈20 ms per frame | YOLOX-tiny HEF on Hailo-10H (vision app) | Hailo lists `yolox_tiny` for Hailo-10H at about 400 FPS (PCIe, batch 1) [S18]. Lets the site run every camera at 5 fps without loading the host. |
| Visual check of each candidate | Qwen2-VL-2B GGUF on CPU | Qwen2-VL-2B HEF on Hailo-10H (GenAI app) | Hailo publishes Qwen2-VL-2B-Instruct for Hailo-10H: time to first token 1.02 s, 7.04 tokens/s, 2,048-token context [S18]. |
| Draft title and scene sentence | Qwen2.5-1.5B GGUF on CPU | Host, or the same Qwen2-VL-2B in text-only mode | Hailo documents one vision app and one GenAI app sharing a device, but not two GenAI apps [S18]; text drafting is short and runs acceptably on the CPU. |
| Checks, review UI, export | Host | Host | Rule checks and the audit trail stay where the record is stored. |
| Broader search (Grounding DINO) | Host | Host | No Hailo path; kept as the high-recall fallback for uncalibrated camera views. |

Performance envelope from published figures (estimates, to be measured): one visual check ≈ 1.02 s + about 40 output tokens ÷ 7.04 tokens/s ≈ 6.7 s per candidate. Detector demand for 8 cameras at 5 fps is 40 frames/s. Users report that small YOLO models over UGen300 USB reach about 170–210 FPS at batch 1, below the PCIe figures [S19], which still leaves about four times headroom. The visual check, not the detector, is the bottleneck, so it runs only on candidates.

## Detector: ONNX to HEF

Our checkpoint (`weights/yolox-tiny-warehouse.onnx`) is opset 11 with a single `raw_predictions` output `[1, 3549, 7]` (two classes, 416×416 input, BGR 0–255, top-left letterbox padded with 114). Its operator mix matches the Model Zoo `yolox_tiny` ONNX, so the standard parse → optimize → compile route applies. `scripts/hailo/compile_yolox_hef.py` runs it with the Dataflow Compiler's `ClientRunner`:

1. Parse with explicit end nodes before the decode head (the nine `reg_preds`/`Sigmoid` outputs of the three strides); the Model Zoo YAML's node names do not exist in our export and its ImageNet normalization must not be applied to this non-legacy model.
2. Add `nms_postprocess(..., meta_arch=yolox, engine=cpu)` with the Model Zoo YOLOX NMS config changed to two classes. HailoRT then runs NMS on the host and returns per-class boxes normalized to 0–1.
3. Optimize at level 2 with at least 1,024 calibration frames drawn only from the 51 training runs, then compile for `hailo10h`. Optimization above level 0 needs a GPU, so we compile on our NCHC GPU allocation.
4. Accept the HEF only if pair precision and recall on the same nine scorable test runs stay within two points of the ONNX model and every output's SNR is at least 10 dB.

## Visual check on the device

`hailo_platform.genai.VLM(vdevice, "Qwen2-VL-2B-Instruct.hef")` takes 336×336 RGB frames. Hailo's chat example centre-crops images, which would cut off the person or the forklift in our pair crops, so we pad each crop to a square before resizing. The context is cleared after every crop so the system message stays valid and nothing carries over between candidates. Both the detector and the VLM create their `VDevice` with the same `group_id` so they share one device. The HEF (about 3.3 GB) is downloadable from Hailo's public resource server.

## Interface choice: USB

We will request the USB version. It plugs into the PC that already stores the NVR exports without opening the case, which is the installation IT departments approve most easily. Our demand is well inside the USB throughput users report, and the visual check is compute-bound, not bandwidth-bound. Conditions we will hold to: a direct USB-C 10 Gbps port able to supply 3 A, the supplied cable, no hub, firmware updated before testing, and a 12-hour thermal soak (users report shutdowns in long runs inside the stock enclosure) [S19].

## Stage II measurements

| Question | Measure | Pass |
|---|---|---|
| Detector equivalence | Pair precision/recall, same 9 test runs, ONNX vs HEF | Within 2 points |
| Visual check on device | Seconds per candidate, parse failures, agreement with the CPU answers on the same crops | ≤ 7 s per candidate; same verdict on ≥ 95% of crops |
| Shared device | Detector FPS while the VLM is running; errors over a 12-hour soak | No failed jobs; detector ≥ 40 FPS |
| Host headroom | Host CPU load during an 8-camera overnight batch, host only vs host + UGen300 | Lower host load with the same outputs |
| Whole job | Video in to reviewed draft, per candidate and per hour of footage | Reported for both configurations |

Every device run records HailoRT and HEF versions with hashes, the raw outputs and the host-side comparison. A run that falls back to the CPU is labeled as a CPU run.
