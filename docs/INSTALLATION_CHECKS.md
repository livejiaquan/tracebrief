# Source installation checks

Checked 2026-10-04 on Linux, Python 3.13.5. A separate copy of the source tree and three newly created virtual environments were used. This was an isolated-environment check on the development host, not a clean operating-system installation or cross-platform certification.

## Passed

- CPU PyTorch 2.14.1 and torchvision 0.29.1 installed from the documented CPU index.
- Vision requirements installed, including the new environment's own OpenCV. Imported torch, torchvision, transformers and cv2 paths all pointed into the new environment; no old research environment was included in its Python path or `.pth` files.
- Pinned Grounding DINO processor and model weights loaded with offline flags. No new detector inference was run for this installation check.
- llama-cpp-python 0.3.36 built from source with CPU settings and two compiler jobs, then installed into its own fresh environment.
- The pinned Qwen GGUF passed SHA-256 verification. A tiny local-generation smoke returned `READY`; this checks runtime operation, not incident-draft quality.
- Full setup preflight passed; vision and generation environments passed `pip check`.
- The documented source server launch served HTML, JavaScript and health endpoints from the copied checkout, with the local generation model configured.
- All 10 review-integrity and API-handler regression tests that existed on 2026-10-04 passed in the new application environment. Later tests were added; the current count is reported by `pytest`.
- Eleven isolated HTTP checks passed, covering static assets, health, malformed JSON shapes, unknown sources, invalid uploads, path traversal and invalid observation edits.

Pinned model bytes were reused from verified local caches, copied into the new checkout's model directory. Model weights were not downloaded again in full. Small Qwen license/card files and Python packages were fetched through the documented setup path. System compiler, CMake, FFmpeg and Python were already installed, so the OS package-manager prerequisite command was not rerun.

## Repairs made during verification

Malformed observation entries and a numeric upload identifier initially caused dropped HTTP connections. They now return structured HTTP 400 errors. Regression fixtures cover these cases without running inference. Export tests also verify portable embedded image bytes and escaped reviewer text; fixture image transport checks are distinct from visual or model-accuracy evaluation.

## Packaging boundary

The release is a **source checkout / source ZIP**. A wheel audit confirmed that the present wheel does not include browser assets or worker scripts. Use the README's `PYTHONPATH=src` launch with the complete source directory; do not distribute the wheel as a standalone application.

The release builder uses an explicit allowlist, validates sample hashes, and scans included text for common secret patterns and private home-directory paths. These checks supplement manual review and license provenance; they cannot guarantee the absence of every possible sensitive string or rights issue.
