# Third-party materials

Inventory updated 2026-10-20 against the bundled assets. Third-party rights are separate from any license selected for original TraceBrief code. A download link does not itself authorize use, modification or redistribution.

## Dataset and demonstration footage

**NVIDIA PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes (PhysicalAI SDG-Warehouse)**

- Origin: NVIDIA Corporation, [dataset repository](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes).
- Dataset card: synthetic NVIDIA Isaac Sim warehouse scenes; no real-world footage in this dataset. Our modifications include extraction of selected clips/frames and any displayed detections, timestamps or annotations.
- License: [OpenMDW-1.1](licenses/OpenMDW-1.1.txt), [upstream](https://openmdw.ai/license/1-1/).
- Any distributed source clips, stills or annotation portions retain this license and applicable origin/copyright notices. The demo credits NVIDIA and the dataset; this does not imply NVIDIA endorsement.
- `data/samples/manifest.json` identifies each bundled clip's source file, dataset revision and SHA-256.
- OpenMDW retains a requirement to clear other applicable rights. The dataset card and license are the available provenance; they are not a separate warranty about every underlying scene asset.

## Models

| Component | Source / license evidence | Distribution rule |
|---|---|---|
| Qwen2.5-1.5B-Instruct, official Q4_K_M GGUF (used for local draft generation) | [Official GGUF repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF), revision `91cad51170dc346986eccefdc2dd33a9da36ead9`; [Apache-2.0 copy](licenses/Qwen2.5-Apache-2.0.txt) | Separate explicit download. File `qwen2.5-1.5b-instruct-q4_k_m.gguf`; SHA-256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`. Preserve license and origin notices if redistributed. |
| Grounding DINO tiny (selected RGB detector) | [IDEA Research model](https://huggingface.co/IDEA-Research/grounding-dino-tiny), revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`; official [model card](licenses/GroundingDINO-MODEL-CARD.md) declares Apache-2.0; [upstream license copy](licenses/GroundingDINO-Apache-2.0.txt) | Separate explicit download. `model.safetensors` SHA-256 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`. This is an existing model, not a model trained by TraceBrief. |
| YOLOX-tiny person/forklift detector (`weights/yolox-tiny-warehouse.onnx`, Fast search) | Fine-tuned by TraceBrief from the official [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) tiny COCO checkpoint (Apache-2.0, commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`) on NVIDIA PhysicalAI SDG-Warehouse synthetic data (OpenMDW-1.1) | Included in this repository. ONNX SHA-256 `535cbfdaaf32e8a3975b34534504c49a67a1215a02141aa305cc9c689e88bf1d`. Keep the YOLOX Apache-2.0 notice and the dataset attribution with redistributed weights. |
| Qwen2-VL-2B-Instruct, ggml-org GGUF: Q4_K_M model and Q8_0 vision projector (used for the optional Look check) | [ggml-org GGUF repository](https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/tree/bb307c036e8a1ed7b663bbd0c35b41c4c9294cfd), revision `bb307c036e8a1ed7b663bbd0c35b41c4c9294cfd`; base model [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct); [Apache-2.0 copy](licenses/Qwen2-VL-Apache-2.0.txt) | Downloaded separately; not redistributed in this repository. `Qwen2-VL-2B-Instruct-Q4_K_M.gguf` SHA-256 `5745685d2e607a82a0696c1118e56a2a1ae0901da450fd9cd4f161c6b62867d7`; `mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf` SHA-256 `a0ad91f00a7a80dcf84d719a61b00ee2e07b71794f4ee2dfa81a254621a8c418`. Preserve license and origin notices if redistributed. |
| Hailo precompiled HEF (Stage II) | [Hailo model table](https://github.com/hailo-ai/hailo_model_zoo_genai/blob/main/docs/MODELS.rst), each row has model license/source | No HEF is represented as implemented simply because a public download exists. Retain exact model terms and compiler/runtime provenance when adopted. |

## Runtime and authoring tools

Python package versions are pinned in `scripts/*_requirements.txt` and recorded in `data/evaluation/environment.json`. Exploratory keremberke weights were not used because their license was unstated; exploratory OWLv2 weights are not part of the release. The deck and PDF use Liberation Sans (SIL OFL 1.1).

## Demo narration (authoring only)

The English demo voice is locally synthesized, using Kokoro v1.0 with stock voice `am_michael`; it is not a recording of a team member or a custom voice clone. These tools are used to author the submitted MP4 and are not dependencies of the TraceBrief review app.

- Engine: [`thewh1teagle/kokoro-onnx`](https://github.com/thewh1teagle/kokoro-onnx), installed version 0.6.1, [MIT license](licenses/Kokoro-ONNX-MIT.txt), copyright 2025 github.com/thewh1teagle.
- Model/voice assets: [`model-files-v1.1` release](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1), `kokoro-v1.0.onnx` and `voices-v1.0.bin`. The upstream [Kokoro-82M model card](https://huggingface.co/hexgrad/Kokoro-82M) declares Apache-2.0; [standard license copy](licenses/Kokoro-82M-Apache-2.0.txt). The ONNX project's license section also identifies the model as Apache 2.0.
- ONNX SHA-256: `beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a`.
- Voices SHA-256: `bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d`.
- Runtime: ONNX Runtime 1.30.0, SoundFile 0.14.0. Only generated narration is included in the final video; runtime libraries and model/voice binaries are not bundled with the app.
- Upstream training-data notices retained: Kokoro's model card credits [Koniwa](https://github.com/koniwa/koniwa) under CC BY 3.0 and [SIWIS](https://datashare.ed.ac.uk/handle/10283/2353) under CC BY 4.0. This is attribution to upstream provenance, not a claim that TraceBrief collected or trained on those recordings.

## Contest rights

The organizer's consent does not change upstream licenses. The official rules require rights in submitted material and broad organizer use terms. See [SOURCES.md](SOURCES.md), S1–S2 and S10.
