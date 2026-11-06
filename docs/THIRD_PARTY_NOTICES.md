# Third-party materials

Inventory updated 2026-11-05 against the bundled assets (detector v4 and its training data added). Third-party rights are separate from any license selected for original TraceBrief code. A download link does not itself authorize use, modification or redistribution.

## Dataset and demonstration footage

**NVIDIA PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes (PhysicalAI SDG-Warehouse)**

- Origin: NVIDIA Corporation, [dataset repository](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Warehouse-Operations-Scenes).
- Dataset card: synthetic NVIDIA Isaac Sim warehouse scenes; no real-world footage in this dataset. Our modifications include extraction of selected clips/frames and any displayed detections, timestamps or annotations.
- License: [OpenMDW-1.1](licenses/OpenMDW-1.1.txt), [upstream](https://openmdw.ai/license/1-1/).
- Any distributed source clips, stills or annotation portions retain this license and applicable origin/copyright notices. The demo credits NVIDIA and the dataset; this does not imply NVIDIA endorsement.
- `data/samples/manifest.json` identifies each bundled clip's source file, dataset revision and SHA-256.
- OpenMDW retains a requirement to clear other applicable rights. The dataset card and license are the available provenance; they are not a separate warranty about every underlying scene asset.

## Detector training data (detector v4)

Detector v4 (`weights/yolox-s-warehouse-v4.onnx`, Fast search) was trained on the SDG-Warehouse runs above, on NVIDIA's synthetic SmartSpaces scenes and on two real-image sets, LOCO and keremberke. None of their images is redistributed in this repository; `data/evaluation/stage2/detector-v4/` holds only scores, split definitions and audits. License record of the study: [`DATA_LICENSES.md`](../data/evaluation/stage2/detector-v4/DATA_LICENSES.md). Use of these datasets does not imply endorsement by their authors.

**NVIDIA PhysicalAI-SmartSpaces, `MTMC_Tracking_2026`**

- Origin: NVIDIA Corporation, [dataset repository](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces); synthetic multi-camera warehouse scenes. The revision was not pinned when the frames were fetched.
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Used for: training (Warehouse_000/002/004/006/008/009), selection (012–016) and a re-used test (010/018/019/022).
- Changes: frames resized to 640×360, boxes below a size floor grey-filled, classes mapped to person and forklift.
- Attribution carried with the weights: "NVIDIA PhysicalAI-SmartSpaces (MTMC_Tracking_2026), NVIDIA Corporation, https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces, licensed under CC BY 4.0; frames resized and partly masked for training."

**LOCO: Logistics Objects in Context**

- Origin: Mayershofer et al., Technical University of Munich, [github.com/tum-fml/loco](https://github.com/tum-fml/loco); real images from five logistics environments, of which we use 5,097 annotated images.
- License: [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public-domain dedication); no attribution is required. The authors ask for a citation: Mayershofer, Holm, Molter, Fottner, "LOCO: Logistics Objects in Context", ICMLA 2020.
- Used for: training (subsets 1, 2, 4 and 5 except three sessions), selection (those three sessions) and the held-out test site (subset 3).
- Changes: LOCO has no person class; persons in the training and selection images were pseudo-labelled by the YOLOX-L COCO checkpoint below, and its uncertain boxes were grey-filled as ignore regions. Training copies are stored 640 px wide.

**keremberke/forklift-object-detection (Roboflow Universe "forklift-dsitv")**

- Origin: [Hugging Face dataset](https://huggingface.co/datasets/keremberke/forklift-object-detection), revision `2c0f4e8e9085f3079b80137e780aab0e54936cfc`, a mirror of [Roboflow Universe forklift-dsitv](https://universe.roboflow.com/mohamed-traore-2ekkp/forklift-dsitv) v1 by Mohamed Traore (2022); images originally from images.cv.
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), stated in the Roboflow export and in every image's COCO license field.
- Used for: training (train split, 295 images), selection (valid, 84) and test (test, 42).
- Changes: person ignore regions from the YOLOX-L teacher grey-filled in the training copies.
- Attribution carried with the weights: "Forklift Dataset, Mohamed Traore, Roboflow Universe, https://universe.roboflow.com/mohamed-traore-2ekkp/forklift-dsitv, CC BY 4.0; mirrored as keremberke/forklift-object-detection; ignore regions grey-filled for training."

**Official YOLOX COCO checkpoints** (release 0.1.1rc0, [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [Apache-2.0](licenses/YOLOX-Apache-2.0.txt))

| Checkpoint | SHA-256 | Used for |
|---|---|---|
| `yolox_s.pth` | `f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30` | Initialisation of detector v4 (v4-S) |
| `yolox_tiny.pth` | `9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0` | Initialisation of the Stage I detector and of the v4-T arm (not adopted) |
| `yolox_l.pth` | `1e6b7fa6240375370b2a8a8eab9066b3cdd43fd1d0bfa8d2027fd3a51def2917` | Teacher only: person pseudo-labels on the LOCO training and selection images and ignore regions on keremberke train/valid; not part of any shipped model |

The checkpoints are not redistributed. Their authors trained them on MS COCO (annotations CC BY 4.0, images under their Flickr terms); we use only the released weights.

## Models

| Component | Source / license evidence | Distribution rule |
|---|---|---|
| Qwen2.5-1.5B-Instruct, official Q4_K_M GGUF (used for local draft generation on the CPU path) | [Official GGUF repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF), revision `91cad51170dc346986eccefdc2dd33a9da36ead9`; [Apache-2.0 copy](licenses/Qwen2.5-Apache-2.0.txt) | Separate explicit download. File `qwen2.5-1.5b-instruct-q4_k_m.gguf`; SHA-256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`. Preserve license and origin notices if redistributed. |
| Grounding DINO tiny (selected RGB detector) | [IDEA Research model](https://huggingface.co/IDEA-Research/grounding-dino-tiny), revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`; official [model card](licenses/GroundingDINO-MODEL-CARD.md) declares Apache-2.0; [upstream license copy](licenses/GroundingDINO-Apache-2.0.txt) | Separate explicit download. `model.safetensors` SHA-256 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`. This is an existing model, not a model trained by TraceBrief. |
| Detector v4, YOLOX-s person/forklift detector (`weights/yolox-s-warehouse-v4.onnx`, Fast search) | Fine-tuned by TraceBrief from the official [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) s COCO checkpoint (Apache-2.0, code commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`) on NVIDIA PhysicalAI SDG-Warehouse (OpenMDW-1.1), NVIDIA PhysicalAI-SmartSpaces (CC BY 4.0), LOCO (CC0-1.0) and keremberke/Roboflow forklift-dsitv (CC BY 4.0), with person pseudo-labels from the official YOLOX-L COCO checkpoint (Apache-2.0); see "Detector training data" above | Included in this repository. ONNX SHA-256 `fdcc92d11740c6a33125ef7c6d1412a7e55785a096c72a5fa899288571f05be2`. The weights are derived from CC BY 4.0 data: keep the two attribution lines above, the YOLOX Apache-2.0 notice and the OpenMDW-1.1 notice with redistributed weights. |
| Stage I YOLOX-tiny person/forklift detector (`weights/yolox-tiny-warehouse.onnx`, Fast search (Stage I detector)) | Fine-tuned by TraceBrief from the official [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) tiny COCO checkpoint (Apache-2.0, commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`) on NVIDIA PhysicalAI SDG-Warehouse synthetic data (OpenMDW-1.1) | Included in this repository. ONNX SHA-256 `535cbfdaaf32e8a3975b34534504c49a67a1215a02141aa305cc9c689e88bf1d`. Keep the YOLOX Apache-2.0 notice and the dataset attribution with redistributed weights. |
| Qwen2-VL-2B-Instruct, ggml-org GGUF: Q4_K_M model and Q8_0 vision projector (used for the optional Look check) | [ggml-org GGUF repository](https://huggingface.co/ggml-org/Qwen2-VL-2B-Instruct-GGUF/tree/bb307c036e8a1ed7b663bbd0c35b41c4c9294cfd), revision `bb307c036e8a1ed7b663bbd0c35b41c4c9294cfd`; base model [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct); [Apache-2.0 copy](licenses/Qwen2-VL-Apache-2.0.txt) | Downloaded separately; not redistributed in this repository. `Qwen2-VL-2B-Instruct-Q4_K_M.gguf` SHA-256 `5745685d2e607a82a0696c1118e56a2a1ae0901da450fd9cd4f161c6b62867d7`; `mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf` SHA-256 `a0ad91f00a7a80dcf84d719a61b00ee2e07b71794f4ee2dfa81a254621a8c418`. Preserve license and origin notices if redistributed. |
| Hailo precompiled HEF (Stage II): Qwen2-VL-2B-Instruct for HailoRT 5.3.0 | [Hailo model table](https://github.com/hailo-ai/hailo_model_zoo_genai/blob/main/docs/MODELS.rst), each row has model license/source; base model [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct), [Apache-2.0 copy](licenses/Qwen2-VL-Apache-2.0.txt) | Downloaded separately from Hailo's resource server (`https://dev-public.hailo.ai/v5.3.0/blob/Qwen2-VL-2B-Instruct.hef`); not redistributed. The file used on the device hashed to SHA-256 `7b2a8ee15269172a6b5e63690ac88d3e59ff9eeaacb3d99d6e1049bc59fbc407` (Hailo publishes no hash). Retain exact model terms and compiler/runtime provenance. |
| Detector v4 HEF (Stage II) | Compiled by us from `weights/yolox-s-warehouse-v4.onnx` with Hailo's Dataflow Compiler 5.3.0 (`scripts/hailo/compile_yolox_hef.py`), calibrated on training frames of the datasets above | Not in this repository: we keep the Dataflow Compiler and what it produces out of it under Hailo's licence terms, and the calibration frames are third-party data. Rebuild it as described in [HARDWARE_PLAN.md](HARDWARE_PLAN.md). |

## Runtime and authoring tools

Python package versions are pinned in `scripts/*_requirements.txt` and recorded in `data/evaluation/environment.json`. Exploratory keremberke detector weights (a published model, not the keremberke dataset above) were not used because their license was unstated; exploratory OWLv2 weights are not part of the release. The deck and PDF use Liberation Sans (SIL OFL 1.1).

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
