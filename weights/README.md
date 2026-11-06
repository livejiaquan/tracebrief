# Weights

`yolox-s-warehouse-v4.onnx` — detector v4, the person/forklift YOLOX-s detector used by **Fast search** (`config/vision-compact-v4.json`, confidence 0.25). Fine-tuned from the official Apache-2.0 YOLOX-s COCO checkpoint on NVIDIA PhysicalAI SDG-Warehouse (OpenMDW-1.1), NVIDIA PhysicalAI-SmartSpaces (CC BY 4.0), LOCO (CC0-1.0) and the Roboflow forklift-dsitv images mirrored as keremberke/forklift-object-detection (CC BY 4.0). SHA-256 `fdcc92d11740c6a33125ef7c6d1412a7e55785a096c72a5fa899288571f05be2`, 35.8 MB. Records: `data/evaluation/stage2/detector-v4/`.

`yolox-tiny-warehouse.onnx` — the Stage I person/forklift YOLOX-tiny, used by **Fast search (Stage I detector)** (`config/vision-compact.json`, confidence 0.40) so the Stage I results stay reproducible. Fine-tuned from the official Apache-2.0 YOLOX-tiny COCO checkpoint on NVIDIA PhysicalAI SDG-Warehouse synthetic data (OpenMDW-1.1). SHA-256 `535cbfdaaf32e8a3975b34534504c49a67a1215a02141aa305cc9c689e88bf1d`.

The app refuses a file whose hash differs from its settings file. Both models take the same input and give the same output (`images` 1×3×416×416 → `raw_predictions` 1×3549×7). Details: `docs/COMPACT_DETECTOR.md`; licenses and the CC BY 4.0 attribution lines: `docs/THIRD_PARTY_NOTICES.md` and the YOLOX license copy [`docs/licenses/YOLOX-Apache-2.0.txt`](../docs/licenses/YOLOX-Apache-2.0.txt).
