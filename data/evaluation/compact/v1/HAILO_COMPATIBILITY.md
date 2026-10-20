# Hailo-10H comparison and Stage II work

Summary: this experiment produced a two-class YOLOX-tiny ONNX model that runs on CPU, and compares it with the YOLOX-tiny configuration that Hailo publishes. The Hailo Dataflow Compiler (DFC) has not been run and no custom HEF exists yet. A model that runs as ONNX is not guaranteed to compile or to match the speed of Hailo's published HEF.

References, pinned to the commit we checked: [official YOLOX-tiny configuration](https://github.com/hailo-ai/hailo_model_zoo/blob/49039700f5c1aff9063e028c3bc55ef174371b36/hailo_model_zoo/cfg/networks/yolox_tiny.yaml) and [Hailo-10H object detection table](https://github.com/hailo-ai/hailo_model_zoo/blob/49039700f5c1aff9063e028c3bc55ef174371b36/docs/public_models/HAILO10H/HAILO10H_object_detection.rst). The table lists compiler version 5.4.0 and measurements over PCIe Gen3 x4; these are not USB or M.2 measurements of this project.

| Item | Hailo Model Zoo | This project |
| --- | --- | --- |
| Architecture | YOLOX-tiny, 80 COCO classes | Official YOLOX-tiny (depth 0.33, width 0.375), 2 classes: person, forklift |
| Input | 416×416×3, usually NHWC on the Hailo side | Fixed ONNX NCHW 1×3×416×416 |
| Preprocessing | The tiny YAML enables in-network normalization, mean [123.675, 116.28, 103.53], std [58.395, 57.12, 57.375] | Official YOLOX non-legacy BGR float32 in 0–255, top-left padding with value 114, no mean/std normalization |
| Output | Cut at the regression, objectness and class branches of each of the three scales; Hailo NMS output 80×5×100 | Single raw grid 1×3549×7; decoding and per-class NMS on CPU |
| Parser end nodes | Conv_303, Sigmoid_305/306; Conv_282, Sigmoid_284/285; Conv_261, Sigmoid_263/264 | Node names come from our exporter; the operator and Conv node list is in `onnx_manifest.json` |
| NMS | Base IoU 0.65, score 0.01; hardware post-processing | IoU 0.45; confidence chosen on our validation split only (`FREEZE.json`) |
| Calibration and compilation | COCO calibration, published HEF | Not yet calibrated, quantized or compiled |

## Stage II steps

1. Build a DFC parser configuration from the actual nodes of our ONNX file, cutting the nine prediction branches across the three scales.
2. Configure two classes and the same BGR, 0–255 and padding preprocessing used in training; disable the COCO normalization that does not apply.
3. Optimize and quantize with warehouse calibration frames kept separate from the evaluation runs, and adapt the NMS configuration to two classes.
4. Compile for the delivered Hailo-10H device and the matching DFC version.
5. Re-check boxes, confidences, pair metrics and end-to-end latency over the delivered USB or M.2 interface.

Because the class count, parser nodes, preprocessing and post-processing all differ, Hailo's COCO HEF cannot stand in for a compiled version of our weights.

The export uses ONNX opset 11 and the operators of the official model: convolution, BatchNorm, SiLU (Sigmoid × Mul), Focus (Slice × Concat), nearest-neighbour resize and concat. `onnx_manifest.json` is the authoritative operator list; support still has to be confirmed by an actual DFC parse and compile.
