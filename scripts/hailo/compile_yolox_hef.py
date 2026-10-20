#!/usr/bin/env python3
"""Compile our two-class YOLOX-tiny ONNX to a Hailo-10H HEF with the Hailo Dataflow Compiler.

Stage II tool, not yet run: it needs the Dataflow Compiler (hailo_sdk_client, Hailo Developer Zone
login) and a GPU for optimization level 2. Usage:

    python scripts/hailo/compile_yolox_hef.py --calibration-dir CALIB/ --output build/hailo

CALIB/ holds at least 1,024 JPEG frames sampled only from the 51 training runs (never test runs).
Frames are letterboxed exactly as src/tracebrief/compact_vision.py does for the ONNX model:
BGR, 0-255, top-left placement on a 416x416 canvas filled with 114, no mean/std normalization.
"""
import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT_SIZE = 416
# Outputs of the three decode heads (stride 8, 16, 32): box regression, objectness, class scores.
# The Model Zoo YAML names (Conv_303, ...) do not exist in our export; these are our ONNX node names.
END_NODES = ["/head/reg_preds.0/Conv", "/head/Sigmoid", "/head/Sigmoid_1",
             "/head/reg_preds.1/Conv", "/head/Sigmoid_2", "/head/Sigmoid_3",
             "/head/reg_preds.2/Conv", "/head/Sigmoid_4", "/head/Sigmoid_5"]


def letterbox(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    ratio = min(INPUT_SIZE / image.shape[0], INPUT_SIZE / image.shape[1])
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    h, w = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
    canvas[:h, :w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, default=ROOT / "weights/yolox-tiny-warehouse.onnx")
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "build/hailo")
    parser.add_argument("--min-calibration", type=int, default=1024)
    args = parser.parse_args()

    from hailo_sdk_client import ClientRunner  # Hailo Dataflow Compiler

    frames = sorted(args.calibration_dir.glob("*.jpg"))
    if len(frames) < args.min_calibration:
        raise SystemExit(f"Need at least {args.min_calibration} calibration frames, found {len(frames)}.")
    calibration = np.stack([letterbox(p) for p in frames])

    args.output.mkdir(parents=True, exist_ok=True)
    runner = ClientRunner(hw_arch="hailo10h")
    runner.translate_onnx_model(str(args.onnx), "yolox_tiny_warehouse", start_node_names=["images"],
                                end_node_names=END_NODES, net_input_shapes={"images": [1, 3, INPUT_SIZE, INPUT_SIZE]})
    # NMS runs in HailoRT on the host and returns per-class boxes normalized to 0-1.
    # Check the decoder layer names in nms_yolox_warehouse.json against the parsed HAR before optimizing.
    runner.load_model_script(
        f'nms_postprocess("{HERE / "nms_yolox_warehouse.json"}", meta_arch=yolox, engine=cpu)\n'
        "model_optimization_flavor(optimization_level=2)\n")
    runner.optimize(calibration)
    runner.save_har(str(args.output / "yolox_tiny_warehouse_optimized.har"))
    hef = args.output / "yolox_tiny_warehouse.hef"
    hef.write_bytes(runner.compile())
    print(f"{hef} sha256={hashlib.sha256(hef.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
