#!/usr/bin/env python3
"""Compile our two-class YOLOX ONNX to a Hailo-10H HEF with the Hailo Dataflow Compiler.

The default is the Fast search detector, detector v4 (YOLOX-s, weights/yolox-s-warehouse-v4.onnx, NMS
config nms_yolox_warehouse.json at its frozen confidence 0.25). The Stage I YOLOX-tiny stays available
for reference: --onnx weights/yolox-tiny-warehouse.onnx --nms-config scripts/hailo/nms_yolox_tiny_stage1.json
--name yolox_tiny_warehouse. Both have the same graph apart from channel width, so they share END_NODES.

Run with Dataflow Compiler 5.3.0 on 2026-11-05 (CPU only). Expected Dataflow Compiler: 5.3.0, the pair of the HailoRT 5.3.x that
ASUS supports for the UGen300 (a HEF must come from the DFC release matching the device's HailoRT;
the Model Zoo's HAILO10H tables name the DFC per release). Optimization above level 0 wants a GPU: without
one the DFC picks level 0 when no level is given, and an explicit level runs on the CPU, slowly.

    python scripts/hailo/compile_yolox_hef.py --check-only
    python scripts/hailo/compile_yolox_hef.py --calibration-dir CALIB/ --eval-dir HELDOUT/ --output build/hailo

CALIB/ holds at least 1,024 JPEG frames sampled only from the training data of the model being
compiled, never from a test set (detector v4: its SDG training runs and the SmartSpaces, LOCO and
keremberke training splits in data/evaluation/stage2/detector-v4/PROTOCOL.md; Stage I: the 51
training runs); HELDOUT/ holds frames the calibration never saw, for the acceptance check. Frames
are letterboxed exactly as src/tracebrief/compact_vision.py does for the ONNX model: BGR, 0-255,
top-left on a 416x416 canvas filled with 114, no mean/std normalization.

Nothing is guessed: the nine END_NODES are checked against the ONNX graph before parsing, and the
parsed HAR's output layers must match the NMS config before optimization; on a mismatch the
script prints the decoder block to use instead. Before compiling, the SNR between the full-precision
and quantized emulators must be at least --min-snr dB on each of the nine decode-head outputs (read
as internal layers, since with nms_postprocess the emulator's only output is the NMS box list).
"""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import random

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT_SIZE = 416
CLASSES = 2
NMS_CONFIG = HERE / "nms_yolox_warehouse.json"
# Outputs of the three decode heads (stride 8, 16, 32): box regression (4 channels), objectness
# sigmoid (1) and class sigmoid (2). The same node names in weights/yolox-s-warehouse-v4.onnx and
# weights/yolox-tiny-warehouse.onnx; check_onnx_end_nodes() checks them against the ONNX being compiled.
END_NODES = ["/head/reg_preds.0/Conv", "/head/Sigmoid", "/head/Sigmoid_1",
             "/head/reg_preds.1/Conv", "/head/Sigmoid_2", "/head/Sigmoid_3",
             "/head/reg_preds.2/Conv", "/head/Sigmoid_4", "/head/Sigmoid_5"]
ROLES = {4: "reg_layer", 1: "objectness_layer", CLASSES: "cls_layer"}
CALIBRATION_SEED = 20261105


def letterbox(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    ratio = min(INPUT_SIZE / image.shape[0], INPUT_SIZE / image.shape[1])
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    h, w = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
    canvas[:h, :w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return canvas


def onnx_shapes(onnx_path: Path) -> dict:
    """{node name: output shape [N, C, H, W]} for every graph node, from ONNX shape inference."""
    import onnx
    from onnx import shape_inference
    model = shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    values = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(model.graph.value_info) + list(model.graph.output)}
    return {node.name: values.get(node.output[0]) for node in model.graph.node}


def check_onnx_end_nodes(shapes: dict, end_nodes=END_NODES, input_size=INPUT_SIZE) -> list[str]:
    """Each end node exists and is one decode-head output: 4, 1 or CLASSES channels at stride 8, 16 or 32."""
    problems, seen = [], set()
    for name in end_nodes:
        shape = shapes.get(name)
        if not shape or len(shape) != 4:
            problems.append(f"ONNX end node {name} is missing or has no 4-D output shape.")
            continue
        stride = input_size // shape[2] if shape[2] else 0
        if shape[1] not in ROLES or stride not in (8, 16, 32) or shape[2] != shape[3]:
            problems.append(f"ONNX end node {name} has shape {shape}, not a decode-head output.")
        seen.add((stride, ROLES.get(shape[1])))
    if len(seen) != 9:
        problems.append(f"The end nodes cover {sorted(seen, key=str)}, not every role at every stride.")
    return problems


def hn_outputs(hn: dict) -> list[dict]:
    """The parsed HAR's outputs: {layer, shape [H, W, C], original_names} of each layer feeding an output.

    Uses the HN dictionary of runner.get_hn(): net_params.output_layers_order lists output layers,
    each output_layer's input is the layer named in the NMS JSON, original_names are the ONNX nodes.
    """
    layers = hn["layers"]
    order = (hn.get("net_params") or {}).get("output_layers_order") or [n for n, l in layers.items() if l.get("type") == "output_layer"]
    outputs = []
    for name in order:
        layer = layers[name]
        source = layer["input"][0] if layer.get("type") == "output_layer" else name
        shape = layers[source]["output_shapes"][0]
        outputs.append({"layer": source.split("/")[-1], "shape": list(shape[-3:]),
                        "original_names": list(layers[source].get("original_names", []))})
    return outputs


def expected_decoders(outputs: list[dict], input_size=INPUT_SIZE) -> list[dict]:
    """bbox_decoders entries derived from the HAR outputs: stride from height, role from channel count."""
    by_stride = {}
    for output in outputs:
        height, _, channels = output["shape"]
        stride = input_size // height
        role = ROLES.get(channels)
        if role is None or role in by_stride.setdefault(stride, {}):
            raise ValueError(f"HAR output {output['layer']} {output['shape']} does not map to one decoder role.")
        by_stride[stride][role] = output["layer"]
    return [{"name": f"bbox_decoder_{stride}", "stride": stride, **{r: by_stride[stride].get(r) for r in ROLES.values()}}
            for stride in sorted(by_stride)]


def check_nms_config(config: dict, outputs: list[dict], end_nodes=END_NODES, input_size=INPUT_SIZE) -> tuple[list[str], dict]:
    """Problems that would make HailoRT's NMS read the wrong tensors, and the config that fixes them."""
    problems = []
    suggested = json.loads(json.dumps(config))
    try:
        decoders = expected_decoders(outputs, input_size)
    except ValueError as exc:
        return [str(exc)], suggested
    suggested.update(classes=CLASSES, image_dims=[input_size, input_size], number_of_detection_heads=len(decoders),
                     bbox_decoders=decoders)
    covered = {name for output in outputs for name in output["original_names"]}
    for node in end_nodes:
        if covered and node not in covered:
            problems.append(f"ONNX end node {node} is not among the HAR outputs' original names.")
    if config.get("classes") != CLASSES:
        problems.append(f"classes is {config.get('classes')}, the model has {CLASSES}.")
    if config.get("image_dims") != [input_size, input_size]:
        problems.append(f"image_dims is {config.get('image_dims')}, the input is {[input_size, input_size]}.")
    actual = {d.get("stride"): d for d in config.get("bbox_decoders", [])}
    for decoder in decoders:
        given = actual.get(decoder["stride"], {})
        for role in ROLES.values():
            if given.get(role) != decoder[role]:
                problems.append(f"stride {decoder['stride']} {role} is {given.get(role)!r}; the HAR output is {decoder[role]!r}.")
    if len(actual) != len(decoders):
        problems.append(f"The config has {len(actual)} decoders; the HAR has {len(decoders)} strides.")
    return problems, suggested


def snr_db(reference, test) -> float:
    """10 * log10(signal / noise), as Hailo reports layer SNR; Hailo treats 10 dB as the minimum."""
    reference = np.asarray(reference, dtype=np.float64)
    noise = float(np.sum((reference - np.asarray(test, dtype=np.float64)) ** 2))
    signal = float(np.sum(reference ** 2))
    return float("inf") if noise == 0 else 10 * np.log10(max(signal, 1e-30) / noise)


def emulate(runner, frames: np.ndarray, layers: list[str], batch_size: int = 8) -> dict:
    """Full-precision and quantized emulator outputs on the same frames.

    Once nms_postprocess is attached, runner.infer returns only the NMS output ([N, classes, 5,
    max_proposals], a sorted and thresholded box list), so the decode-head outputs that feed it are
    read as internal layers of the emulator's model (acceleras HailoModel.set_output_interal_layers).
    Returns {"fp" | "int8": {"nms": array, "heads": [one array per entry of layers]}}.
    """
    import tensorflow as tf
    from hailo_sdk_client import InferenceContext
    results = {}
    for name, context in (("fp", InferenceContext.SDK_FP_OPTIMIZED), ("int8", InferenceContext.SDK_QUANTIZED)):
        with runner.infer_context(context) as ctx:
            model = runner.get_keras_model(ctx)
            inner = getattr(model, "model", model)  # the inference wrapper holds the HailoModel
            inner.set_output_interal_layers(list(layers))
            try:
                if not inner.built:
                    inner.build((1, *frames.shape[1:]))
                predict = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
                nms, heads = [], []
                for start in range(0, len(frames), batch_size):
                    out, internal = predict(tf.convert_to_tensor(frames[start:start + batch_size]))
                    nms.append(np.asarray(out))
                    heads.append([np.asarray(t) for t in internal])
            finally:
                inner.reset_output_interal_layers()
        results[name] = {"nms": np.concatenate(nms), "heads": [np.concatenate([h[i] for h in heads]) for i in range(len(layers))]}
    return results


def snr_report(emulated: dict, outputs: list[dict], input_size=INPUT_SIZE) -> list[dict]:
    """Per decode-head SNR between the full-precision and quantized emulators (outputs from hn_outputs)."""
    return [{"layer": output["layer"], "stride": input_size // output["shape"][0], "role": ROLES.get(output["shape"][2]),
             "shape": list(np.shape(fp)), "snr_db": round(snr_db(fp, q), 2)}
            for output, fp, q in zip(outputs, emulated["fp"]["heads"], emulated["int8"]["heads"], strict=True)]


def check_dfc(expected: str):
    try:
        installed = version("hailo-dataflow-compiler")
    except PackageNotFoundError:
        print(f"warning: cannot read the Dataflow Compiler version; this script expects {expected}.x.")
        return
    if installed.split(".")[:len(expected.split("."))] != expected.split("."):
        raise SystemExit(f"Dataflow Compiler {installed} is installed; the device runs HailoRT {expected}.x, so compile with DFC {expected}.x "
                         "or pass --dfc with the HailoRT release of the device.")


def frames_in(directory: Path, minimum: int, label: str, seed: int | None = None) -> tuple[list[Path], np.ndarray]:
    """The folder's JPEG frames, letterboxed: sorted by name, or in a seeded shuffle when seed is given.

    The calibration frames are shuffled because the Dataflow Compiler's statistics collector reads
    the first calibset_size frames in the order given (64 by default at every optimization level), so a
    sorted folder of <source>_<n>.jpg files would calibrate on the alphabetically first source only.
    """
    paths = sorted(directory.glob("*.jpg"))
    if len(paths) < minimum:
        raise SystemExit(f"Need at least {minimum} {label} frames in {directory}, found {len(paths)}.")
    if seed is not None:
        random.Random(seed).shuffle(paths)
    return paths, np.stack([letterbox(p) for p in paths])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", type=Path, default=ROOT / "weights/yolox-s-warehouse-v4.onnx")
    parser.add_argument("--nms-config", type=Path, default=NMS_CONFIG,
                        help="HailoRT NMS config; nms_scores_th is the profile's frozen confidence")
    parser.add_argument("--name", default="yolox_s_warehouse_v4", help="network name, also used for the HAR and HEF files")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--eval-dir", type=Path, help="held-out frames for the FP vs INT8 acceptance check")
    parser.add_argument("--output", type=Path, default=ROOT / "build/hailo")
    parser.add_argument("--min-calibration", type=int, default=1024)
    parser.add_argument("--calibration-seed", type=int, default=CALIBRATION_SEED, help="seed of the calibration frame shuffle")
    parser.add_argument("--calibset-size", type=int,
                        help="frames the statistics collector uses (model_optimization_config calibration); omit for the DFC default of 64")
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--optimization-level", type=int, default=2, choices=range(0, 5),
                        help="Dataflow Compiler optimization level; levels above 0 want 1,024+ calibration frames and a GPU "
                             "(an explicit level also runs on a CPU-only host, much more slowly; only an unset level falls back to 0)")
    parser.add_argument("--compression-level", type=int, choices=range(0, 6), help="omit to keep the compiler default")
    parser.add_argument("--min-snr", type=float, default=10.0)
    parser.add_argument("--dfc", default="5.3", help="Dataflow Compiler release expected (match the device's HailoRT)")
    parser.add_argument("--check-only", action="store_true", help="parse and verify the end nodes, then stop")
    parser.add_argument("--no-compile", action="store_true", help="stop after the optimized HAR and the acceptance check")
    parser.add_argument("--skip-noise-analysis", action="store_true", help="skip runner.analyze_noise (the profiler's per-layer SNR)")
    parser.add_argument("--analyze-mode", choices=("simple", "advanced"), default="simple",
                        help="runner.analyze_noise mode: simple gives every layer's SNR in the full quantized net; advanced adds "
                             "the layer-by-layer pass (about 3 h for YOLOX-s on six CPU cores)")
    args = parser.parse_args()

    try:
        problems = check_onnx_end_nodes(onnx_shapes(args.onnx))
    except ImportError:
        problems = []
        print("warning: the onnx package is missing; ONNX end nodes are checked against the HAR only.")
    if problems:
        raise SystemExit("ONNX end nodes do not match the decode head:\n  " + "\n  ".join(problems))
    if not args.check_only:
        if not args.calibration_dir or not args.eval_dir:
            raise SystemExit("--calibration-dir and --eval-dir are required unless --check-only.")
        calibration_paths, calibration = frames_in(args.calibration_dir, args.min_calibration, "calibration", args.calibration_seed)
        held_out = frames_in(args.eval_dir, args.eval_count, "held-out")[1][:args.eval_count].astype(np.float32)

    print("ONNX end nodes match the decode head.")
    check_dfc(args.dfc)
    try:
        from hailo_sdk_client import ClientRunner  # Hailo Dataflow Compiler
    except ImportError as exc:
        raise SystemExit(f"The Hailo Dataflow Compiler (hailo_sdk_client) is not installed: {exc}")

    args.output.mkdir(parents=True, exist_ok=True)
    runner = ClientRunner(hw_arch="hailo10h")
    runner.translate_onnx_model(str(args.onnx), args.name, start_node_names=["images"],
                                end_node_names=END_NODES, net_input_shapes={"images": [1, 3, INPUT_SIZE, INPUT_SIZE]})
    runner.save_har(str(args.output / f"{args.name}_parsed.har"))
    outputs = hn_outputs(runner.get_hn())
    print("HAR outputs:")
    for output in outputs:
        print(f"  {output['layer']:<12} {str(output['shape']):<14} {', '.join(output['original_names'])}")
    config = json.loads(args.nms_config.read_text(encoding="utf-8"))
    problems, suggested = check_nms_config(config, outputs)
    if problems:
        raise SystemExit(f"{args.nms_config.name} does not match the parsed HAR:\n  " + "\n  ".join(problems)
                         + f"\nReplace bbox_decoders in {args.nms_config} with:\n" + json.dumps(suggested["bbox_decoders"], indent=4))
    print(f"{args.nms_config.name} matches the HAR outputs.")
    if args.check_only:
        return

    # NMS runs in HailoRT on the host and returns per-class boxes normalized to 0-1.
    flavor = f"optimization_level={args.optimization_level}"
    if args.compression_level is not None:
        flavor += f", compression_level={args.compression_level}"
    script = (f'nms_postprocess("{args.nms_config.resolve()}", meta_arch=yolox, engine=cpu)\n'
              f"model_optimization_flavor({flavor})\n")
    if args.calibset_size:
        script += f"model_optimization_config(calibration, calibset_size={args.calibset_size})\n"
    runner.load_model_script(script)
    order = "\n".join(p.name for p in calibration_paths) + "\n"
    (args.output / "calibration_order.txt").write_text(order, encoding="utf-8")
    runner.optimize(calibration)
    runner.save_har(str(args.output / f"{args.name}_optimized.har"))

    # Acceptance before compiling: Hailo's per-layer noise report, then our FP vs INT8 emulator check
    # on each of the nine decode-head outputs (not on the NMS box list, whose SNR means little).
    if not args.skip_noise_analysis:
        runner.analyze_noise(held_out, batch_size=2, data_count=len(held_out), analyze_mode=args.analyze_mode)
        runner.save_har(str(args.output / f"{args.name}_optimized.har"))  # view with `hailo profiler`
    layers = [f"{runner.get_hn()['name']}/{output['layer']}" for output in outputs]
    emulated = emulate(runner, held_out, layers)
    for name in ("fp", "int8"):
        np.savez(args.output / f"emulator_{name}.npz", nms=emulated[name]["nms"],
                 **{output["layer"]: head for output, head in zip(outputs, emulated[name]["heads"])})
    report = snr_report(emulated, outputs)
    (args.output / "acceptance.json").write_text(json.dumps({"min_snr_db": args.min_snr, "held_out_frames": len(held_out),
                                                             "optimization_level": args.optimization_level,
                                                             "compression_level": args.compression_level,
                                                             "calibration_frames": len(calibration),
                                                             "calibration_seed": args.calibration_seed,
                                                             "calibset_size": args.calibset_size or "DFC default (64)",
                                                             "calibration_order_sha256": hashlib.sha256(order.encode()).hexdigest(),
                                                             "outputs": report}, indent=2), encoding="utf-8")
    for row in report:
        print(f"  {row['layer']:<12} stride {row['stride']:<2} {row['role']:<16} {row['shape']}: SNR {row['snr_db']} dB")
    low = [row for row in report if row["snr_db"] < args.min_snr]
    if low:
        raise SystemExit(f"{len(low)} output(s) below {args.min_snr} dB; not compiling. See {args.output / 'acceptance.json'}.")
    if args.no_compile:
        print(f"Acceptance passed; --no-compile given, so {args.output / (args.name + '_optimized.har')} is the result.")
        return

    hef = args.output / f"{args.name}.hef"
    hef.write_bytes(runner.compile())
    print(f"{hef} sha256={hashlib.sha256(hef.read_bytes()).hexdigest()}")
    print("Record this sha256 for the detector in config/hailo-manifest.json, then run the pair-precision/recall comparison.")


if __name__ == "__main__":
    main()
