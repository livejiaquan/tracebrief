"""scripts/hailo/compile_yolox_hef.py checks, without the Dataflow Compiler: synthetic HN dictionaries only."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NMS = json.loads((ROOT / "scripts" / "hailo" / "nms_yolox_warehouse.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compile_script():
    pytest.importorskip("cv2"); pytest.importorskip("numpy")
    spec = importlib.util.spec_from_file_location("compile_yolox_hef", ROOT / "scripts" / "hailo" / "compile_yolox_hef.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hn(names, end_nodes):
    """An HN dictionary shaped like runner.get_hn(): output layers fed by one conv per decoder role."""
    layers, order = {}, []
    for (stride, size), triple, nodes in zip(((8, 52), (16, 26), (32, 13)), names, end_nodes):
        for name, channels, node in zip(triple, (4, 1, 2), nodes):
            layers[f"net/{name}"] = {"type": "conv", "output_shapes": [[-1, size, size, channels]], "original_names": [node]}
            layers[f"net/output_{name}"] = {"type": "output_layer", "input": [f"net/{name}"], "output_shapes": [[-1, size, size, channels]]}
            order.append(f"net/output_{name}")
    return {"name": "net", "net_params": {"output_layers_order": order}, "layers": layers}


def json_names():
    return [(d["reg_layer"], d["objectness_layer"], d["cls_layer"]) for d in NMS["bbox_decoders"]]


def grouped(end_nodes):
    return [end_nodes[i:i + 3] for i in (0, 3, 6)]


@pytest.mark.parametrize("weights", ["yolox-s-warehouse-v4.onnx", "yolox-tiny-warehouse.onnx"])
def test_onnx_end_nodes_are_the_decode_head_outputs(compile_script, weights):
    pytest.importorskip("onnx")
    shapes = compile_script.onnx_shapes(ROOT / "weights" / weights)
    assert compile_script.check_onnx_end_nodes(shapes) == []
    assert compile_script.check_onnx_end_nodes({**shapes, "/head/Sigmoid_5": [1, 96, 13, 13]})


def test_nms_configs_use_each_profiles_frozen_thresholds():
    """The default NMS config is for Fast search (detector v4); the Stage I one is kept for reference."""
    profiles = {p["id"]: p for p in json.loads((ROOT / "config" / "search_profiles.json").read_text(encoding="utf-8"))["profiles"]}
    for profile_id, name in (("fast", "nms_yolox_warehouse.json"), ("stage1", "nms_yolox_tiny_stage1.json")):
        settings = json.loads((ROOT / "config" / profiles[profile_id]["settings_file"]).read_text(encoding="utf-8"))
        config = json.loads((ROOT / "scripts" / "hailo" / name).read_text(encoding="utf-8"))
        assert (config["nms_scores_th"], config["nms_iou_th"]) == (settings["confidence"], settings["nms_iou"])
        assert {k: v for k, v in config.items() if k != "nms_scores_th"} == {k: v for k, v in NMS.items() if k != "nms_scores_th"}


def test_matching_har_passes(compile_script):
    outputs = compile_script.hn_outputs(hn(json_names(), grouped(compile_script.END_NODES)))
    problems, suggested = compile_script.check_nms_config(NMS, outputs)
    assert problems == [] and suggested["bbox_decoders"] == NMS["bbox_decoders"]


def test_renumbered_har_layers_are_reported_with_the_fix(compile_script):
    renumbered = [("conv61", "conv62", "conv60"), ("conv75", "conv76", "conv74"), ("conv88", "conv89", "conv87")]
    outputs = compile_script.hn_outputs(hn(renumbered, grouped(compile_script.END_NODES)))
    problems, suggested = compile_script.check_nms_config(NMS, outputs)
    assert any(f"stride 8 reg_layer is '{NMS['bbox_decoders'][0]['reg_layer']}'; the HAR output is 'conv61'" in p for p in problems)
    assert [(d["reg_layer"], d["objectness_layer"], d["cls_layer"]) for d in suggested["bbox_decoders"]] == renumbered
    assert [d["stride"] for d in suggested["bbox_decoders"]] == [8, 16, 32]


def test_end_node_missing_from_the_har_is_reported(compile_script):
    nodes = grouped(compile_script.END_NODES)
    nodes[2] = ["/head/reg_preds.2/Conv", "/head/Sigmoid_4", "/head/cls_preds.2/Conv"]
    problems, _ = compile_script.check_nms_config(NMS, compile_script.hn_outputs(hn(json_names(), nodes)))
    assert problems == ["ONNX end node /head/Sigmoid_5 is not among the HAR outputs' original names."]


def test_snr_matches_hailos_definition(compile_script):
    np = pytest.importorskip("numpy")
    signal = np.ones(1000)
    assert compile_script.snr_db(signal, signal) == float("inf")
    assert compile_script.snr_db(signal, signal + np.sqrt(0.1)) == pytest.approx(10.0)
    assert compile_script.snr_db(signal, signal * 1.01) == pytest.approx(40.0)


# The HAR outputs of `compile_yolox_hef.py --check-only` with Dataflow Compiler 5.3.0 (2026-11-05), the same for
# weights/yolox-s-warehouse-v4.onnx and weights/yolox-tiny-warehouse.onnx: the regression head parses to the
# higher-numbered conv of each stride, so reg/objectness names are not the Model Zoo yolox order.
DFC_530_OUTPUTS = [
    ("conv56", [52, 52, 4], ["/head/reg_preds.0/Conv"]), ("conv55", [52, 52, 1], ["/head/obj_preds.0/Conv", "/head/Sigmoid"]),
    ("conv54", [52, 52, 2], ["/head/cls_preds.0/Conv", "/head/Sigmoid_1"]),
    ("conv70", [26, 26, 4], ["/head/reg_preds.1/Conv"]), ("conv69", [26, 26, 1], ["/head/obj_preds.1/Conv", "/head/Sigmoid_2"]),
    ("conv68", [26, 26, 2], ["/head/cls_preds.1/Conv", "/head/Sigmoid_3"]),
    ("conv83", [13, 13, 4], ["/head/reg_preds.2/Conv"]), ("conv82", [13, 13, 1], ["/head/obj_preds.2/Conv", "/head/Sigmoid_4"]),
    ("conv81", [13, 13, 2], ["/head/cls_preds.2/Conv", "/head/Sigmoid_5"]),
]


def dfc_outputs():
    return [{"layer": layer, "shape": shape, "original_names": names} for layer, shape, names in DFC_530_OUTPUTS]


@pytest.mark.parametrize("name", ["nms_yolox_warehouse.json", "nms_yolox_tiny_stage1.json"])
def test_nms_configs_match_the_dfc_530_parse(compile_script, name):
    config = json.loads((ROOT / "scripts" / "hailo" / name).read_text(encoding="utf-8"))
    problems, suggested = compile_script.check_nms_config(config, dfc_outputs())
    assert problems == [] and suggested["bbox_decoders"] == config["bbox_decoders"]


class FakeContext:
    def __init__(self, kind):
        self.kind = kind

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeHailoModel:
    """The acceleras model behind get_keras_model(): outputs the NMS tensor plus the requested internal layers."""

    def __init__(self, kind, outputs, np):
        self.kind, self.outputs, self.np, self.layers, self.built = kind, outputs, np, None, False

    def set_output_interal_layers(self, layers):
        self.layers = layers

    def reset_output_interal_layers(self):
        self.layers = None

    def build(self, shape):
        self.built = True

    def __call__(self, x, training=False):
        assert self.layers and self.built
        n, np = len(x), self.np
        shapes = {o["layer"]: o["shape"] for o in self.outputs}
        heads = [np.full([n, *shapes[name.split("/")[-1]]], 1.0 if self.kind == "fp" else 1.01, dtype=np.float32) for name in self.layers]
        return np.zeros((n, 2, 5, 100), np.float32), heads


class FakeKerasModel:
    def __init__(self, inner):
        self.model = inner

    def __call__(self, x, training=False):
        return self.model(x, training=training)


class FakeRunner:
    def __init__(self, outputs, np):
        self.outputs, self.np, self.models = outputs, np, []

    def infer_context(self, kind):
        return FakeContext(kind)

    def get_keras_model(self, ctx):
        self.models.append(FakeKerasModel(FakeHailoModel(ctx.kind, self.outputs, self.np)))
        return self.models[-1]

    def infer(self, *args, **kwargs):
        raise AssertionError("runner.infer returns only the NMS output; the acceptance check must read the heads")


def test_acceptance_snr_is_computed_per_decode_head_not_on_the_nms_output(compile_script, monkeypatch):
    """With nms_postprocess attached the emulator's only output is the NMS tensor (DFC 5.3.0): SNR must use the nine heads."""
    import sys
    import types
    np = pytest.importorskip("numpy")
    fake_tf = types.SimpleNamespace(function=lambda f, **_: f, convert_to_tensor=lambda x: x)
    fake_sdk = types.SimpleNamespace(InferenceContext=types.SimpleNamespace(SDK_FP_OPTIMIZED="fp", SDK_QUANTIZED="int8"))
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(sys.modules, "hailo_sdk_client", fake_sdk)
    outputs = dfc_outputs()
    runner = FakeRunner(outputs, np)
    frames = np.zeros((10, 416, 416, 3), np.float32)
    emulated = compile_script.emulate(runner, frames, [f"net/{o['layer']}" for o in outputs], batch_size=4)
    assert set(emulated) == {"fp", "int8"} and emulated["fp"]["nms"].shape == (10, 2, 5, 100)
    assert [h.shape for h in emulated["int8"]["heads"]] == [(10, *o["shape"]) for o in outputs]
    assert all(m.model.layers is None for m in runner.models)  # internal outputs reset after each context
    report = compile_script.snr_report(emulated, outputs)
    assert [(r["layer"], r["stride"], r["role"]) for r in report][:3] == [("conv56", 8, "reg_layer"), ("conv55", 8, "objectness_layer"),
                                                                          ("conv54", 8, "cls_layer")]
    assert len(report) == 9 and all(r["snr_db"] == pytest.approx(40.0) for r in report)


def test_calibration_frames_are_shuffled_so_the_first_calibset_mixes_every_source(compile_script, tmp_path):
    """The DFC statistics collector takes the first calibset_size frames as given (64 by default): a sorted
    folder of <source>_<n>.jpg frames would calibrate on keremberke only (DFC 5.3.0, 2026-11-05)."""
    cv2 = pytest.importorskip("cv2"); np = pytest.importorskip("numpy")
    for source, value in (("kb", 20), ("loco", 80), ("sdg", 160), ("ss", 240)):
        for n in range(40):
            cv2.imwrite(str(tmp_path / f"{source}_{n:04d}.jpg"), np.full((36, 64, 3), value, np.uint8))
    paths, frames = compile_script.frames_in(tmp_path, 160, "calibration", compile_script.CALIBRATION_SEED)
    assert len(paths) == len(frames) == 160
    first = {p.name.split("_")[0] for p in paths[:64]}
    assert first == {"kb", "loco", "sdg", "ss"}
    assert [int(round(f[0, 0, 0] / 10)) * 10 for f in frames[:8]] == [
        {"kb": 20, "loco": 80, "sdg": 160, "ss": 240}[p.name.split("_")[0]] for p in paths[:8]]  # frames follow the order
    again, _ = compile_script.frames_in(tmp_path, 160, "calibration", compile_script.CALIBRATION_SEED)
    assert again == paths  # reproducible
    sorted_paths, _ = compile_script.frames_in(tmp_path, 160, "held-out")
    assert sorted_paths == sorted(paths)  # held-out frames keep their names' order
