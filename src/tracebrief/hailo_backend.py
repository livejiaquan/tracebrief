"""UGen300 (Hailo-10H) backends for the detector and the visual check. Stage II, not yet run on a device.

Written against the HailoRT 5.4 Python API (`hailo_platform`) following Hailo's hailo-apps examples.
Both backends open a VDevice with the same group_id, so the detector (a vision app) and Qwen2-VL-2B
(a GenAI app) share one accelerator; Hailo documents that pairing as supported. The functions that
do not touch the device are pure and covered by tests; the device paths are verified in Stage II.
"""
from __future__ import annotations

from pathlib import Path

GROUP_ID = "SHARED"
CLASSES = ("person", "forklift")


def letterbox_ratio(width: int, height: int, size: int = 416) -> float:
    return min(size / height, size / width)


def to_source_boxes(per_class, width: int, height: int, size: int = 416, confidence: float = 0.40) -> list[dict]:
    """Convert HailoRT NMS_BY_CLASS output ([y0, x0, y1, x1, score] normalized to the 416 input) to source pixels.

    The input was letterboxed top-left, so undoing it is a single scale by size / ratio.
    """
    scale = size / letterbox_ratio(width, height, size)
    detections = []
    for name, boxes in zip(CLASSES, per_class):
        for y0, x0, y1, x1, score in boxes:
            if score < confidence:
                continue
            box = [round(min(max(v * scale, 0), limit), 2) for v, limit in ((x0, width), (y0, height), (x1, width), (y1, height))]
            if box[2] > box[0] and box[3] > box[1]:
                detections.append({"class_name": name, "confidence": round(float(score), 4), "bbox_xyxy": box})
    return sorted(detections, key=lambda d: -d["confidence"])


def pad_to_square(image, fill=(114, 114, 114)):
    """Pad a PIL image to a square so the VLM's fixed 336x336 input does not crop off the person or forklift."""
    from PIL import Image
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def shared_vdevice():
    from hailo_platform import HailoSchedulingAlgorithm, VDevice
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    params.group_id = GROUP_ID
    return VDevice(params)


class HailoDetector:
    """YOLOX-tiny HEF compiled by scripts/hailo/compile_yolox_hef.py; NMS runs in HailoRT on the host."""

    def __init__(self, hef: Path, confidence: float = 0.40, nms_iou: float = 0.45):
        self.vdevice = shared_vdevice()
        self.model = self.vdevice.create_infer_model(str(hef))
        self.model.set_batch_size(1)
        self.model.output().set_nms_score_threshold(confidence)
        self.model.output().set_nms_iou_threshold(nms_iou)
        self.confidence = confidence
        self._context = self.model.configure()
        self.configured = self._context.__enter__()

    def detect(self, letterboxed_bgr, width: int, height: int) -> list[dict]:
        """letterboxed_bgr: uint8 416x416x3 prepared exactly as compact_vision.preprocess (without the transpose)."""
        import numpy as np
        bindings = self.configured.create_bindings()
        bindings.input().set_buffer(np.ascontiguousarray(letterboxed_bgr))
        bindings.output().set_buffer(np.empty(self.model.output().shape, dtype=np.float32))
        self.configured.run([bindings], 10000)
        return to_source_boxes(bindings.output().get_buffer(), width, height, confidence=self.confidence)

    def close(self):
        self._context.__exit__(None, None, None)


class HailoVisualCheck:
    """Qwen2-VL-2B-Instruct HEF from Hailo's GenAI Model Zoo, with the same prompt and parser as the CPU path."""

    def __init__(self, hef: Path, vdevice=None):
        from hailo_platform.genai import VLM
        self.vdevice = vdevice or shared_vdevice()
        self.vlm = VLM(self.vdevice, str(hef))

    def check(self, crop) -> dict:
        import numpy as np
        from PIL import Image
        import importlib.util
        spec = importlib.util.spec_from_file_location("look_local", Path(__file__).resolve().parents[2] / "scripts/look_local.py")
        look = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(look)
        height, width = self.vlm.input_frame_shape()[:2]
        frame = np.asarray(pad_to_square(crop.convert("RGB")).resize((width, height), Image.LANCZOS), dtype=np.uint8)
        prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": look.PROMPT}]}]
        text = ""
        with self.vlm.generate(prompt=prompt, frames=[frame], temperature=0.0, seed=19, max_generated_tokens=160) as stream:
            for token in stream:
                if token != "<|im_end|>":
                    text += token
        self.vlm.clear_context()  # each candidate starts from an empty context
        return {"raw_text": text, "answer": look.parse(text)}
