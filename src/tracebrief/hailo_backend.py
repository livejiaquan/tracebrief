"""UGen300 (Hailo-10H) runtime: one VDevice per process, shared by the detector and Qwen2-VL-2B.

Written against the HailoRT Python API (`hailo_platform`) of releases 5.3, 5.4 and 5.5: every call
used here has the same name and arguments in all three (pyhailort.py at hailort tags v5.3.0-v5.5.0).
ASUS's UGen300 samples target HailoRT 5.3.x on Python 3.10, so this module avoids 3.11-only calls.
Run on a UGen300 (USB, HailoRT 5.3.2, Windows host) on 2026-11-05; tests/test_hailo_backend.py drives it with a
fake hailo_platform and checks a buffer recorded on that device.

Device sharing. genai.VLM takes an existing VDevice ("The VDevice must remain active for the lifetime
of the VLM instance", VLM docstring in pyhailort.py), and Hailo's v2a_demo
(hailo-apps gen_ai_apps/v2a_demo/pipeline.py) runs an InferModel and GenAI models on one VDevice in one
process. A shared group_id is only what separate processes need (hailo-apps
doc/user_guide/running_parallel.md), and multi_process_service is for Hailo-8/8L. So this process opens
exactly one VDevice with the ROUND_ROBIN scheduler and hands it to both models. group_id stays "SHARED",
as in Hailo's apps, so a Hailo vision app may share the device; a second GenAI model may not (one
LLM/VLM at a time, hailo_model_zoo_genai docs/USAGE.rst). On the UGen300 over USB with a Windows host
(HailoRT 5.3.2) a second process cannot open the device at all (LIBUSB_ERROR_ACCESS), so there both
models must live in this one process; the VLM's token read holds the GIL, so detection and generation
do not overlap in practice. The two models therefore take turns (Turns): a job runs in phases, all
detection, then every visual check, then every draft (server.Phases), and a second process that finds
the device held fails at once with DeviceError instead of waiting.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from . import drafting, look_prompt

GROUP_ID = "SHARED"
CLASSES = ("person", "forklift")
INPUT_SIZE = 416
BOX = 5  # hailo_bbox_float32_t: y_min, x_min, y_max, x_max, score
# hailo_detection_t inside #pragma pack(1) in hailort.h: the HAILO_NMS_BY_SCORE record after a uint16 count.
BY_SCORE_RECORD = [("y_min", "<f4"), ("x_min", "<f4"), ("y_max", "<f4"), ("x_max", "<f4"),
                   ("score", "<f4"), ("class_id", "<u2")]
# Hailo GenAI Model Zoo v5.5.0 changelog: when a conversation leads with an image, use a system prompt
# saying the model can see images, or Qwen2-VL-2B may answer that it cannot.
VLM_SYSTEM = "You are a helpful assistant that can see images."
END_TOKENS = ("<|im_end|>", "<|endoftext|>")
SEED = 19
SETTLE_S = 1.0  # HailoDetector.close(): minimum gap between the last completed job and shutdown()
LOOK_TOKENS = 160
DRAFT_TOKENS = 320
LLM_FREQUENCY_PENALTY = 1.0  # none, as the CPU worker (HailoLLM); the Qwen2.5 HEF's own default is 1.1
# hailo_status HAILO_LIBUSB_FAILURE (hailort.h). On the UGen300 over USB with a Windows host (HailoRT 5.3.2) a
# process that opens the device while another holds it gets it within 0.1 s: identify() raises
# HailoRTStatusException("98") and VDevice() "libhailort failed with error: 98 (HAILO_LIBUSB_FAILURE)", after
# libusb's LIBUSB_ERROR_ACCESS in the HailoRT log.
LIBUSB_FAILURE = 98
HELD = "already open in another process"


class DeviceError(RuntimeError):
    """The UGen300 run cannot proceed; the message says why and what to change. Never a CPU fallback."""


def held_elsewhere(exc: Exception) -> bool:
    """HailoRT's answer when another process holds the USB device (see LIBUSB_FAILURE)."""
    text = str(exc).strip()
    return "HAILO_LIBUSB_FAILURE" in text or text == str(LIBUSB_FAILURE) or f"error: {LIBUSB_FAILURE} " in text


def held_error(device_id=None) -> DeviceError:
    """The message for a device another process holds: which one, why, and what to stop."""
    device = f"The UGen300 ({device_id})" if device_id else "The UGen300"
    where = "On a Windows host over USB" if sys.platform == "win32" else "Over USB"
    return DeviceError(f"{device} is {HELD}: HailoRT returned HAILO_LIBUSB_FAILURE ({LIBUSB_FAILURE}). {where} only one "
                       "process can open the device (HailoRT 5.3.2 on Windows: LIBUSB_ERROR_ACCESS), and TraceBrief runs both "
                       "models in one process. Stop the other process (another TraceBrief server, "
                       "scripts/hailo/bench_ugen300.py, hailortcli) and try again; if none is running, reconnect the UGen300.")


class Turns:
    """One model at a time on the device: no detector batch is in flight while the VLM generates, and back.

    On the UGen300 over USB (HailoRT 5.3.2) the genai binding holds the GIL while it reads tokens, so a
    detector thread in the same process ran at 0.37-0.38 FPS during generation instead of 72-74 FPS.
    Jobs are scheduled in phases so the models rarely switch; this lock makes any other caller wait
    for its turn, and counts the switches between the two models.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.last, self.switches = None, 0

    @contextmanager
    def hold(self, model: str):
        with self.lock:
            if self.last not in (None, model):
                self.switches += 1
            self.last = model
            yield


def sha256(path: Path) -> str:
    """Chunked SHA-256; hashlib.file_digest needs Python 3.11 and the device host may run 3.10."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_matches(installed: str, expected: str) -> bool:
    """'5.3' accepts any 5.3.x; '5.3.0' only that release."""
    want = str(expected).split(".")
    return str(installed).split(".")[:len(want)] == want


def verify_manifest(path: Path, root: Path, hailort_version: str | None = None) -> dict:
    """Hash every HEF in the manifest before anything is loaded.

    Returns {name: {path, file, sha256, source, hash_s}} for "detector" and "vlm". A missing file,
    an unrecorded or mismatched SHA-256, or a HailoRT release other than the manifest's raises
    DeviceError, so a wrong or corrupted HEF is never handed to HailoRT.
    """
    path = Path(path)
    if not path.is_file():
        raise DeviceError(f"HEF manifest not found: {path}. Set TRACEBRIEF_HAILO_MANIFEST or create config/hailo-manifest.json.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.get("hailort")
    if hailort_version and expected and not release_matches(hailort_version, expected):
        raise DeviceError(f"The HEFs in {path.name} are for HailoRT {expected}, but HailoRT {hailort_version} is installed. "
                          "Install the matching HailoRT, or download and compile HEFs for this release and update the manifest.")
    models = {}
    for name in ("detector", "vlm"):
        entry = (manifest.get("models") or {}).get(name)
        if not entry or not entry.get("file"):
            raise DeviceError(f"{path.name} has no '{name}' HEF entry.")
        file = Path(entry["file"])
        file = file if file.is_absolute() else Path(root) / file
        if not file.is_file():
            raise DeviceError(f"The {name} HEF is missing: {file} (source: {entry.get('source', 'not recorded')}).")
        tick = time.perf_counter()
        digest = sha256(file)
        hash_s = round(time.perf_counter() - tick, 3)
        if not entry.get("sha256"):
            raise DeviceError(f"{path.name} records no SHA-256 for the {name} HEF. {file.name} hashes to {digest}; "
                              "record that value in the manifest only if this is the intended build.")
        if digest != str(entry["sha256"]).lower():
            raise DeviceError(f"{name} HEF SHA-256 mismatch: {file} hashes to {digest}, but {path.name} records "
                              f"{entry['sha256']}. Refusing to load it.")
        models[name] = {"path": file, "file": file.name, "sha256": digest, "source": entry.get("source"), "hash_s": hash_s}
    return models


def letterbox_ratio(width: int, height: int, size: int = INPUT_SIZE) -> float:
    return min(size / height, size / width)


def letterbox(image, size: int = INPUT_SIZE):
    """uint8 NHWC canvas for the HEF: compact_vision.preprocess's top-left BGR letterbox, untransposed.

    The calibration frames in scripts/hailo/compile_yolox_hef.py are prepared the same way.
    """
    import cv2
    import numpy as np
    ratio = min(size / image.shape[0], size / image.shape[1])
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    h, w = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
    canvas[:h, :w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    return canvas, ratio


def parse_nms(buffer, classes: int = len(CLASSES)) -> list:
    """Per-class lists of [y_min, x_min, y_max, x_max, score], normalized to the HEF input.

    Accepts every NMS layout HailoRT 5.3-5.5 documents (hailort.h, ConfiguredInferModel.Bindings
    .InferStream.get_buffer in pyhailort.py):
    - HAILO_NMS_BY_CLASS from get_buffer(): one (n, 5) array per class;
    - the same wrapped in a batch of one ([frame][class], as the vstream API returns it);
    - HAILO_NMS_BY_CLASS raw float32 buffer: per class, a box count followed by that many boxes;
    - HAILO_NMS_BY_SCORE from get_buffer(): Detection objects carrying class_id;
    - HAILO_NMS_BY_SCORE raw uint8 buffer: a uint16 count, then packed 22-byte hailo_detection_t records.
    Plain floats are returned, so the device buffer can be reused at once; numpy is needed only for raw buffers.
    """
    if hasattr(buffer, "dtype") and getattr(buffer, "ndim", 0) == 1:
        if buffer.dtype.kind == "u" and buffer.dtype.itemsize == 1:
            return _by_score_bytes(buffer, classes)
        return _by_class_raw(buffer, classes)
    items = list(buffer)
    if items and all(hasattr(item, "class_id") for item in items):
        return _by_score([(d.y_min, d.x_min, d.y_max, d.x_max, d.score, d.class_id) for d in items], classes)
    if not items:
        return [[] for _ in range(classes)]  # HAILO_NMS_BY_SCORE with no detections
    if len(items) == 1 and classes != 1 and isinstance(items[0], (list, tuple)) and len(items[0]) == classes:
        items = list(items[0])
    if len(items) != classes:
        raise ValueError(f"Expected {classes} per-class NMS arrays, got {len(items)}.")
    per_class = []
    for boxes in items:
        rows = [[float(v) for v in box] for box in boxes]
        if any(len(row) != BOX for row in rows):
            raise ValueError(f"Unexpected NMS box length; expected {BOX} values (y_min, x_min, y_max, x_max, score).")
        per_class.append(rows)
    return per_class


def _by_class_raw(flat, classes):
    flat = flat.reshape(-1).tolist()
    per_class, offset = [], 0
    for index in range(classes):
        if offset >= len(flat):
            raise ValueError(f"NMS buffer ended before class {index}.")
        count = flat[offset]
        if count < 0 or count != int(count):
            raise ValueError(f"Invalid NMS box count {count} for class {index}.")
        start, end = offset + 1, offset + 1 + int(count) * BOX
        if end > len(flat):
            raise ValueError("NMS buffer is shorter than its box counts.")
        per_class.append([flat[i:i + BOX] for i in range(start, end, BOX)])
        offset = end
    return per_class


def _by_score(rows, classes):
    per_class = [[] for _ in range(classes)]
    for *box, class_id in rows:
        if not 0 <= int(class_id) < classes:
            raise ValueError(f"NMS class_id {class_id} is outside the model's {classes} classes.")
        per_class[int(class_id)].append([float(v) for v in box])
    return per_class


def _by_score_bytes(raw, classes):
    import numpy as np
    raw = np.ascontiguousarray(raw).reshape(-1)
    if raw.size < 2:
        raise ValueError("NMS by-score buffer is shorter than its count.")
    count = int.from_bytes(raw[:2].tobytes(), "little")
    record = np.dtype(BY_SCORE_RECORD)
    end = 2 + count * record.itemsize
    if end > raw.size:
        raise ValueError("NMS by-score buffer is shorter than its detection count.")
    records = np.frombuffer(raw[2:end].tobytes(), dtype=record)
    return _by_score([tuple(r[name] for name, _ in BY_SCORE_RECORD) for r in records], classes)


def to_source_boxes(per_class, width: int, height: int, size: int = INPUT_SIZE, confidence: float = 0.40) -> list[dict]:
    """Convert HailoRT NMS output (any parse_nms layout; [y0, x0, y1, x1, score] normalized to the 416 input) to source pixels.

    The input was letterboxed top-left, so undoing it is a single scale by size / ratio.
    """
    scale = size / letterbox_ratio(width, height, size)
    detections = []
    for name, boxes in zip(CLASSES, parse_nms(per_class, len(CLASSES))):
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


def vlm_frame(crop, shape=(336, 336, 3)):
    """RGB uint8 frame in the VLM's input shape. Hailo's chat example centre-crops; we pad instead."""
    import numpy as np
    from PIL import Image
    height, width = int(shape[0]), int(shape[1])
    frame = pad_to_square(crop.convert("RGB")).resize((width, height), Image.LANCZOS)
    return np.array(frame, dtype=np.uint8, order="C")  # a writeable copy: VLM.generate rejects read-only arrays


def hailort():
    """The hailo_platform module, or a DeviceError naming the interpreter that lacks it."""
    try:
        import hailo_platform
    except ImportError as exc:
        raise DeviceError(f"The HailoRT Python API (hailo_platform) is not installed in {sys.executable}; "
                          f"install the HailoRT wheel that matches the driver ({exc}).") from exc
    return hailo_platform


def _text(value) -> str | None:
    value = str(value).rstrip("\x00").strip() if value is not None else ""
    return value or None


def identify(hailo) -> dict:
    """The Python-API equivalent of `hailortcli scan` and `hailortcli fw-control identify`.

    No device is a DeviceError. Identity fields are reported when the firmware returns them; a failed
    identify is recorded, not fatal, because the device is present and HailoRT will report real faults.
    """
    ids = list(hailo.Device.scan())
    if not ids:
        raise DeviceError("No Hailo device was found (Device.scan() returned none). "
                          "Connect the UGen300 directly to a USB-C 10 Gbps port and check `hailortcli scan`.")
    devices = []
    for device_id in ids:
        record = {"id": str(device_id)}
        try:
            with hailo.Device(device_id) as device:
                info = device.control.identify()
            architecture = info.device_architecture
            record.update(architecture=getattr(architecture, "name", None) or str(architecture),
                          firmware=str(info.firmware_version), serial_number=_text(info.serial_number),
                          part_number=_text(info.part_number), product_name=_text(info.product_name))
        except Exception as exc:
            if str(device_id).startswith("usb") and held_elsewhere(exc):
                raise held_error(device_id) from exc  # before any HEF is hashed (the VLM's takes about 7 s)
            record["identify_error"] = f"{type(exc).__name__}: {exc}"
        devices.append(record)
    return {"hailort_version": getattr(hailo, "__version__", None), "devices": devices}


def shared_vdevice(hailo=None):
    """The process's one VDevice: ROUND_ROBIN scheduler, group "SHARED". A device held by another process is a DeviceError."""
    hailo = hailo or hailort()
    params = hailo.VDevice.create_params()
    params.scheduling_algorithm = hailo.HailoSchedulingAlgorithm.ROUND_ROBIN
    params.group_id = GROUP_ID
    try:
        return hailo.VDevice(params)
    except Exception as exc:
        if held_elsewhere(exc):
            raise held_error() from exc
        raise


class HailoDetector:
    """YOLOX detector HEF compiled by scripts/hailo/compile_yolox_hef.py; HailoRT runs the NMS on the host.

    The infer model is configured once with explicit formats (UINT8 NHWC in; FLOAT32 HAILO_NMS_BY_CLASS
    out, with score, IoU and per-class box limits), and a ring of bindings with their buffers is
    allocated once and reused. Two groups of frames can be in flight: the host fills one while the
    device runs the other.

    close() waits until settle_s has passed since the last completed job before shutting the model
    down. On the UGen300 (USB, HailoRT 5.3.2) a shutdown() issued right after batched jobs finish
    can time out after 10 s and reset the device, although every job has completed on the host and
    the async queue is empty. In our trials (batch 8, 8-100 frames per cycle) 8 of 13 immediate
    shutdowns failed; with a 0.5 s or 2 s gap, 14 of 14 succeeded.
    """

    def __init__(self, vdevice, hef: Path, confidence: float = 0.40, nms_iou: float = 0.45, max_boxes: int = 100,
                 batch_size: int = 1, timeout_ms: int = 10000, settle_s: float | None = None, turns: Turns | None = None):
        import numpy as np
        from hailo_platform import FormatOrder, FormatType
        self.turns = turns or Turns()
        self.hef, self.confidence, self.nms_iou = Path(hef), float(confidence), float(nms_iou)
        self.max_boxes, self.batch_size, self.timeout_ms = int(max_boxes), int(batch_size), int(timeout_ms)
        self.settle_s, self.last_job = float(SETTLE_S if settle_s is None else settle_s), None
        self.configured = None
        self.model = vdevice.create_infer_model(str(hef))
        self.model.set_batch_size(self.batch_size)
        source = self.model.input()
        if list(source.shape) != [INPUT_SIZE, INPUT_SIZE, 3]:
            raise DeviceError(f"{self.hef.name} takes input {list(source.shape)}, not the 416x416x3 letterbox of the compact detector.")
        source.set_format_type(FormatType.UINT8)
        source.set_format_order(FormatOrder.NHWC)
        output = self.model.output()
        if not output.is_nms:
            raise DeviceError(f"{self.hef.name} has no HailoRT NMS output; compile it with nms_postprocess (scripts/hailo/compile_yolox_hef.py).")
        classes = self.model.hef.get_output_vstream_infos()[0].nms_shape.number_of_classes
        if classes != len(CLASSES):
            raise DeviceError(f"{self.hef.name} has {classes} NMS classes; the compact detector has {len(CLASSES)} (person, forklift).")
        output.set_format_type(FormatType.FLOAT32)
        output.set_format_order(FormatOrder.HAILO_NMS_BY_CLASS)
        output.set_nms_score_threshold(self.confidence)
        output.set_nms_iou_threshold(self.nms_iou)
        output.set_nms_max_proposals_per_class(self.max_boxes)
        self.output_name = self.model.output_names[0]
        self.configured = self.model.configure()
        # wait_for_async_ready times out if asked for more frames than the async queue holds.
        self.group = max(1, min(self.batch_size, int(self.configured.get_async_queue_size())))
        shape = tuple(self.model.output().shape)  # flat by-class host size once the order is set
        self.slots = []
        for _ in range(2 * self.group):
            frame = np.empty((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            result = np.empty(shape, dtype=np.float32)
            bindings = self.configured.create_bindings(output_buffers={self.output_name: result})
            bindings.input().set_buffer(frame)
            self.slots.append((bindings, frame, result))

    def detect(self, letterboxed, width: int, height: int) -> list[dict]:
        """letterboxed: uint8 416x416x3 BGR from letterbox(); returns source-pixel detections."""
        return self.detect_batch([(letterboxed, width, height)])[0]

    def detect_batch(self, frames: list) -> list[list[dict]]:
        """frames: (letterboxed, source width, source height) tuples; results come back in order.

        Every job is collected before the device turn is released, so no batch is in flight afterwards,
        also when a step fails: the jobs already submitted are waited for before the error goes up.
        """
        with self.turns.hold("detector"):
            pending = []
            try:
                return self._detect_batch(frames, pending)
            except BaseException:
                for job, _, _ in pending:
                    try:
                        job.wait(self.timeout_ms)
                    except Exception:
                        pass  # the original error is the one to report
                raise

    def _detect_batch(self, frames: list, pending: list) -> list[list[dict]]:
        results = []
        for start in range(0, len(frames), self.group):
            group = frames[start:start + self.group]
            half = (start // self.group) % 2 * self.group
            slots = self.slots[half:half + len(group)]
            if len(pending) == 2:  # the slots about to be refilled belong to the older job
                results += self._collect(*pending[0])
                pending.pop(0)
            for (_, frame, _), (canvas, _, _) in zip(slots, group):
                if canvas.shape != frame.shape or canvas.dtype != frame.dtype:
                    raise ValueError(f"Detector input must be uint8 {frame.shape}, got {canvas.dtype} {canvas.shape}.")
                frame[...] = canvas
            self.configured.wait_for_async_ready(timeout_ms=self.timeout_ms, frames_count=len(group))
            pending.append((self.configured.run_async([s[0] for s in slots]), slots, group))
        while pending:
            results += self._collect(*pending[0])
            pending.pop(0)
        return results

    def _collect(self, job, slots, group):
        job.wait(self.timeout_ms)
        self.last_job = time.monotonic()
        return [to_source_boxes(bindings.output().get_buffer(), width, height, confidence=self.confidence)
                for (bindings, _, _), (_, width, height) in zip(slots, group)]

    def last_buffers(self) -> tuple:
        """Raw and HailoRT-converted output of the first slot, for recording a test fixture on the device."""
        bindings, _, result = self.slots[0]
        return result.copy(), bindings.output().get_buffer()

    def close(self):
        if self.configured is not None:
            with self.turns.hold("detector"):
                if self.last_job is not None:  # let the device settle after the last job (see the class docstring)
                    time.sleep(max(0.0, self.settle_s - (time.monotonic() - self.last_job)))
                self.configured.shutdown()  # blocks until in-flight jobs finish
                self.configured = None
        self.slots = []


class ObjectEnd:
    """Where the first complete top-level JSON object ends, in text that arrives a token at a time.

    Text before the first "{" is skipped; braces inside JSON strings (with backslash escapes) do not count.
    """

    def __init__(self):
        self.seen, self.depth, self.quoted, self.escaped, self.end = 0, 0, False, False, None

    def feed(self, piece: str) -> int | None:
        for char in piece:
            if self.end is not None:
                break
            self.seen += 1
            if not self.depth:
                self.depth = 1 if char == "{" else 0
            elif self.quoted:
                self.escaped, self.quoted = (False, True) if self.escaped else (char == "\\", char != '"')
            elif char == '"':
                self.quoted = True
            elif char in "{}":
                self.depth += 1 if char == "{" else -1
                if not self.depth:
                    self.end = self.seen
        return self.end


def json_object_end(text: str) -> int | None:
    """Index just past the first complete top-level JSON object in text, or None."""
    return ObjectEnd().feed(text)


def read_reply(model, start) -> dict:
    """One greedy generation on a GenAI model (VLM or LLM), left as soon as the reply holds a complete JSON object.

    start() opens the model's token stream. Both prompts ask for one JSON object, and nothing after it is ever read:
    the look-v2 parser and drafting.parse_draft take the text from the first "{" to the last "}". On the device the
    VLM sometimes went on after the closing brace until the token limit (15 of 148 Stage I crops, 38 s instead of
    14 s); leaving the stream ended the generation within half a second. A reply is cut at the end of its first
    object, so a trailing "}" cannot spoil the parse either. The context is cleared before and after: system
    messages are only accepted on a fresh context, and nothing carries over between items.
    """
    model.clear_context()
    started = time.perf_counter()
    first = last = None
    tokens, scan = [], ObjectEnd()
    try:
        with start() as stream:
            for token in stream:
                last = time.perf_counter()
                first = last if first is None else first
                tokens.append(token)
                if scan.feed(token) is not None:
                    break
    finally:
        model.clear_context()
    text = "".join(tokens)[:scan.end]
    for end in END_TOKENS:
        text = text.replace(end, "")
    return {"text": text, "tokens": len(tokens), "seconds": round(time.perf_counter() - started, 3),
            "ttft_s": round(first - started, 3) if first is not None else None,
            "tokens_per_s": round((len(tokens) - 1) / (last - first), 3) if len(tokens) > 1 and last > first else None,
            "stopped_at_object_end": scan.end is not None}


class HailoVLM:
    """Qwen2-VL-2B-Instruct HEF from Hailo's GenAI Model Zoo, loaded once and kept resident.

    It answers the visual check (pair crop + the frozen look-v2 prompt) and drafts the brief from
    text alone (frames=[], which the VLM API documents as generation without vision input). The
    context is cleared before and after every item: system messages are only accepted on a fresh
    context, and nothing carries over between candidates.
    """

    def __init__(self, vdevice, hef: Path, turns: Turns | None = None, device_format: str = drafting.DEVICE_PROMPT_VERSION):
        if device_format not in drafting.DEVICE_FORMATS:
            raise DeviceError(f"Unknown device draft format {device_format!r}; drafting.py has {sorted(drafting.DEVICE_FORMATS)}.")
        from hailo_platform.genai import VLM
        self.device_format = device_format
        self.turns = turns or Turns()
        self.vlm = VLM(vdevice, str(hef))
        self.frame_shape = tuple(int(v) for v in self.vlm.input_frame_shape())  # (336, 336, 3) for Qwen2-VL-2B

    def generate(self, prompt: list, frames: list, max_tokens: int) -> dict:
        with self.turns.hold("vlm"):
            return self._generate(prompt, frames, max_tokens)

    def _generate(self, prompt: list, frames: list, max_tokens: int) -> dict:
        """Greedy generation, left at the end of the reply's JSON object (read_reply)."""
        return read_reply(self.vlm, lambda: self.vlm.generate(prompt=prompt, frames=frames, max_generated_tokens=max_tokens,
                                                              do_sample=False, seed=SEED))

    def check(self, crop) -> dict:
        """Visual check of one pair crop (PIL image) with the look-v2 prompt and parser of the CPU path."""
        prompt = [{"role": "system", "content": [{"type": "text", "text": VLM_SYSTEM}]},
                  {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": look_prompt.PROMPT}]}]
        result = self.generate(prompt, [vlm_frame(crop, self.frame_shape)], LOOK_TOKENS)
        entry = {"raw_text": result["text"],
                 **{k: result[k] for k in ("seconds", "ttft_s", "tokens", "tokens_per_s", "stopped_at_object_end")}}
        try:
            entry["answer"] = look_prompt.parse(result["text"])
        except ValueError as exc:
            entry["error"] = str(exc)
        return entry

    def draft(self, event: dict) -> dict:
        """The draft-v2 messages answered from text alone; the reply is parsed and checked by the caller.

        The device format hint (drafting.DEVICE_FORMATS[device_format]) is appended to the system message.
        """
        system, user = drafting.prompt(event)
        system = f"{system} {drafting.DEVICE_FORMATS[self.device_format]}"
        prompt = [{"role": "system", "content": [{"type": "text", "text": system}]},
                  {"role": "user", "content": [{"type": "text", "text": user}]}]
        return {**self.generate(prompt, [], DRAFT_TOKENS), "prompt_sha256": hashlib.sha256((system + user).encode()).hexdigest(),
                "prompt_version": self.device_format}

    def close(self):
        with self.turns.hold("vlm"):  # not while a generation is running in another thread
            self.vlm.release()


class HailoLLM:
    """Qwen2.5-1.5B-Instruct HEF from Hailo's GenAI Model Zoo: the CPU path's draft model, compiled for the device.

    It drafts the brief with the CPU worker's settings where HailoRT has them (scripts/draft_worker.py): the draft-v2
    messages, greedy decoding, seed 19, at most 320 tokens and no repetition penalty. The HEF defaults to
    frequency_penalty 1.1 (Qwen2.5's repetition_penalty; the VLM HEF's default is 1.0), so it is set to 1.0 here, as
    llama.cpp's repeat_penalty 1.0 in the worker. There is no JSON grammar, so the device format hint is appended to
    the system message, as for the VLM, and the reply is parsed and checked afterwards. Message content is plain
    text, the form this HEF's chat template takes. Only one GenAI model may be loaded on the device at a time
    (hailo_model_zoo_genai docs/USAGE.rst), so the VLM must be released before this model is loaded.

    The app does not use it. In a comparison fixed in advance (data/evaluation/stage2/device/draft-llm, UGen300 at
    0.9 A) none of its 20 drafts passed the source checks (it wrote the questions under "reviewQuestions"), it drafted
    more slowly than the VLM, and loading the VLM again after it made the device drop off USB every time.
    """

    def __init__(self, vdevice, hef: Path, turns: Turns | None = None, device_format: str = drafting.DEVICE_PROMPT_VERSION):
        if device_format not in drafting.DEVICE_FORMATS:
            raise DeviceError(f"Unknown device draft format {device_format!r}; drafting.py has {sorted(drafting.DEVICE_FORMATS)}.")
        from hailo_platform.genai import LLM
        self.device_format = device_format
        self.turns = turns or Turns()
        self.llm = LLM(vdevice, str(hef))

    def generate(self, prompt: list, max_tokens: int) -> dict:
        with self.turns.hold("llm"):
            return read_reply(self.llm, lambda: self.llm.generate(prompt=prompt, max_generated_tokens=max_tokens, do_sample=False,
                                                                  seed=SEED, frequency_penalty=LLM_FREQUENCY_PENALTY))

    def draft(self, event: dict) -> dict:
        """The draft-v2 messages with the device format hint; the reply is parsed and checked by the caller."""
        system, user = drafting.prompt(event)
        system = f"{system} {drafting.DEVICE_FORMATS[self.device_format]}"
        prompt = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return {**self.generate(prompt, DRAFT_TOKENS), "prompt_sha256": hashlib.sha256((system + user).encode()).hexdigest(),
                "prompt_version": self.device_format}

    def close(self):
        with self.turns.hold("llm"):  # not while a generation is running in another thread
            self.llm.release()


class HailoRuntime:
    """Everything this process runs on the UGen300: one VDevice, the detector and the VLM, loaded once.

    Construction raises DeviceError before any model is loaded when HailoRT is missing, no device is
    found, another process holds it, the HailoRT release differs from the manifest's or a HEF's
    SHA-256 does not match. Both models share one Turns, so they never run at the same time.
    """

    def __init__(self, manifest: Path, root: Path, confidence: float = 0.40, nms_iou: float = 0.45,
                 batch_size: int = 1, max_boxes: int = 100):
        hailo = hailort()
        self.version = getattr(hailo, "__version__", None)
        self.identity = identify(hailo)
        self.models = verify_manifest(manifest, root, self.version)
        self.settings = {"confidence": float(confidence), "nms_iou": float(nms_iou), "batch_size": int(batch_size),
                         "max_boxes_per_class": int(max_boxes)}
        self.load_s = {}
        self.turns = Turns()
        self.vdevice = self.detector = self.vlm = None
        try:
            self.vdevice = shared_vdevice(hailo)
            tick = time.perf_counter()
            self.detector = HailoDetector(self.vdevice, self.models["detector"]["path"], confidence, nms_iou, max_boxes, batch_size,
                                          turns=self.turns)
            self.load_s["detector"] = round(time.perf_counter() - tick, 3)
            tick = time.perf_counter()
            self.vlm = HailoVLM(self.vdevice, self.models["vlm"]["path"], turns=self.turns)
            self.load_s["vlm"] = round(time.perf_counter() - tick, 3)
        except DeviceError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise DeviceError(f"Loading the UGen300 models failed: {type(exc).__name__}: {exc}") from exc

    def record(self) -> dict:
        """Identity, versions and HEF hashes, for the health endpoint and every result made on the device."""
        return {"backend": "hailo", "hailort_version": self.version, "devices": self.identity["devices"],
                "hefs": {name: {k: m[k] for k in ("file", "sha256", "source")} for name, m in self.models.items()},
                "load_seconds": dict(self.load_s), "vdevice": {"group_id": GROUP_ID, "scheduling": "ROUND_ROBIN",
                                                               "models": "one at a time; jobs run in phases"},
                "detector": {**self.settings, "frames_per_submission": self.detector.group if self.detector else None},
                "vlm": {"frame_shape": list(self.vlm.frame_shape) if self.vlm else None, "system_prompt": VLM_SYSTEM,
                        "decoding": "greedy (do_sample=False)", "seed": SEED}}

    def hardware_label(self) -> str:
        """The device that ran the job, as it identified itself."""
        device = self.identity["devices"][0]
        name = device.get("architecture") or "Hailo device"
        if device.get("product_name"):
            name += f" ({device['product_name']})"
        firmware = [f"firmware {device['firmware']}"] if device.get("firmware") else []
        return " · ".join([name, f"device {device['id']}", *firmware, f"HailoRT {self.version}"])

    def close(self):
        """Release the VLM, the detector and then the VDevice; safe to call more than once."""
        for name in ("vlm", "detector"):
            part = getattr(self, name, None)
            if part is not None:
                try:
                    part.close()
                except Exception as exc:
                    print(f"TraceBrief: releasing the UGen300 {name} failed: {exc}", file=sys.stderr)
                setattr(self, name, None)
        if getattr(self, "vdevice", None) is not None:
            try:
                self.vdevice.release()
            except Exception as exc:
                print(f"TraceBrief: releasing the VDevice failed: {exc}", file=sys.stderr)
            self.vdevice = None
