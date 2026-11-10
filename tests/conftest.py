"""A fake hailo_platform for the UGen300 path, and the marker for tests that need a real device.

The fake follows the HailoRT 5.3-5.5 Python API surface TraceBrief uses (pyhailort.py): VDevice,
Device.scan/identify, InferModel streams and formats, ConfiguredInferModel bindings and run_async,
genai.VLM and genai.LLM. It records calls so tests can check how the device is driven, not what a model sees:
fake.timeline lists the detector jobs and GenAI generations in the order they started, fake.overlaps
every moment a detector job was in flight while a GenAI model generated, fake.genai every GenAI load and
release in order, and fake.genai_overlaps every load made while another GenAI model was still loaded
(the device takes one at a time). fake.held makes it answer as a UGen300 on USB that another process
holds (HailoRT 5.3.2 on Windows). fake.job_error is the exception the next detector jobs report: as in HailoRT 5.3,
only through the completion callback, while wait() returns as usual.
"""
from contextlib import contextmanager
import importlib.util
import json
import os
import re
import sys
import time
import types

import pytest

BOX = 5
LOOK_REPLY = '{"forklift": "yes", "person": "yes", "description": "A worker walks beside a forklift in the aisle."}'


def pytest_collection_modifyitems(config, items):
    """Tests marked ugen300 run only when HailoRT is installed and Device.scan() finds a device."""
    present = False
    if importlib.util.find_spec("hailo_platform") is not None:
        try:
            from hailo_platform import Device
            present = bool(Device.scan())
        except Exception:
            present = False
    skip = pytest.mark.skip(reason="needs a UGen300 (Hailo-10H) and HailoRT; none found")
    for item in items:
        if "ugen300" in item.keywords and not present:
            item.add_marker(skip)


def encode_by_class(per_class, max_boxes):
    """HAILO_NMS_BY_CLASS host layout (hailort.h): per class a float32 box count, then the boxes."""
    import numpy as np
    flat = []
    for boxes in per_class:
        flat.append(float(len(boxes)))
        for box in boxes:
            flat.extend(float(v) for v in box)
    buffer = np.zeros(len(per_class) * (1 + BOX * max_boxes), dtype=np.float32)
    buffer[:len(flat)] = flat
    return buffer


def decode_by_class(buffer, classes):
    """What HailoRT's get_buffer() returns for HAILO_NMS_BY_CLASS: one (n, 5) array per class."""
    per_class, offset = [], 0
    for _ in range(classes):
        count = int(buffer[offset]); offset += 1
        per_class.append(buffer[offset:offset + BOX * count].reshape(count, BOX))
        offset += BOX * count
    return per_class


def draft_reply(prompt):
    """A draft that passes the source checks, citing every evidence ID the prompt lists."""
    content = prompt[-1]["content"]
    user = content if isinstance(content, str) else content[-1]["text"]  # the LLM takes plain text, the VLM a list
    ids = re.findall(r"^- ([^:]+):", user, re.M)
    return json.dumps({"title": "Person beside forklift in aisle",
                       "summary": "A worker walks beside a forklift in the aisle.",
                       "review_questions": ["Is another camera view available?", "Which site procedure applies?"],
                       "evidence_ids": ids})


def make_fake_hailo(devices=("0001:01:00.0",), classes=2, queue=8, version="5.3.0"):
    fake = types.ModuleType("hailo_platform")
    fake.__version__ = version
    fake.devices = list(devices)
    fake.classes, fake.queue = classes, queue
    fake.vdevices, fake.models, fake.vlms, fake.llms, fake.opened = [], [], [], [], []
    fake.genai, fake.genai_overlaps = [], []  # ("load" | "release", "vlm" | "llm") in order; loads beside a loaded model
    fake.fault = None
    fake.timeline, fake.overlaps = [], []  # ("detect", frames), ("look",) or ("draft",) as each starts; both models active at once
    fake.in_flight = fake.generating = 0
    fake.delay = 0.0  # seconds each detector job and each generated token take, to widen the windows in concurrency tests
    fake.held = False  # another process holds the device
    fake.job_error = None  # what completion_info.exception carries for each detector job submitted from now on
    fake.HailoRTException = type("HailoRTException", (Exception,), {})
    fake.HailoRTStatusException = type("HailoRTStatusException", (fake.HailoRTException,), {})
    fake.boxes = lambda frame: [[] for _ in range(classes)]
    fake.reply = lambda prompt, frames: LOOK_REPLY if frames else draft_reply(prompt)
    fake.FormatType = types.SimpleNamespace(AUTO="AUTO", UINT8="UINT8", UINT16="UINT16", FLOAT32="FLOAT32")
    fake.FormatOrder = types.SimpleNamespace(AUTO="AUTO", NHWC="NHWC", HAILO_NMS_BY_CLASS="HAILO_NMS_BY_CLASS",
                                             HAILO_NMS_BY_SCORE="HAILO_NMS_BY_SCORE")
    fake.HailoSchedulingAlgorithm = types.SimpleNamespace(NONE="NONE", ROUND_ROBIN="ROUND_ROBIN")

    class Device:
        @classmethod
        def scan(cls):
            return list(fake.devices)

        def __init__(self, device_id=None):
            if fake.held:  # what identify() gets on the UGen300 over USB while another process holds it
                raise fake.HailoRTStatusException("98")
            self.device_id = device_id
            fake.opened.append(device_id)
            self.control = types.SimpleNamespace(identify=lambda: types.SimpleNamespace(
                device_architecture=types.SimpleNamespace(name="HAILO10H"), firmware_version="5.3.0 (release,app)",
                serial_number="UGEN300-0001\x00\x00", part_number="HM21LB1C2LAE", product_name="UGen300"))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Stream:
        def __init__(self, shape, nms=False):
            self._shape, self.is_nms = list(shape), nms
            self.format_type = self.format_order = None
            self.nms = {}

        @property
        def shape(self):
            if self.format_order == "HAILO_NMS_BY_CLASS":
                return [classes * (1 + BOX * self.nms.get("max_proposals_per_class", 100))]
            return self._shape

        def set_format_type(self, value):
            self.format_type = value

        def set_format_order(self, value):
            self.format_order = value

        def set_nms_score_threshold(self, value):
            self.nms["score_threshold"] = value

        def set_nms_iou_threshold(self, value):
            self.nms["iou_threshold"] = value

        def set_nms_max_proposals_per_class(self, value):
            self.nms["max_proposals_per_class"] = value

    class BindingStream:
        def __init__(self, convert=None):
            self.buffer, self.convert = None, convert

        def set_buffer(self, buffer):
            assert buffer.flags.c_contiguous
            self.buffer = buffer

        def get_buffer(self, tf_format=False):
            return self.convert(self.buffer) if self.convert and tf_format is not None else self.buffer

    class Bindings:
        def __init__(self, output_buffers):
            self._input = BindingStream()
            self._output = BindingStream(lambda buffer: decode_by_class(buffer, classes))
            for buffer in (output_buffers or {}).values():
                self._output.set_buffer(buffer)

        def input(self, name=""):
            return self._input

        def output(self, name=""):
            return self._output

    class Job:
        done = False

        def __init__(self, callback=None, error=None):
            self.callback, self.error = callback, error

        def wait(self, timeout_ms):  # like AsyncInferJobImpl::wait: success once the job and its callback ran
            self.waited = timeout_ms
            if not self.done:
                time.sleep(fake.delay)
                self.done = True
                fake.in_flight -= 1
                if self.callback:
                    self.callback(completion_info=types.SimpleNamespace(exception=self.error))

    class Configured:
        def __init__(self, model):
            self.model, self.bindings, self.submissions, self.shut = model, [], [], False

        def get_async_queue_size(self):
            return fake.queue

        def create_bindings(self, input_buffers=None, output_buffers=None):
            bindings = Bindings(output_buffers)
            self.bindings.append(bindings)
            return bindings

        def wait_for_async_ready(self, timeout_ms=1000, frames_count=1):
            assert frames_count <= fake.queue

        def run_async(self, bindings, callback=None):
            if fake.fault:
                raise fake.fault
            if fake.generating:
                fake.overlaps.append("a detector job was submitted while the VLM generated")
            fake.in_flight += 1
            fake.timeline.append(("detect", len(bindings)))
            max_boxes = self.model.output_stream.nms["max_proposals_per_class"]
            for item in bindings:
                item.output().buffer[...] = encode_by_class(fake.boxes(item.input().buffer), max_boxes)
            self.submissions.append(len(bindings))
            return Job(callback, fake.job_error)

        def shutdown(self):
            self.shut = True

    class InferModel:
        def __init__(self, vdevice, path):
            self.vdevice, self.path = vdevice, path
            self.input_stream, self.output_stream = Stream([416, 416, 3]), Stream([], nms=True)
            self.output_names = ["yolox_tiny_warehouse/nms_postprocess"]
            self.batch_size = self.configured = self.formats_at_configure = None
            info = types.SimpleNamespace(nms_shape=types.SimpleNamespace(number_of_classes=classes, max_bboxes_per_class=100))
            self.hef = types.SimpleNamespace(get_output_vstream_infos=lambda: [info])

        def set_batch_size(self, value):
            self.batch_size = value

        def input(self, name=""):
            return self.input_stream

        def output(self, name=""):
            return self.output_stream

        def configure(self):
            i, o = self.input_stream, self.output_stream
            self.formats_at_configure = (i.format_type, i.format_order, o.format_type, o.format_order, dict(o.nms))
            self.configured = Configured(self)
            return self.configured

    class VDevice:
        def __init__(self, params=None):
            if fake.held:
                raise fake.HailoRTException("libhailort failed with error: 98 (HAILO_LIBUSB_FAILURE)")
            self.params, self.released = params, False
            fake.vdevices.append(self)

        @staticmethod
        def create_params():
            return types.SimpleNamespace(scheduling_algorithm=None, group_id=None, multi_process_service=False)

        def create_infer_model(self, hef_source, name=""):
            assert not self.released
            model = InferModel(self, hef_source)
            fake.models.append(model)
            return model

        def release(self):
            self.released = True

    def load(kind, model):
        loaded = [m for m in fake.vlms + fake.llms if not m.released]
        if loaded:
            fake.genai_overlaps.append(f"{kind} loaded while {len(loaded)} GenAI model(s) were loaded")
        fake.genai.append(("load", kind))
        (fake.vlms if kind == "vlm" else fake.llms).append(model)

    class VLM:
        def __init__(self, vdevice, model_path, optimize_memory_on_device=False):
            assert not vdevice.released
            self.vdevice, self.path, self.calls, self.events, self.released = vdevice, model_path, [], [], False
            load("vlm", self)

        def input_frame_shape(self):
            return [336, 336, 3]

        def input_frame_format_type(self):
            import numpy as np
            return np.uint8

        def clear_context(self):
            self.events.append("clear")

        @contextmanager
        def generate(self, prompt, frames, videos=None, temperature=None, top_p=None, top_k=None,
                     frequency_penalty=None, max_generated_tokens=None, do_sample=None, seed=None):
            if any(m.get("role") == "system" for m in prompt):  # HailoRT accepts system messages only on a fresh context
                assert self.events[-1:] == ["clear"]
            self.events.append("generate")
            self.calls.append({"prompt": prompt, "frames": frames, "max_generated_tokens": max_generated_tokens,
                               "do_sample": do_sample, "seed": seed, "temperature": temperature})
            if fake.in_flight:
                fake.overlaps.append("the VLM generated while a detector job was in flight")
            fake.generating += 1
            fake.timeline.append(("look",) if frames else ("draft",))
            reply = fake.reply(prompt, frames)

            def tokens():
                for token in re.findall(r"\S+\s*", reply) + ["<|im_end|>"]:
                    time.sleep(fake.delay)
                    yield token
            try:
                yield tokens()
            finally:
                fake.generating -= 1

        def release(self):
            if not self.released:
                fake.genai.append(("release", "vlm"))
            self.released = True
            fake.vlm_release_order = [v.released for v in fake.vdevices]

    class LLM:
        """genai.LLM: plain-text chat messages, no frames; generate() returns a context manager that yields tokens."""

        def __init__(self, vdevice, model_path, lora_name="", optimize_memory_on_device=False):
            assert not vdevice.released
            self.vdevice, self.path, self.calls, self.events, self.released = vdevice, model_path, [], [], False
            load("llm", self)

        def clear_context(self):
            assert not self.released
            self.events.append("clear")

        @contextmanager
        def generate(self, prompt, tools=None, temperature=None, top_p=None, top_k=None, frequency_penalty=None,
                     max_generated_tokens=None, do_sample=None, seed=None):
            assert not self.released
            assert all(isinstance(m["content"], str) for m in prompt)  # the HEF's chat template takes plain text
            if any(m.get("role") == "system" for m in prompt):
                assert self.events[-1:] == ["clear"]
            self.events.append("generate")
            self.calls.append({"prompt": prompt, "max_generated_tokens": max_generated_tokens, "do_sample": do_sample,
                               "seed": seed, "temperature": temperature, "frequency_penalty": frequency_penalty})
            if fake.in_flight:
                fake.overlaps.append("the LLM generated while a detector job was in flight")
            fake.generating += 1
            fake.timeline.append(("draft",))
            reply = fake.reply(prompt, [])

            def tokens():
                for token in re.findall(r"\S+\s*", reply) + ["<|im_end|>"]:
                    time.sleep(fake.delay)
                    yield token
            try:
                yield tokens()
            finally:
                fake.generating -= 1

        def release(self):
            if not self.released:
                fake.genai.append(("release", "llm"))
            self.released = True

    fake.Device, fake.VDevice = Device, VDevice
    fake.genai_module = types.ModuleType("hailo_platform.genai")
    fake.genai_module.VLM, fake.genai_module.LLM, fake.genai_module.VDevice = VLM, LLM, VDevice
    return fake


@pytest.fixture
def fake_hailo(monkeypatch):
    """Install a fake hailo_platform (and hailo_platform.genai); returns a factory taking make_fake_hailo options."""
    def install(**options):
        from tracebrief import hailo_backend
        fake = make_fake_hailo(**options)
        monkeypatch.setitem(sys.modules, "hailo_platform", fake)
        monkeypatch.setitem(sys.modules, "hailo_platform.genai", fake.genai_module)
        monkeypatch.setattr(hailo_backend, "SETTLE_S", 0.0)  # the fake device needs no settle before shutdown
        return fake
    return install


@pytest.fixture
def manifest(tmp_path):
    """A manifest whose two stand-in HEF files match their recorded SHA-256."""
    import hashlib
    entries = {}
    for name, payload in (("detector", b"fixture detector hef"), ("vlm", b"fixture vlm hef")):
        path = tmp_path / "hefs" / f"{name}.hef"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)
        entries[name] = {"file": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "source": "test fixture"}
    path = tmp_path / "hailo-manifest.json"
    path.write_text(json.dumps({"hailort": "5.3", "models": entries}), encoding="utf-8")
    return path


# A stub llama_cpp for the resident CPU workers

STUB_LLAMA = '''"""Stub llama_cpp: deterministic replies that expose the model state, the prompt and the settings."""
import base64
import hashlib
import json
import os
import re
import sys
import time


class LlamaGrammar:
    @staticmethod
    def from_json_schema(schema, verbose=False):
        return ["grammar", json.loads(schema)]


class Llama:
    def __init__(self, model_path, chat_handler=None, seed=0, **settings):
        if os.environ.get("STUB_LLAMA_FAIL"):
            raise RuntimeError("stub load failure")
        print("stub llama: Python print while loading")  # neither may reach the worker protocol
        os.write(1, b"stub llama: C-level stdout while loading\\n")
        self.model_path, self.handler, self.seed, self.settings = model_path, chat_handler, seed, settings
        self.n_tokens = 0

    def reset(self):
        self.n_tokens = 0

    def set_seed(self, seed):
        self.seed = seed

    def create_chat_completion(self, messages, grammar=None, **options):
        text = json.dumps(messages, sort_keys=True)
        images = b"".join(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]) for message in messages
                          if isinstance(message["content"], list) for part in message["content"] if part["type"] == "image_url")
        if b"crash" in images:  # a crop whose bytes say crash or slow makes the model crash or hang
            os._exit(3)
        if b"slow" in images:
            time.sleep(60)
        os.write(1, b"stub llama: C-level stdout while answering\\n")
        print("stub llama: stderr while answering", file=sys.stderr)
        state = {"fresh": self.n_tokens == 0, "seed": self.seed, "messages": text, "grammar": grammar,
                 "options": sorted(options.items()), "settings": sorted(self.settings.items()),
                 "projector": getattr(self.handler, "clip_model_path", None)}
        self.n_tokens += 100  # an unreset model would answer the next request from this state
        # Letters only, so the digest adds no numbers for the draft checks to reject.
        digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest().translate(str.maketrans("0123456789", "ghijklmnop"))
        if self.handler is not None:
            content = json.dumps({"forklift": "yes", "person": "yes", "description": f"A worker stands beside a forklift {digest}."})
        else:
            ids = re.findall(r"- (event-[0-9]+-obs-[0-9]+):", messages[-1]["content"])
            content = json.dumps({"title": "Person beside forklift", "summary": f"Stub draft {digest}.",
                                  "review_questions": ["Is another camera view available?"], "evidence_ids": ids})
        return {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": len(text), "completion_tokens": 9}}
'''
STUB_CHAT_FORMAT = '''class MTMDChatHandler:
    def __init__(self, clip_model_path, verbose=True, use_gpu=True):
        self.clip_model_path, self.use_gpu = clip_model_path, use_gpu
'''


@pytest.fixture
def stub_llama(tmp_path, monkeypatch):
    """Put a stub llama_cpp (with package metadata) first on PYTHONPATH for child interpreters.

    Returns the folder; worker processes started by the test see the stub through os.environ, and
    every worker is closed when the test ends.
    """
    from tracebrief import workers
    folder = tmp_path / "stub-site"
    package = folder / "llama_cpp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(STUB_LLAMA, encoding="utf-8")
    (package / "llama_chat_format.py").write_text(STUB_CHAT_FORMAT, encoding="utf-8")
    metadata = folder / "llama_cpp_python-0.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: llama-cpp-python\nVersion: 0.0.0\n", encoding="utf-8")
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(folder)] + ([existing] if existing else [])))
    monkeypatch.setenv("TRACEBRIEF_RUNTIME", str(tmp_path / "runtime"))
    yield folder
    workers.close_all()


# Synthetic videos (ffmpeg's native mpeg4 encoder; no model, no sample clip)

def have_ffmpeg():
    import shutil
    from tracebrief import media
    return all(shutil.which(tool()) for tool in (media.ffmpeg, media.ffprobe))


needs_ffmpeg = pytest.mark.skipif(not have_ffmpeg(), reason="needs ffmpeg and ffprobe")


def vfr_output():
    """ffmpeg's flag for a variable-frame-rate output: -fps_mode vfr (ffmpeg 5.1+), else -vsync vfr."""
    import subprocess
    from tracebrief import media
    try:
        text = subprocess.run([media.ffmpeg(), "-hide_banner", "-h", "long"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        text = ""
    return ["-fps_mode", "vfr"] if "-fps_mode" in text else ["-vsync", "vfr"]


# Numbered clips for frame-exact extraction: frame n spells n in its pixels, so any image cut from it
# (an evidence frame, a visual-check crop) says which frame it came from.

NUMBER_BITS = 10  # frames 0-1023
ENCODERS: dict = {}


def numbered_frame(n, size=(160, 32)):
    """Frame n of a numbered clip: bit b of n is the b-th of ten vertical bands, light when set."""
    import numpy as np
    width, height = size
    image = np.full((height, width, 3), 16, dtype=np.uint8)
    for bit in range(NUMBER_BITS):
        if n >> bit & 1:
            image[:, bit * width // NUMBER_BITS:(bit + 1) * width // NUMBER_BITS] = 235
    return image


def frame_number(image) -> int:
    """The number numbered_frame drew, read from an image (array or file) of the frame's full width."""
    import numpy as np
    from PIL import Image
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as opened:
            image = opened.convert("L")
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 3:
        values = values.mean(axis=2)
    width = values.shape[1]
    return sum(1 << bit for bit in range(NUMBER_BITS)
               if values[:, int((bit + 0.25) * width / NUMBER_BITS):int((bit + 0.75) * width / NUMBER_BITS)].mean() > 128)


def has_encoder(name: str) -> bool:
    import subprocess
    from tracebrief import media
    if name not in ENCODERS:
        try:
            listing = subprocess.run([media.ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            listing = ""
        ENCODERS[name] = re.search(rf"^\s*V\S*\s+{re.escape(name)}\s", listing, re.M) is not None
    return ENCODERS[name]


def mark_every_sample_sync(path):
    """Flag every sample of a fragmented MP4 as a sync sample, as the SDG-Warehouse exports do.

    Their fragments (one per frame, no index) give each frame the flags sample_depends_on = 2 and
    not non-sync, so ffmpeg's seek index takes every frame for a keyframe: -ss lands on a frame the
    decoder cannot decode alone, and H.264 output resumes at the next real keyframe (MPEG-4 part 2
    shows damaged frames instead). Rewrites the tfhd default and trun sample flags in place.
    """
    import struct
    sync = (0x02000000).to_bytes(4, "big")
    data = bytearray(open(path, "rb").read())

    def boxes(start, end):
        while start + 8 <= end:
            size, kind = struct.unpack(">I4s", data[start:start + 8])
            yield start, size, kind
            start += size
    for moof, size, kind in boxes(0, len(data)):
        if kind != b"moof":
            continue
        for traf, traf_size, traf_kind in boxes(moof + 8, moof + size):
            if traf_kind != b"traf":
                continue
            for box, _, box_kind in boxes(traf + 8, traf + traf_size):
                flags = int.from_bytes(data[box + 9:box + 12], "big")
                if box_kind == b"tfhd" and flags & 0x20:  # default-sample-flags-present
                    at = box + 16 + sum(n for bit, n in ((0x1, 8), (0x2, 4), (0x8, 4), (0x10, 4)) if flags & bit)
                    data[at:at + 4] = sync
                elif box_kind == b"trun":
                    count, at = int.from_bytes(data[box + 12:box + 16], "big"), box + 16 + (4 if flags & 0x1 else 0)
                    if flags & 0x4:  # first-sample-flags-present
                        data[at:at + 4] = sync
                        at += 4
                    if flags & 0x400:  # sample-flags-present
                        per, offset = 4 * bin(flags & 0xF00).count("1"), 4 * bin(flags & 0x300).count("1")
                        for n in range(count):
                            data[at + n * per + offset:at + n * per + offset + 4] = sync
    open(path, "wb").write(bytes(data))
    return path


def make_numbered_video(path, count=300, fps=30, gop=30, layout="mp4", codec="mpeg4", drop_after=None, drop=(), size=(160, 32)):
    """A clip of count numbered frames (numbered_frame) with a keyframe every gop frames.

    layout: "mp4", or "fragmented": one fragment per frame, no index, every sample flagged sync (the
    SDG-Warehouse layout, mark_every_sample_sync). codec: "mpeg4" (ffmpeg's own encoder) or "libx264".
    drop_after: keep every frame before it, then every other one, with the original timestamps
    (a variable frame rate; frame k then shows source frame number k, k + (k - drop_after), ...).
    drop: source frames left out with the others' timestamps kept, as a camera that drops a frame
    (the average rate stays within ffprobe's sign of a variable rate).
    """
    import subprocess
    from tracebrief import media
    width, height = size
    raw = b"".join(numbered_frame(n, size).tobytes() for n in range(count))
    keep = f"lt(n,{drop_after})+not(mod(n,2))" if drop_after else "*".join(f"not(eq(n,{n}))" for n in drop)
    vfr = ["-vf", f"select='{keep}'", *vfr_output()] if keep else []
    keyframes = ["-g", str(gop)]
    encoder = (["-c:v", "libx264", "-crf", "12", "-threads", "2", "-keyint_min", str(gop), "-sc_threshold", "0", *keyframes]
               if codec == "libx264" else ["-c:v", "mpeg4", "-q:v", "2", "-sc_threshold", "1000000000", *keyframes])
    container = ["-bf", "0", "-video_track_timescale", str(fps), "-movflags", "frag_every_frame+empty_moov+skip_trailer"] \
        if layout == "fragmented" else ([] if codec == "libx264" else ["-bf", "0"])
    subprocess.run([media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", *vfr, *encoder, "-pix_fmt", "yuv420p", *container, str(path)],
                   input=raw, check=True)
    return mark_every_sample_sync(path) if layout == "fragmented" else path
