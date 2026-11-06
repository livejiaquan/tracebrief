"""ffmpeg and ffprobe calls of the non-frozen code, with configurable executables.

TRACEBRIEF_FFMPEG and TRACEBRIEF_FFPROBE name the executables (for example a full path to
ffmpeg.exe on Windows); otherwise they are looked up on PATH.

frame_at writes the frame a single-clip detector analysed, so the evidence frames and the
visual-check crop of the CPU and UGen300 paths show the moment of the boxes. The frozen
look.crop_frame seeks with -ss instead, which is not frame-exact (docs/ERRATA.md); it stays only to
reproduce Stage I.
"""
from __future__ import annotations

from fractions import Fraction
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

from .look import MAX_SIDE

# frame_at: targets this close to the start are found by counting decoded frames from the first one;
# later targets by their exact timestamp, decoding from SEEK_BACK_S before them.
FULL_DECODE_S = 30.0
SEEK_BACK_S = 10.0
# showinfo's lines, by filter instance: each frame's integer pts and keyframe flag, and the time base they count in.
SHOWINFO = re.compile(r"\[Parsed_showinfo_(\d+) @[^]]*\]\s*n:\s*\d+\s+pts:\s*(-?\d+)\s.*?\biskey:\s*(\d)")
SHOWINFO_BASE = re.compile(r"\[Parsed_showinfo_(\d+) @[^]]*\]\s*config in time_base:\s*(\d+)/(\d+)")
STREAMS: dict = {}


def executable(name: str) -> str:
    """TRACEBRIEF_<NAME> if set, else the PATH match, else the bare name (so the error names it)."""
    return os.environ.get(f"TRACEBRIEF_{name.upper()}") or shutil.which(name) or name


def ffmpeg() -> str:
    return executable("ffmpeg")


def ffprobe() -> str:
    return executable("ffprobe")


def crop_filter(region: list[int]) -> str:
    """look.crop_frame's filter: the pair crop, then the longest side at most MAX_SIDE."""
    x, y, w, h = region
    scale = "scale='if(gt(iw,ih),min(iw,{0}),-2)':'if(gt(iw,ih),-2,min(ih,{0}))'".format(MAX_SIDE)
    return f"crop={w}:{h}:{x}:{y},{scale}"


def frame_index(seconds: float, fps: float) -> int:
    """The decoded-frame number a single-clip detector timed at seconds (it records round(n / fps, 3))."""
    return max(0, round(float(seconds) * float(fps)))


def _rate(value) -> float | None:
    try:
        top, _, bottom = str(value).partition("/")
        number = float(top) / float(bottom or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _identity(video: Path) -> tuple:
    stat = video.stat()
    return str(video.resolve()), stat.st_size, stat.st_mtime_ns


def stream(video: Path) -> dict:
    """The first video stream's frame rate as OpenCV reports it (average, else nominal), and its time base."""
    key = ("stream", *_identity(Path(video)))
    if key not in STREAMS:
        process = subprocess.run([ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=avg_frame_rate,r_frame_rate,time_base", "-of", "default=noprint_wrappers=1", str(video)],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        fields = dict(line.split("=", 1) for line in process.stdout.splitlines() if "=" in line)
        try:
            base = Fraction(fields.get("time_base", ""))
        except (ValueError, ZeroDivisionError):
            base = None
        STREAMS[key] = {"fps": _rate(fields.get("avg_frame_rate")) or _rate(fields.get("r_frame_rate")),
                        "time_base": base if base and base > 0 else None}
    return STREAMS[key]


def _decode(command: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=True)


def _frames_seen(log: str) -> dict:
    """showinfo instance -> [(timestamp in seconds as an exact fraction, keyframe)] for every frame it printed."""
    bases = {int(m[1]): Fraction(int(m[2]), int(m[3])) for m in SHOWINFO_BASE.finditer(log) if int(m[3])}
    seen: dict = {}
    for match in SHOWINFO.finditer(log):
        instance = int(match[1])
        if instance in bases:
            seen.setdefault(instance, []).append((int(match[2]) * bases[instance], match[3] == "1"))
    return seen


def _timestamps(video: Path) -> list | None:
    """Every frame's timestamp (exact seconds) in presentation order, from the packets; None if any is missing.

    Read from the container without decoding: the packets the decoder keeps (not flagged discard),
    sorted. Frame n of the detector's count is the n-th of them when every kept packet decodes to one
    frame; _seek checks the first one against the decoder.
    """
    key = ("timestamps", *_identity(video))
    if key not in STREAMS:
        base = stream(video)["time_base"]
        process = subprocess.run([ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts,flags",
                                  "-of", "csv=p=0", str(video)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 timeout=600)
        values = []
        for line in process.stdout.splitlines():
            pts, _, flags = line.strip().partition(",")
            if "D" in flags:
                continue
            values.append(int(pts) if re.fullmatch(r"-?\d+", pts) else None)
        STREAMS[key] = sorted(v * base for v in values) if process.returncode == 0 and base and values and None not in values else None
    return STREAMS[key]


def _first_frame(video: Path) -> Fraction | None:
    """The timestamp of the first frame the decoder returns (frame 0), exact."""
    key = ("first", *_identity(video))
    if key not in STREAMS:
        process = _decode([ffmpeg(), "-hide_banner", "-nostats", "-loglevel", "info", "-copyts", "-i", str(video), "-map", "0:v:0",
                           "-vf", "showinfo", "-frames:v", "1", "-f", "null", "-"], 60)
        frames = _frames_seen(process.stderr).get(0)
        STREAMS[key] = frames[0][0] if frames else None
    return STREAMS[key]


def _seek(video: Path, index: int, chain: list[str], tail: list[str], output: Path) -> bool:
    """Frame index found by its exact timestamp after a seek SEEK_BACK_S before it; False when that cannot be checked.

    The target is the index-th packet timestamp (_timestamps), used only if the first packet's is the
    first decoded frame's. The seek is not trimmed (-noaccurate_seek), so showinfo before the select
    prints the first frame the decoder returned after it, and showinfo after the select the frame
    kept. The seek counts only if that first frame is a keyframe at or before the target and the kept
    frame has exactly the target's timestamp. A seek can land on a frame the decoder cannot use: on
    fragmented MP4s that flag every frame as a sync sample, H.264 decoding then starts at the next
    real keyframe, after the target, and MPEG-4 part 2 returns damaged frames that are not keyframes.
    """
    times = _timestamps(video)
    if not times or index >= len(times) or _first_frame(video) != times[0]:
        return False
    target = times[index]
    lower = (times[index - 1] + target) / 2 if index else target - Fraction(1, 1000)
    begin = max(0.0, float(target - times[0]) - SEEK_BACK_S)
    graph = ",".join(["showinfo", f"select='gte(t,{float(lower):.9f})'", "showinfo", *chain])
    process = subprocess.run([ffmpeg(), "-hide_banner", "-nostats", "-loglevel", "info", "-noaccurate_seek", "-ss", f"{begin:.3f}",
                              "-copyts", "-i", str(video), "-map", "0:v:0", "-vf", graph, "-frames:v", "1", *tail, "-y", str(output)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120 + 2 * SEEK_BACK_S)
    seen = _frames_seen(process.stderr)
    first, kept = (seen.get(0) or [(None, False)])[0], (seen.get(2) or [(None, False)])[0]
    if process.returncode == 0 and output.is_file() and first[1] and first[0] <= target and kept[0] == target:
        return True
    output.unlink(missing_ok=True)
    return False


def frame_at(video: Path, seconds: float, output: Path, fps: float | None = None, filters: str | None = None,
             quality: int | None = None) -> Path:
    """Write the frame a single-clip detector analysed at seconds, through filters, as an image.

    That frame is decoded frame number n = round(seconds x fps), counted from the first frame in the
    order decoding returns them, exactly as compact_vision and vision count OpenCV's sequential reads
    (they record frame n at round(n / fps, 3)); fps is the detector's recorded rate, else the file's.
    On a constant-rate file without gaps frame n is the one whose timestamp is the first frame's +
    n / fps. On a variable-rate file, or one with dropped frames, it is still the detector's frame n,
    although its own timestamp can differ from the n / fps the record (and the review) shows.

    Targets within FULL_DECODE_S of the start are found by counting frames from the start. Later
    targets are found by the n-th frame's own timestamp after a seek (_seek), and by counting when
    that seek cannot be checked. FileNotFoundError: no frame n.
    """
    video, output = Path(video), Path(output)
    fps = float(fps or stream(video)["fps"] or 0)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{video.name} has no readable frame rate")
    index = frame_index(seconds, fps)
    chain = [filters] if filters else []
    tail = ["-q:v", str(quality)] if quality else []
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    if index / fps > FULL_DECODE_S:
        try:
            if _seek(video, index, chain, tail, output):
                return output
        except (OSError, ValueError, subprocess.SubprocessError):  # an unreadable index or seek: count instead
            output.unlink(missing_ok=True)
    _decode([ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(video), "-map", "0:v:0",
             "-vf", ",".join([f"select='eq(n,{index})'", *chain]), "-frames:v", "1", *tail, "-y", str(output)],
            120 + 2 * index / fps)
    if not output.is_file():
        raise FileNotFoundError(f"{video.name} has no frame {index} ({float(seconds):g} s at {fps:g} frames/s)")
    return output


def crop_at(video: Path, seconds: float, region: list[int], output: Path, fps: float | None = None) -> Path:
    """The visual check's pair crop (crop_filter, JPEG quality 2) of the frame frame_at finds."""
    return frame_at(video, seconds, output, fps, crop_filter(region), quality=2)


def duration(video: Path) -> float | None:
    """Container duration in seconds from ffprobe, or None when it cannot be read."""
    try:
        process = subprocess.run([ffprobe(), "-v", "error", "-show_entries", "format=duration",
                                  "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        value = float(process.stdout.strip().splitlines()[0]) if process.returncode == 0 and process.stdout.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return value if value is not None and math.isfinite(value) and value > 0 else None
