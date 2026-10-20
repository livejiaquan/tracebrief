"""Isolated vision runner; input/output JSON provide an auditable process boundary."""
import json
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tracebrief.vision import analyze_video
import tracebrief.vision as vision_module

request = json.loads(Path(sys.argv[1]).read_text())
if request["settings"].get("backend") == "compact-onnx":
    import tracebrief.compact_vision as vision_module
    analyze_video = vision_module.analyze_video
implementation_sha256 = hashlib.sha256(Path(vision_module.__file__).read_bytes()).hexdigest()
result = analyze_video(request["path"], request["settings"])
result["implementation"] = {"vision_sha256": implementation_sha256}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2, default=str))
