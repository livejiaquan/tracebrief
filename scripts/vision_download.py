#!/usr/bin/env python3
"""Explicitly fetch the pinned Apache-2.0 detector; inference stays offline."""
import argparse,hashlib,json
from pathlib import Path
from huggingface_hub import snapshot_download
MODEL='IDEA-Research/grounding-dino-tiny'
REVISION='a2bb814dd30d776dcf7e30523b00659f4f141c71'
p=argparse.ArgumentParser();p.add_argument('--output',default='models/grounding-dino-tiny');a=p.parse_args()
out=Path(a.output);snapshot_download(MODEL,revision=REVISION,local_dir=out,allow_patterns=['*.json','*.txt','*.safetensors','README.md','LICENSE*'])
files={str(f.relative_to(out)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in out.rglob('*') if f.is_file() and '.cache' not in f.parts}
(out/'tracebrief_model_manifest.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'files':files},indent=2))
print(f'TRACEBRIEF_MODEL_DIR={out.resolve()}')
