#!/usr/bin/env python3
"""Check source-checkout runtime prerequisites without running model inference."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
checks = []

def check(name, ready, detail, required=True):
    checks.append({'check': name, 'ready': bool(ready), 'detail': detail, 'required': required})

check('Python', sys.version_info >= (3, 11), sys.version.split()[0])
for binary in ('ffmpeg', 'ffprobe'):
    check(binary, shutil.which(binary), shutil.which(binary) or 'Install ffmpeg using your OS package manager')
for asset in ('web/index.html', 'web/app.js', 'config/vision.json', 'config/search_profiles.json', 'weights/yolox-tiny-warehouse.onnx', 'data/samples/manifest.json'):
    check(asset, (root/asset).is_file(), asset)
vision_python = os.environ.get('TRACEBRIEF_VISION_PYTHON', sys.executable)
llm_python = os.environ.get('TRACEBRIEF_LLM_PYTHON', sys.executable)
for label, interpreter, packages in (('vision imports', vision_python, ['torch', 'torchvision', 'transformers', 'cv2', 'PIL', 'numpy', 'scipy', 'onnxruntime']), ('generation and visual-check imports', llm_python, ['llama_cpp', 'llama_cpp.mtmd_cpp'])):
    expression = ';'.join(f'import {name}' for name in packages)
    try:
        result = subprocess.run([interpreter, '-c', expression], capture_output=True, text=True, timeout=60)
        check(label, result.returncode == 0, 'Imports succeeded' if result.returncode == 0 else result.stderr[-2000:])
    except (OSError, subprocess.TimeoutExpired) as exc:
        check(label, False, str(exc))
model_dir = Path(os.environ.get('TRACEBRIEF_MODEL_DIR', root/'models/grounding-dino-tiny'))
check('vision model', (model_dir/'config.json').is_file() and any(model_dir.glob('*.safetensors')), str(model_dir))
llm_model = Path(os.environ.get('TRACEBRIEF_LLM_MODEL', root/'models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf'))
check('generation model', llm_model.is_file(), str(llm_model))
vlm_model = os.environ.get('TRACEBRIEF_VLM_MODEL'); vlm_projector = os.environ.get('TRACEBRIEF_VLM_PROJECTOR')
check('visual-check model (optional)', vlm_model and vlm_projector and Path(vlm_model).is_file() and Path(vlm_projector).is_file(),
      f'{vlm_model} + {vlm_projector}' if vlm_model else 'Not configured: candidates are reviewed without the visual check', required=False)
ready = all(row['ready'] for row in checks if row['required'])
print(json.dumps({'ready': ready, 'scope': 'Files, tools and imports only; this does not execute inference or validate model accuracy.', 'checks': checks}, indent=2))
sys.exit(0 if ready else 1)
