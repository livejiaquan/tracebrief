#!/usr/bin/env python3
"""Check source-checkout runtime prerequisites without running model inference.

With TRACEBRIEF_BACKEND=hailo it checks the UGen300 path instead of the CPU models: the HailoRT
Python package and its release, the host packages the device path imports, and every HEF of the
manifest (TRACEBRIEF_HAILO_MANIFEST, default config/hailo-manifest.json) against its recorded
SHA-256. Run it with the interpreter that will run the server. It does not open the device.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
checks = []
backend = os.environ.get('TRACEBRIEF_BACKEND', 'cpu').strip().lower() or 'cpu'

def check(name, ready, detail, required=True):
    checks.append({'check': name, 'ready': bool(ready), 'detail': detail, 'required': required})

def imports(label, interpreter, packages):
    expression = ';'.join(f'import {name}' for name in packages)
    try:
        result = subprocess.run([interpreter, '-c', expression], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
        check(label, result.returncode == 0, 'Imports succeeded' if result.returncode == 0 else result.stderr[-2000:])
    except (OSError, subprocess.TimeoutExpired) as exc:
        check(label, False, str(exc))

check('Python', sys.version_info >= (3, 10), sys.version.split()[0])
check('backend', backend in ('cpu', 'hailo'), f'TRACEBRIEF_BACKEND={backend}' if backend in ('cpu', 'hailo') else f'TRACEBRIEF_BACKEND must be cpu or hailo, not {backend!r}')
for binary in ('ffmpeg', 'ffprobe'):
    # TRACEBRIEF_FFMPEG / TRACEBRIEF_FFPROBE name the executable; otherwise it is looked up on PATH.
    found = shutil.which(os.environ.get(f'TRACEBRIEF_{binary.upper()}') or binary)
    check(binary, found, found or f'Install ffmpeg using your OS package manager, or set TRACEBRIEF_{binary.upper()} to the executable')
for asset in ('web/index.html', 'web/app.js', 'config/vision.json', 'config/search_profiles.json', 'weights/yolox-s-warehouse-v4.onnx', 'weights/yolox-tiny-warehouse.onnx', 'data/samples/manifest.json'):
    check(asset, (root/asset).is_file(), asset)

if backend == 'hailo':
    # The device path runs in the server's own process, so this interpreter needs HailoRT and the host packages.
    sys.path.insert(0, str(root/'src'))
    check('Python 3.10 (ASUS ships HailoRT 5.3.2 for the UGen300 on Python 3.10; other versions are not tested)',
          sys.version_info[:2] == (3, 10), sys.version.split()[0], required=False)
    imports('device host imports', sys.executable, ['numpy', 'cv2', 'PIL'])
    try:
        from tracebrief import hailo_backend
        version = hailo_backend.hailort().__version__
        check('HailoRT', True, f'hailo_platform {version}')
    except Exception as exc:  # missing wheel, wrong platform, broken install
        version = None
        check('HailoRT', False, f'{exc.__class__.__name__}: {exc}. Install the HailoRT 5.3.x wheel that the HailoRT installer puts in its python folder.')
    manifest = Path(os.environ.get('TRACEBRIEF_HAILO_MANIFEST', root/'config'/'hailo-manifest.json'))
    try:
        models = hailo_backend.verify_manifest(manifest, root, version)
        check('HEFs', True, {name: f"{model['file']} sha256={model['sha256']}" for name, model in models.items()})
    except Exception as exc:  # DeviceError names the missing file, the unrecorded hash or the mismatch
        check('HEFs', False, f'{manifest}: {exc}')
else:
    vision_python = os.environ.get('TRACEBRIEF_VISION_PYTHON', sys.executable)
    llm_python = os.environ.get('TRACEBRIEF_LLM_PYTHON', sys.executable)
    imports('vision imports', vision_python, ['torch', 'torchvision', 'transformers', 'cv2', 'PIL', 'numpy', 'scipy', 'onnxruntime'])
    imports('generation and visual-check imports', llm_python, ['llama_cpp', 'llama_cpp.mtmd_cpp'])
    model_dir = Path(os.environ.get('TRACEBRIEF_MODEL_DIR', root/'models/grounding-dino-tiny'))
    check('vision model', (model_dir/'config.json').is_file() and any(model_dir.glob('*.safetensors')), str(model_dir))
    llm_model = Path(os.environ.get('TRACEBRIEF_LLM_MODEL', root/'models/qwen/qwen2.5-1.5b-instruct-q4_k_m.gguf'))
    check('generation model', llm_model.is_file(), str(llm_model))
    vlm_model = os.environ.get('TRACEBRIEF_VLM_MODEL'); vlm_projector = os.environ.get('TRACEBRIEF_VLM_PROJECTOR')
    check('visual-check model (optional)', vlm_model and vlm_projector and Path(vlm_model).is_file() and Path(vlm_projector).is_file(),
          f'{vlm_model} + {vlm_projector}' if vlm_model else 'Not configured: candidates are reviewed without the visual check', required=False)
ready = all(row['ready'] for row in checks if row['required'])
scope = ('UGen300 path: HailoRT, host imports and HEF hashes; the device is not opened.' if backend == 'hailo'
         else 'Files, tools and imports only; this does not execute inference or validate model accuracy.')
print(json.dumps({'ready': ready, 'backend': backend, 'scope': scope, 'checks': checks}, indent=2))
sys.exit(0 if ready else 1)
