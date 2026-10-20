#!/usr/bin/env python3
"""Download the pinned Qwen2-VL-2B GGUF and its vision projector, or verify existing files.

Uses Python standard library; downloading is an explicit online setup operation.
The visual-check process itself has no model download path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

REPO = 'ggml-org/Qwen2-VL-2B-Instruct-GGUF'
REVISION = 'bb307c036e8a1ed7b663bbd0c35b41c4c9294cfd'
FILES = {
    'Qwen2-VL-2B-Instruct-Q4_K_M.gguf': '5745685d2e607a82a0696c1118e56a2a1ae0901da450fd9cd4f161c6b62867d7',
    'mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf': 'a0ad91f00a7a80dcf84d719a61b00ee2e07b71794f4ee2dfa81a254621a8c418',
}
BASE = f'https://huggingface.co/{REPO}/resolve/{REVISION}/'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def verify(path, expected):
    actual = digest(path)
    if actual != expected:
        raise SystemExit(f'{path.name}: hash mismatch, expected {expected}, got {actual}.')
    print(f'Verified {path.name}: sha256={actual}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/qwen2-vl'))
    parser.add_argument('--verify', action='store_true', help='Verify files already in --output without network access')
    args = parser.parse_args()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, expected in FILES.items():
            target = args.output / name
            if target.exists():
                continue
            temporary = target.with_suffix(target.suffix + '.part')
            print(f'Downloading {name} from {REPO}…', flush=True)
            with urllib.request.urlopen(BASE + name, timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
            verify(temporary, expected)
            temporary.replace(target)
    for name, expected in FILES.items():
        verify(args.output / name, expected)
    manifest = {'repository': REPO, 'revision': REVISION, 'base_model': 'Qwen/Qwen2-VL-2B-Instruct', 'license': 'Apache-2.0',
                'source_url': f'https://huggingface.co/{REPO}/tree/{REVISION}',
                'files': {name: {'sha256': sha, 'bytes': (args.output / name).stat().st_size} for name, sha in FILES.items()}}
    if not args.verify:
        (args.output / 'tracebrief_model_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    model, projector = (args.output.resolve() / name for name in FILES)
    print(f'TRACEBRIEF_VLM_MODEL={model}\nTRACEBRIEF_VLM_PROJECTOR={projector}')


if __name__ == '__main__':
    main()
