#!/usr/bin/env python3
"""Download the pinned official Qwen GGUF and provenance, or verify an existing file.

Uses Python standard library; downloading is an explicit online setup operation.
The inference process itself has no model download path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

REPO = 'Qwen/Qwen2.5-1.5B-Instruct-GGUF'
REVISION = '91cad51170dc346986eccefdc2dd33a9da36ead9'
FILENAME = 'qwen2.5-1.5b-instruct-q4_k_m.gguf'
SHA256 = '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
BASE = f'https://huggingface.co/{REPO}/resolve/{REVISION}/'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def verify(path):
    actual = digest(path)
    if actual != SHA256:
        raise SystemExit(f'Model hash mismatch: expected {SHA256}, got {actual}.')
    print(f'Verified {path.name}: sha256={actual}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/qwen'))
    parser.add_argument('--verify-file', type=Path, help='Verify this existing GGUF without network access or writes')
    args = parser.parse_args()
    if args.verify_file:
        verify(args.verify_file)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for name in (FILENAME, 'LICENSE', 'README.md'):
        target = args.output / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + '.part')
            print(f'Downloading {name} from pinned official Qwen repository…', flush=True)
            with urllib.request.urlopen(BASE + name, timeout=120) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output, length=4*1024*1024)
            if name == FILENAME:
                verify(temporary)
            temporary.replace(target)
    verify(args.output / FILENAME)
    files = {name: {'sha256': digest(args.output/name), 'bytes': (args.output/name).stat().st_size} for name in (FILENAME, 'LICENSE', 'README.md')}
    manifest = {'repository': REPO, 'revision': REVISION, 'license': 'Apache-2.0', 'quantization': 'Q4_K_M', 'source_url': f'https://huggingface.co/{REPO}/tree/{REVISION}', 'files': files}
    (args.output/'tracebrief_model_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'TRACEBRIEF_LLM_MODEL={args.output.resolve()/FILENAME}')


if __name__ == '__main__':
    main()
