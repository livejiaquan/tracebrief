#!/usr/bin/env python3
"""Build a source-only release from an explicit allowlist after local content checks.

Weights, environments, runtime records, private research, caches and Git metadata
are excluded. Run --check while documents are still being edited; package only
after final review. The scan supplements manual review, not a secrets guarantee.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP = {'README.md', 'LICENSE', 'NOTICE', 'pyproject.toml', '.gitignore'}
TREES = {'src', 'web', 'config', 'docs', 'tests', 'scripts'}
WEIGHTS = {'.onnx'}
EXCLUDE = {'__pycache__', 'pycache', '.pytest_cache', '.runtime', '.git', 'models', 'dist', 'build'}
TEXT_SUFFIXES = {'.py', '.js', '.html', '.css', '.md', '.json', '.toml', '.txt', '.sh', '.yml', '.yaml'}
PATTERNS = {
    'private-home-path': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/'),
    'Windows-user-path': re.compile(r'[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9_.-]+'),
    'provider-secret-shape': re.compile(r'\b(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16})\b'),
}


def allowed(path):
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDE or part.startswith('.venv') for part in relative.parts):
        return False
    if len(relative.parts) == 1:
        return relative.name in TOP
    if relative.parts[0] == 'weights':
        return path.suffix in WEIGHTS or path.suffix == '.md'
    if relative.parts[:2] == ('docs', 'images'):
        return path.suffix in {'.png', '.jpg'}
    if relative.parts[0] in TREES:
        return path.suffix in TEXT_SUFFIXES or relative.name.startswith(('LICENSE', 'NOTICE'))
    if relative.parts[:3] == ('data', 'evaluation', 'look'):
        return path.suffix in {'.json', '.md', '.jpg'}
    if relative.parts[:2] == ('data', 'evaluation'):
        return path.suffix in {'.json', '.md', '.csv', '.txt'}
    if relative.parts[:2] == ('data', 'samples'):
        return path.suffix in {'.mp4', '.json', '.txt', '.md'}
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Scan only; do not create a release')
    parser.add_argument('--output', type=Path, default=ROOT/'dist'/'tracebrief-source.zip')
    args = parser.parse_args()
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and allowed(p))
    issues = []
    entries = []
    payloads = {}
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            issues.append({'file': relative, 'kind': 'symlink-not-permitted'})
            continue
        content = path.read_bytes()
        payloads[relative] = content
        entries.append({'path': relative, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
        if path.suffix not in {'.mp4', '.jpg', '.png', '.onnx'}:
            text = content.decode('utf-8', errors='replace')
            for label, pattern in PATTERNS.items():
                for match in pattern.finditer(text):
                    issues.append({'file': relative, 'line': text.count('\n', 0, match.start())+1, 'kind': label})
    manifest_path = ROOT/'data'/'samples'/'manifest.json'
    for sample in json.loads(manifest_path.read_text()).get('samples', []):
        clip = ROOT/'data'/'samples'/sample['path']
        if not clip.is_file() or hashlib.sha256(clip.read_bytes()).hexdigest() != sample['sha256']:
            issues.append({'file': sample['path'], 'kind': 'sample-hash-mismatch-or-missing'})
    report = {'checked_files': len(entries), 'bytes': sum(e['bytes'] for e in entries), 'issues': issues, 'scope': 'Explicit source and sample allowlist; pattern scan is not a guarantee that all secrets or rights issues are absent.'}
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)
    if args.check:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in entries:
            info = zipfile.ZipInfo('tracebrief/'+entry['path'], date_time=(2026, 10, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payloads[entry['path']])
        info = zipfile.ZipInfo('tracebrief/SOURCE_MANIFEST.json', date_time=(2026, 10, 20, 0, 0, 0))
        archive.writestr(info, json.dumps({'files': entries}, indent=2)+'\n')
    print(f'Created {args.output} sha256={hashlib.sha256(args.output.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
