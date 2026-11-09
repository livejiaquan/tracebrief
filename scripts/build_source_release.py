#!/usr/bin/env python3
"""Build a source-only release from an explicit allowlist after local content checks.

Environments, runtime records, private research, caches and Git metadata are
excluded; our two detector weights in weights/ are included, and every file must
stay under GitHub's 100 MB per-file limit. Run --check while documents are still
being edited; package only after final review. The scan supplements manual
review, not a secrets guarantee.

SOURCE_MANIFEST.json at the repository root lists every released file with its size and SHA-256
(--write-manifest rewrites it after the checks pass); tests/test_source_manifest.py fails when the
tree no longer matches it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP = {'README.md', 'LICENSE', 'NOTICE', 'pyproject.toml', '.gitignore'}
TREES = {'src', 'web', 'config', 'docs', 'tests', 'scripts', '.github'}
WEIGHTS = {'.onnx'}
MANIFEST = 'SOURCE_MANIFEST.json'
GITHUB_FILE_LIMIT = 100 * 1024 * 1024  # GitHub rejects larger files in a push
EXCLUDE = {'__pycache__', 'pycache', '.pytest_cache', '.runtime', '.git', 'models', 'dist', 'build'}
TEXT_SUFFIXES = {'.py', '.js', '.html', '.css', '.md', '.json', '.toml', '.txt', '.sh', '.ps1', '.yml', '.yaml'}
PATTERNS = {
    'private-home-path': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/'),
    'Windows-user-path': re.compile(r'[A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)[A-Za-z0-9_.-]+'),  # also JSON-escaped
    'provider-secret-shape': re.compile(r'\b(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16})\b'),
}


def allowed(path, root=ROOT):
    relative = path.relative_to(root)
    if any(part in EXCLUDE or part.startswith('.venv') for part in relative.parts):
        return False
    if len(relative.parts) == 1:
        return relative.name in TOP
    if relative.parts[0] == 'weights':
        return path.suffix in WEIGHTS or path.suffix == '.md'
    if relative.parts[:2] == ('docs', 'images'):
        return path.suffix in {'.png', '.jpg'}
    if relative.parts[:2] == ('tests', 'fixtures'):
        return path.suffix in TEXT_SUFFIXES or path.suffix == '.npz'
    if relative.parts[0] in TREES:
        return path.suffix in TEXT_SUFFIXES or relative.name.startswith(('LICENSE', 'NOTICE'))
    if relative.parts[:3] == ('data', 'evaluation', 'look'):
        return path.suffix in {'.json', '.md', '.jpg'}
    if relative.parts[:3] == ('data', 'evaluation', 'stage2'):
        # Device and host-load records ship with the scripts that drove the device (host paths are arguments).
        return path.suffix in {'.json', '.jsonl', '.md', '.csv', '.txt', '.py'}
    if relative.parts[:2] == ('data', 'evaluation'):
        return path.suffix in {'.json', '.jsonl', '.md', '.csv', '.txt'}
    if relative.parts[:2] == ('data', 'samples'):
        return path.suffix in {'.mp4', '.json', '.txt', '.md'}
    return False


def release_files(root=ROOT):
    """Every file the release contains, sorted by path."""
    return sorted(p for p in root.rglob('*') if p.is_file() and allowed(p, root))


def manifest_entry(path, root=ROOT):
    content = path.read_bytes()
    return {'path': path.relative_to(root).as_posix(), 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Scan only; do not create a release')
    parser.add_argument('--output', type=Path, default=ROOT/'dist'/'tracebrief-source.zip')
    parser.add_argument('--write-manifest', action='store_true', help=f'rewrite {MANIFEST} after the checks pass')
    args = parser.parse_args()
    files = release_files()
    issues = []
    entries = []
    payloads = {}
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            issues.append({'file': relative, 'kind': 'symlink-not-permitted'})
            continue
        content = path.read_bytes()
        if len(content) >= GITHUB_FILE_LIMIT:
            issues.append({'file': relative, 'kind': 'over-github-file-limit', 'bytes': len(content)})
        payloads[relative] = content
        entries.append(manifest_entry(path))
        if path.suffix not in {'.mp4', '.jpg', '.png', '.onnx', '.npz'}:
            text = content.decode('utf-8', errors='replace')
            for label, pattern in PATTERNS.items():
                for match in pattern.finditer(text):
                    issues.append({'file': relative, 'line': text.count('\n', 0, match.start())+1, 'kind': label})
    manifest_path = ROOT/'data'/'samples'/'manifest.json'
    for sample in json.loads(manifest_path.read_text(encoding='utf-8')).get('samples', []):
        clip = ROOT/'data'/'samples'/sample['path']
        if not clip.is_file() or hashlib.sha256(clip.read_bytes()).hexdigest() != sample['sha256']:
            issues.append({'file': sample['path'], 'kind': 'sample-hash-mismatch-or-missing'})
    report = {'checked_files': len(entries), 'bytes': sum(e['bytes'] for e in entries), 'issues': issues, 'scope': 'Explicit source and sample allowlist; pattern scan is not a guarantee that all secrets or rights issues are absent.'}
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)
    manifest_text = json.dumps({'files': entries}, indent=2)+'\n'
    if args.write_manifest:
        (ROOT/MANIFEST).write_text(manifest_text, encoding='utf-8', newline='\n')
        print(f'Wrote {MANIFEST}: {len(entries)} files')
    if args.check:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in entries:
            info = zipfile.ZipInfo('tracebrief/'+entry['path'], date_time=(2026, 10, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payloads[entry['path']])
        info = zipfile.ZipInfo('tracebrief/'+MANIFEST, date_time=(2026, 10, 20, 0, 0, 0))
        archive.writestr(info, manifest_text)
    print(f'Created {args.output} sha256={hashlib.sha256(args.output.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
