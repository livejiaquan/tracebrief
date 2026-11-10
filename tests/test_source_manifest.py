"""SOURCE_MANIFEST.json lists exactly the released files with their current SHA-256.

It is written by `python scripts/build_source_release.py --check --write-manifest` after the release
checks pass. A source tree that has no manifest yet (before its first release) skips this test.
"""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_source_release as release  # noqa: E402

MANIFEST = ROOT / release.MANIFEST


def digest(path):
    """SHA-256 as released; a checkout that turned LF into CRLF (Git on Windows) still matches its text files."""
    content = path.read_bytes()
    digests = {hashlib.sha256(content).hexdigest()}
    if b"\r\n" in content and b"\0" not in content:  # text files, including LICENSE, NOTICE and .gitignore
        digests.add(hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest())
    return digests


@pytest.mark.skipif(not MANIFEST.is_file(), reason="no SOURCE_MANIFEST.json in this tree (written at release)")
def test_source_manifest_matches_the_tree():
    listed = {entry["path"]: entry["sha256"] for entry in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]}
    present = {path.relative_to(ROOT).as_posix(): path for path in release.release_files()}
    rerun = "rerun python scripts/build_source_release.py --check --write-manifest"
    assert sorted(set(present) - set(listed)) == [], f"released files missing from {release.MANIFEST}; {rerun}"
    assert sorted(set(listed) - set(present)) == [], f"{release.MANIFEST} lists files that are not in the tree; {rerun}"
    stale = sorted(name for name, path in present.items() if listed[name] not in digest(path))
    assert stale == [], f"{release.MANIFEST} has stale hashes; {rerun}"
