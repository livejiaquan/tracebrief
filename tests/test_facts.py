"""docs/FACTS.md is current, and every number marked <!-- fact: KEY --> in README.md and docs/*.md matches it.

scripts/build_facts.py recomputes the facts from data/evaluation/, config/ and a pytest collection, so a
changed record, a stale FACTS.md or an edited number fails here with the file, line and expected value.
"""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def build():
    spec = importlib.util.spec_from_file_location("build_facts", ROOT / "scripts" / "build_facts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("build_facts", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report(build):
    return build.collect(ROOT)


def test_facts_md_is_current(build, report):
    expected = build.render(report).splitlines()
    path = ROOT / build.FACTS_MD
    actual = path.read_text(encoding="utf-8").replace("\r\n", "\n").splitlines() if path.is_file() else []
    changed = [f"line {n}: expected {e!r}, found {a!r}" for n, (e, a) in enumerate(zip(expected, actual), 1) if e != a]
    if len(expected) != len(actual):
        changed.append(f"expected {len(expected)} lines, found {len(actual)}")
    assert not changed, f"{build.FACTS_MD} is out of date; run `python scripts/build_facts.py`.\n" + "\n".join(changed[:5])


def test_marked_numbers_match_the_facts(build, report):
    problems, checked = build.check_documents(ROOT, report)
    assert not problems, "\n".join(problems)
    assert checked >= 20, f"only {checked} marked numbers found in README.md and docs/"


def test_settings_in_use_are_the_frozen_ones(report):
    changed = [f"{c.record} vs {c.compared}" for c in report.checks if c.required and not c.matches]
    assert not changed, "Frozen settings or files changed: " + "; ".join(changed)


@pytest.mark.parametrize("quote, ok", [
    ("**98.1%** (51/52)", True),
    ("98% (51/52)", True),  # less precise, correctly rounded
    ("98.08%", True),  # more precise, still the same value
    ("**97.1%** (51/52)", False),
    ("98.1% (51/53)", False),
    ("98.1% | 86.4% (51/59)", True),  # a table cell border ends the quote
])
def test_a_marked_number_must_round_to_the_fact(build, report, quote, ok):
    problems, checked = build.check_text("doc.md", f"Fast: <!-- fact: fast.pair_precision -->{quote}\n", report.facts())
    assert checked == 1 and (not problems) == ok, problems


def test_unknown_keys_and_marks_without_a_number_are_reported(build, report):
    problems, _ = build.check_text("doc.md", "a <!-- fact: no.such.key -->1 b <!-- fact: look.crops --> crops\n", report.facts())
    assert len(problems) == 2 and "unknown fact 'no.such.key'" in problems[0] and "not followed by a number" in problems[1]
