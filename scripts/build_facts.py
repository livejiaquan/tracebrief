#!/usr/bin/env python3
"""Recompute every headline number from the records in this repository and write docs/FACTS.md.

Each fact names the file it is computed from and the frozen setting it depends on. README.md and
docs/*.md mark a quoted fact with an HTML comment placed right before the number:

    it confirmed <!-- fact: look.tp_confirmed -->104 of 106 pairs

A marked number must agree with the fact: the numbers that follow the mark, up to the next mark,
table cell border or line end, are compared in order with the fact's numbers, each rounded half-up
to the decimals the document prints. tests/test_facts.py runs the same check, so a changed record,
a stale FACTS.md or a mistyped number fails the tests.

Standard library only; the test count comes from `python -m pytest --collect-only`.

    python scripts/build_facts.py           # rewrite docs/FACTS.md, then check the documents
    python scripts/build_facts.py --check   # exit 1 if FACTS.md is stale or a marked number differs
"""

import argparse
import ast
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EVAL = "data/evaluation"
FACTS_MD = "docs/FACTS.md"
TAG = re.compile(r"<!--\s*fact:\s*([A-Za-z0-9_.\-]+)\s*-->")
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)")
SPAN_END = re.compile(r"<!--|\||\n")


class FactError(ValueError):
    """The records disagree with each other, so no fact can be stated."""


@dataclass
class Fact:
    key: str
    values: list  # exact Decimals, in the order `text` prints them
    text: str
    source: str
    setting: str
    note: str = ""


@dataclass
class FreezeCheck:
    record: str
    compared: str
    matches: bool
    required: bool = True
    note: str = ""


@dataclass
class Report:
    sections: list = field(default_factory=list)  # (title, [Fact])
    checks: list = field(default_factory=list)  # [FreezeCheck]

    def facts(self) -> dict:
        return {fact.key: fact for _, facts in self.sections for fact in facts}


def dec(value) -> Decimal:
    """Exact decimal of a JSON number (repr of a float is its shortest round-trip form)."""
    return value if isinstance(value, Decimal) else Decimal(repr(value) if isinstance(value, float) else str(value))


def median(values) -> Decimal:
    ordered = sorted(dec(v) for v in values)
    if not ordered:
        raise FactError("median of no values")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def pct(numerator, denominator) -> Decimal:
    return Decimal(numerator) * 100 / Decimal(denominator)


def show(value, places: int | None = None) -> str:
    value = dec(value)
    if places is not None:
        value = value.quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP)
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def agrees(quoted: str, exact: Decimal) -> bool:
    """A quoted number agrees when the exact value, rounded half-up to the quoted decimals, equals it."""
    places = len(quoted.split(".", 1)[1]) if "." in quoted else 0
    return exact.quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP) == Decimal(quoted)


def load(root: Path, relative: str):
    with open(root / relative, encoding="utf-8") as handle:
        return json.load(handle)


def text_sha256(path: Path) -> str:
    """SHA-256 of a text file with LF line endings, so a Windows checkout (CRLF) hashes the same."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def count_tests(root: Path) -> int:
    """Test cases pytest collects from tests/ (skipped ones included), in a separate interpreter."""
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    process = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"],
                             cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=600)
    found = re.search(r"^(\d+) tests? collected", process.stdout, re.M)
    if process.returncode or not found:
        tail = "\n".join((process.stdout + process.stderr).strip().splitlines()[-15:])
        raise FactError(f"pytest could not collect tests/ with {sys.executable} (exit {process.returncode}); "
                        f"install pytest or fix the collection error:\n{tail}")
    return int(found.group(1))


def pair_facts(key: str, record: dict, source: str, setting: str) -> list:
    pair = record["pair_evidence"]
    tp, fp, fn = int(pair["tp"]), int(pair["fp"]), int(pair["fn"])
    for name, stored, exact in (("precision", pair["precision"], pct(tp, tp + fp) / 100), ("recall", pair["recall"], pct(tp, tp + fn) / 100)):
        if abs(dec(stored) - exact) > Decimal("1e-9"):
            raise FactError(f"{source}: stored pair {name} {stored} differs from {tp}/{tp + fp if name == 'precision' else tp + fn}")
    precision, recall = pct(tp, tp + fp), pct(tp, tp + fn)
    return [
        Fact(f"{key}.pair_counts", [Decimal(tp), Decimal(fp), Decimal(fn)], f"{tp} / {fp} / {fn} (TP / FP / FN)", source, setting,
             "A pair counts only when both predicted boxes match an annotated pair at IoU ≥ 0.5."),
        Fact(f"{key}.pair_precision", [precision, Decimal(tp), Decimal(tp + fp)], f"{show(precision, 1)}% ({tp}/{tp + fp})", source, setting),
        Fact(f"{key}.pair_recall", [recall, Decimal(tp), Decimal(tp + fn)], f"{show(recall, 1)}% ({tp}/{tp + fn})", source, setting),
    ]


def detector(root: Path, configs: dict) -> tuple:
    fast = load(root, f"{EVAL}/compact/v1/evaluation_fps1.json")
    fast5 = load(root, f"{EVAL}/compact/v1/evaluation_fps5.json")
    broader = load(root, f"{EVAL}/phase2/comparator_v1/evaluation.json")
    stricter = load(root, f"{EVAL}/phase2/evaluation.json")
    sizes = {(r["n_runs"], r["n_frames"]) for r in (fast, fast5, broader, stricter)}
    if len(sizes) != 1:
        raise FactError(f"The profiles were scored on different run/frame sets: {sorted(sizes)}")
    (runs, frames), = sizes
    gold = {r["pair_evidence"]["tp"] + r["pair_evidence"]["fn"] for r in (fast, fast5, broader, stricter)}
    if len(gold) != 1:
        raise FactError(f"The profiles were scored against different gold pair counts: {sorted(gold)}")
    compact, wide, strict = configs["vision-compact.json"], configs["vision-broader.json"], configs["vision-stricter.json"]
    fast_setting = (f"`config/vision-compact.json`: YOLOX-tiny ONNX `{compact['model_sha256'][:12]}`, confidence {show(compact['confidence'])}, "
                    f"{show(compact['sample_fps'])} frame/s; frozen in `compact/v1/FREEZE.json`")
    broader_setting = (f"`config/vision-broader.json`: Grounding DINO tiny `{wide['model_revision'][:8]}`, confidence {show(wide['confidence'])}, "
                       f"default 800/1333 resize, {show(wide['sample_fps'])} frame/s; frozen in `phase2/comparator_v1/freeze.json`")
    size = strict["processor_size"]
    stricter_setting = (f"`config/vision-stricter.json`: confidence {show(strict['confidence'])}, resize "
                        f"{size['shortest_edge']}/{size['longest_edge']}, {show(strict['sample_fps'])} frame/s; frozen in `phase2/freeze.json`")
    scope = f"`{EVAL}/phase2/evaluation.json`, `{EVAL}/phase2/comparator_v1/evaluation.json`, `{EVAL}/compact/v1/evaluation_fps1.json`"
    gold_pairs = gold.pop()
    facts = [
        Fact("test.scorable", [Decimal(runs), Decimal(frames)], f"{runs} scorable test runs, {frames} sampled frames", scope,
             "Run selection frozen in `phase2/freeze.json` before inference",
             "One of the 10 test runs is excluded for misaligned annotations (`phase2/coverage_exclusions.json`); its predictions are kept."),
        Fact("test.gold_pairs", [Decimal(gold_pairs)], f"{gold_pairs} annotated person–forklift pairs in those frames", scope,
             "Annotation boxes from `phase2/gold.json`, pair rule gap ratio 0.30"),
    ]
    facts += pair_facts("fast", fast, f"`{EVAL}/compact/v1/evaluation_fps1.json`", fast_setting)
    facts += pair_facts("broader", broader, f"`{EVAL}/phase2/comparator_v1/evaluation.json`", broader_setting)
    facts += pair_facts("stricter", stricter, f"`{EVAL}/phase2/evaluation.json`", stricter_setting)
    p5 = fast5["pair_evidence"]
    facts.append(Fact("fast.fps5_pair_counts", [Decimal(p5["tp"]), Decimal(p5["fp"]), Decimal(p5["fn"])],
                      f"{p5['tp']} / {p5['fp']} / {p5['fn']} (TP / FP / FN) at 5 frames/s", f"`{EVAL}/compact/v1/evaluation_fps5.json`",
                      "Same model and thresholds; 5 frames/s was a planned sensitivity run (`compact/v1/FREEZE.json`, `evaluation_plan`)",
                      "Scored on the same 1 frame/s gold frames. Not a shipped setting."))

    timings = load(root, f"{EVAL}/compact/v1/timings.json")
    one = [r for r in timings if r["fps"] == 1 and r["cohort"] == "primary"]
    five = [r for r in timings if r["fps"] == 5 and r["cohort"] == "primary"]
    if len(one) != 10 or len(five) != 10:
        raise FactError(f"compact/v1/timings.json should hold 10 primary clips per rate, found {len(one)} and {len(five)}")
    source = f"`{EVAL}/compact/v1/timings.json` (10 test clips, 1 frame/s)"
    setting = f"`config/vision-compact.json`: {compact['threads']} threads, {show(compact['sample_fps'])} frame/s"
    wall, load_s, inference = (median(r[k] for r in one) for k in ("wall_s", "model_load_s", "inference_s"))
    outside = median(dec(r["wall_without_model_load_s"]) - dec(r["inference_s"]) for r in one)
    outside5 = median(dec(r["wall_without_model_load_s"]) - dec(r["inference_s"]) for r in five)
    ms = median(r["inference_ms_per_frame"] for r in one)
    rate = Decimal(1000) / ms
    wall5 = median(r["wall_s"] for r in five)
    comparison = load(root, f"{EVAL}/phase2/comparator_v1/paired_comparison.json")
    wide_wall = dec(comparison["v1_all10_median_wall_s"])
    strict_wall = dec(comparison["v2_all10_runtime"]["median_wall_s"])
    timing = [
        Fact("fast.wall_s", [wall], f"{show(wall)} s per 10 s clip, end to end (median)", source, setting,
             "Includes model load, decoding, inference, grouping, evidence images and the source hash."),
        Fact("fast.load_s", [load_s], f"{show(load_s, 3)} s model load (hash and ONNX Runtime session; median)", source, setting),
        Fact("fast.inference_s", [inference], f"{show(inference, 3)} s inference per clip (median)", source, setting),
        Fact("fast.inference_ms", [ms], f"{show(ms)} ms per frame, inference only (median of per-clip means)", source, setting),
        Fact("fast.inference_fps", [rate], f"{show(rate, 1)} frames/s, inference only (1000 / median ms per frame)", source, setting),
        Fact("fast.other_s", [outside], f"{show(outside, 3)} s per clip outside the model (median)", source, setting,
             f"Wall time minus model load and inference: decoding every frame of the 30 fps clip, preprocessing, grouping, evidence "
             f"images and the source hash, not timed separately. At 5 frames/s (50 instead of 10 frames through the model) it was "
             f"{show(outside5, 3)} s, because every frame is decoded at either rate."),
        Fact("fast.fps5_wall_s", [wall5], f"{show(wall5)} s per 10 s clip at 5 frames/s (median)", f"`{EVAL}/compact/v1/timings.json`",
             "5 frames/s sensitivity run; not a shipped setting"),
        Fact("broader.wall_s", [wide_wall], f"{show(wide_wall)} s per 10 s clip (median of all 10 inputs)",
             f"`{EVAL}/phase2/comparator_v1/paired_comparison.json`", broader_setting, "Includes model loading; shared workstation."),
        Fact("stricter.wall_s", [strict_wall], f"{show(strict_wall)} s per 10 s clip (median of all 10 inputs)",
             f"`{EVAL}/phase2/comparator_v1/paired_comparison.json`", stricter_setting, "Includes model loading; shared workstation."),
        Fact("fast.speedup", [wide_wall / wall], f"{show(wide_wall / wall, 1)} times less vision wall time than Broader (median vs median)",
             "`fast.wall_s`, `broader.wall_s`", "As above", "Observed on a shared host; not a controlled benchmark."),
    ]
    return facts, timing


def detector_v4(root: Path, configs: dict) -> list:
    """Detector v4 (Stage II): the pre-registered comparison of the Stage I model ("old") with v4-S and v4-T."""
    base = f"{EVAL}/stage2/detector-v4"
    summary = load(root, f"{base}/out/summary.json")
    results, criteria = summary["results"], summary["criteria"]
    old, v3, small, tiny = (results[k] for k in ("old", "v3", "v4-S", "v4-T"))
    stage1 = load(root, f"{EVAL}/compact/v1/evaluation_fps1.json")["pair_evidence"]
    if [old["stage1"]["pair"][k] for k in ("tp", "fp", "fn")] != [stage1[k] for k in ("tp", "fp", "fn")]:
        raise FactError(f"{base}/out/summary.json: the Stage I model's pair counts differ from compact/v1/evaluation_fps1.json")
    fast = configs["vision-compact-v4.json"]
    if fast["model_sha256"] != small["freeze"]["onnx_sha256"] or dec(fast["confidence"]) != dec(small["freeze"]["confidence"]):
        raise FactError("config/vision-compact-v4.json is not the frozen v4-S model and confidence")
    setting = (f"`config/vision-compact-v4.json`: YOLOX-s ONNX `{fast['model_sha256'][:12]}`, confidence {show(fast['confidence'])}; "
               f"frozen in `stage2/detector-v4/freeze/v4-S/FREEZE.json`")
    old_setting = "Stage I YOLOX-tiny (`config/vision-compact.json`), confidence 0.4"
    source = f"`{base}/out/summary.json`"
    facts = pair_facts("v4", {"pair_evidence": small["stage1"]["pair"]}, source, setting)
    facts[0].note = "Stage I test frames (9 runs, 90 frames), current app pipeline; the Stage I model reproduces its published 51 / 1 / 8."
    event = lambda r: r["events_a1"]["primary_peak_time_and_location"]
    facts += [
        Fact("v4.events_strict", [Decimal(event(small)["strict"]), Decimal(event(small)["n"])],
             f"{event(small)['strict']} of {event(small)['n']} held-out synthetic events found (strict: peak time and location)", source, setting,
             "Pre-registered criterion C1b: at least 24 of 33."),
        Fact("v4.old_events_strict", [Decimal(event(old)["strict"]), Decimal(event(old)["n"])],
             f"{event(old)['strict']} of {event(old)['n']} held-out synthetic events found by the Stage I model", source, old_setting),
        Fact("v4.events_ci", [dec(x) * 100 for x in event(small)["recall_wilson95"] + event(old)["recall_wilson95"]],
             f"{show(dec(event(small)['recall_wilson95'][0]) * 100, 0)}–{show(dec(event(small)['recall_wilson95'][1]) * 100, 0)}% "
             f"for v4-S and {show(dec(event(old)['recall_wilson95'][0]) * 100, 0)}–{show(dec(event(old)['recall_wilson95'][1]) * 100, 0)}% "
             "for the Stage I model: Wilson 95% intervals of the strict held-out event recall", source, setting,
             "The intervals overlap almost entirely: one more event found is not a measured improvement."),
    ]
    loco = lambda r: r["loco_test"]["summary"]["forklift"]
    kb = lambda r: r["kb_test"]["summary"]
    loco_ci, kb_ci = small["loco_test"]["bootstrap"]["ci95"], small["kb_test"]["bootstrap"]["ci95"]
    loco_source = f"`{base}/out/v4-S/loco_test_results.json`, {source}"
    loco_note = (f"LOCO subset 3, a site never used for training or tuning: {small['loco_test']['summary']['images']} images, "
                 f"{loco(small)['gt']} forklifts. Scored once after the freeze.")
    facts += [
        Fact("v4.loco_ap50", [dec(loco(small)["ap50"])], f"{show(loco(small)['ap50'], 3)} forklift AP50 at the held-out real site (LOCO)",
             loco_source, setting, loco_note),
        Fact("v4.loco_pr", [dec(loco(small)["precision"]), dec(loco(small)["recall"]), Decimal(loco(small)["tp"]), Decimal(loco(small)["fp"])],
             f"{show(loco(small)['precision'], 3)} / {show(loco(small)['recall'], 3)} forklift precision / recall "
             f"({loco(small)['tp']} TP, {loco(small)['fp']} FP) at the held-out real site", loco_source, setting,
             "Pre-registered criterion C2a: AP50 and recall at least 0.20 above the Stage I model, precision at least 0.50."),
        Fact("v4.old_loco", [dec(loco(old)["ap50"]), dec(loco(old)["precision"]), dec(loco(old)["recall"])],
             f"{show(loco(old)['ap50'], 4)} / {show(loco(old)['precision'], 3)} / {show(loco(old)['recall'], 3)} forklift AP50 / precision / recall "
             f"of the Stage I model at the same site ({loco(old)['tp']} TP, {loco(old)['fp']} FP)", source, old_setting),
        Fact("v4.tiny_loco", [dec(loco(tiny)["ap50"]), dec(loco(tiny)["recall"]), dec(loco(tiny)["precision"])],
             f"{show(loco(tiny)['ap50'], 3)} / {show(loco(tiny)['recall'], 3)} / {show(loco(tiny)['precision'], 3)} forklift AP50 / recall / precision "
             f"of v4-T (YOLOX-tiny) at the held-out real site", source, f"v4-T, confidence {show(tiny['freeze']['confidence'])}; frozen in `stage2/detector-v4/freeze/v4-T/FREEZE.json`",
             "Recall is below the pre-registered 0.20, so v4-T fails C2a and is not adopted."),
        Fact("v4.kb_ap50", [dec(kb(small)["forklift"]["ap50"]), dec(kb(small)["person"]["ap50"])],
             f"{show(kb(small)['forklift']['ap50'], 3)} / {show(kb(small)['person']['ap50'], 3)} forklift / person AP50 on the keremberke test split "
             f"({kb(small)['images']} images)", f"`{base}/out/v4-S/kb_test_results.json`, {source}", setting,
             "Real forklift photos (Roboflow forklift-dsitv, CC BY 4.0); the test split was first used after registration."),
        Fact("v4.kb_pair_recall", [dec(kb(small)["pair"]["recall"]), Decimal(kb(small)["pair"]["detected"]), Decimal(kb(small)["pair"]["pair_images"])],
             f"{show(kb(small)['pair']['recall'], 2)} pair recall ({kb(small)['pair']['detected']} of {kb(small)['pair']['pair_images']} test images "
             f"with a person and a forklift)", f"`{base}/out/v4-S/kb_test_results.json`, {source}", setting),
        Fact("v4.loco_ci", [dec(x) for x in loco_ci["forklift_ap50"] + loco_ci["forklift_recall"]],
             f"{show(loco_ci['forklift_ap50'][0], 2)}–{show(loco_ci['forklift_ap50'][1], 2)} forklift AP50 and "
             f"{show(loco_ci['forklift_recall'][0], 2)}–{show(loco_ci['forklift_recall'][1], 2)} recall: bootstrap 95% intervals at the held-out real site",
             loco_source, setting, "Image-level bootstrap; the images come from one site, so they are correlated and the intervals are optimistic."),
        Fact("v4.kb_ci", [dec(x) for x in kb_ci["forklift_ap50"] + kb_ci["person_ap50"] + kb_ci["pair_recall"]],
             f"{show(kb_ci['forklift_ap50'][0], 2)}–{show(kb_ci['forklift_ap50'][1], 2)} forklift AP50, "
             f"{show(kb_ci['person_ap50'][0], 2)}–{show(kb_ci['person_ap50'][1], 2)} person AP50 and "
             f"{show(kb_ci['pair_recall'][0], 2)}–{show(kb_ci['pair_recall'][1], 2)} pair recall: bootstrap 95% intervals on the keremberke test split",
             f"`{base}/out/v4-S/kb_test_results.json`, {source}", setting,
             f"{kb(small)['images']} images, {kb(small)['pair']['pair_images']} of them with a person and a forklift."),
        Fact("v4.old_kb", [dec(kb(old)["forklift"]["ap50"]), dec(kb(old)["person"]["ap50"]), dec(kb(old)["pair"]["recall"])],
             f"{show(kb(old)['forklift']['ap50'], 3)} / {show(kb(old)['person']['ap50'], 3)} / {show(kb(old)['pair']['recall'], 2)} "
             f"forklift AP50 / person AP50 / pair recall of the Stage I model on the same images", source, old_setting),
    ]
    ss_note = "NVIDIA SmartSpaces Warehouse_010/018/019/022: a re-used test (3rd use), not fresh evidence."
    for prefix, model, label, model_setting in (("v4.", small, "v4-S", setting), ("v4.old_", old, "the Stage I model", old_setting),
                                                ("v4.v3_", v3, "v3, not shipped", "v3 YOLOX-tiny (detector-v3 study), confidence 0.35")):
        ss = model["ss"]
        humanoids = sum(scene["fp_confusion"]["person"].get(name, 0) for scene in ss["per_scene"].values()
                        for name in ("FourierGR1T2", "AgilityDigit"))
        facts += [
            Fact(f"{prefix}ss_found", [dec(ss["found_pooled"])], f"{show(ss['found_pooled'], 3)} forklift found rate per camera, {label} "
                 "(SmartSpaces, re-used test, 3rd use)", source, model_setting, ss_note),
            Fact(f"{prefix}ss_person", [dec(ss["person"]["P"]), dec(ss["person"]["R"])],
                 f"{show(ss['person']['P'], 3)} / {show(ss['person']['R'], 3)} person precision / recall, {label} (SmartSpaces, re-used test, 3rd use)",
                 source, model_setting),
            Fact(f"{prefix}ss_alerts", [Decimal(ss["alerts"]["strict"]), Decimal(ss["alerts"]["false_alerts"])],
                 f"{ss['alerts']['strict']} / {ss['alerts']['false_alerts']} strict / false alerts, {label} "
                 f"(of {ss['alerts']['events']} annotated events; SmartSpaces, re-used test, 3rd use)", source, model_setting),
            Fact(f"{prefix}ss_humanoid_fp", [Decimal(humanoids)], f"{humanoids} false person boxes on humanoid robots, {label} "
                 "(SmartSpaces, re-used test, 3rd use)", source, model_setting, "FourierGR1T2 and AgilityDigit robots; secondary measure."),
        ]
    runs = [load(root, f"{base}/out/{name}.json")["models"] for name in ("cpu_latency", "cpu_latency_run2")]
    ms = lambda name: [dec(run[name]["session_run_ms"]["mean"]) for run in runs]
    cpu_source, cpu_setting = f"`{base}/out/cpu_latency.json`, `{base}/out/cpu_latency_run2.json`", "ONNX Runtime, 4 threads, 300 runs on 30 validation frames per pass"
    facts += [
        Fact("v4.cpu_ms", ms("v4-S"), f"{show(ms('v4-S')[0], 1)} / {show(ms('v4-S')[1], 1)} ms per frame for v4-S (session.run mean, two passes)",
             cpu_source, cpu_setting, "The second pass ran the models in reverse order."),
        Fact("v4.old_cpu_ms", ms("old"), f"{show(ms('old')[0], 1)} / {show(ms('old')[1], 1)} ms per frame for the Stage I YOLOX-tiny (same passes)",
             cpu_source, cpu_setting),
    ]
    check = load(root, f"{base}/app_check.json")["vision_worker"]["runs"]
    for key, name, label in (("v4.app_wall_s", "vision-compact-v4.json", "detector v4"), ("v4.app_stage1_wall_s", "vision-compact.json", "Stage I detector")):
        wall = median(r["wall_s"] for r in check if r["settings_file"] == f"config/{name}")
        inference = median(r["inference_ms_per_frame"] for r in check if r["settings_file"] == f"config/{name}")
        count = sum(1 for r in check if r["settings_file"] == f"config/{name}")
        facts.append(Fact(key, [wall, inference], f"{show(wall, 2)} s end to end and {show(inference, 1)} ms per frame of inference "
                          f"for a 10 s clip ({label}, median of {count} runs)", f"`{base}/app_check.json` (three bundled samples, two passes)", f"`config/{name}`, 1 frame/s",
                          "Through scripts/vision_worker.py and the frozen compact_vision.py; includes model load."))
    passed = {arm: sum(1 for key, value in criteria[arm].items() if key != "all_pass" and value["pass"]) for arm in ("v4-S", "v4-T")}
    totals = {arm: sum(1 for key in criteria[arm] if key != "all_pass") for arm in ("v4-S", "v4-T")}
    if criteria["v4-S"]["all_pass"] is not True or criteria["v4-T"]["all_pass"] is not False:
        raise FactError(f"{base}/out/summary.json: unexpected verdicts")
    facts.append(Fact("v4.criteria", [Decimal(passed["v4-S"]), Decimal(totals["v4-S"]), Decimal(passed["v4-T"]), Decimal(totals["v4-T"])],
                      f"{passed['v4-S']} of {totals['v4-S']} pre-registered criteria passed by v4-S, {passed['v4-T']} of {totals['v4-T']} by v4-T",
                      source, f"`stage2/detector-v4/PREREG.json`, registered {load(root, f'{base}/PREREG.json')['written_utc']}",
                      "v4-T fails C2a (real-site forklift recall); only v4-S is adopted."))
    return facts


def visual_check(root: Path) -> list:
    results = load(root, f"{EVAL}/look/results.json")
    summary = load(root, f"{EVAL}/look/summary.json")
    dev = load(root, f"{EVAL}/look/dev.json")
    items = results["items"]
    confirmed = lambda answer: isinstance(answer, dict) and answer.get("forklift") == "yes" and answer.get("person") == "yes"
    flagged = lambda answer: isinstance(answer, dict) and "no" in (answer.get("forklift"), answer.get("person"))
    tp = [i for i in items if i["gold_status"] == "tp"]
    fp = [i for i in items if i["gold_status"] == "fp"]
    agree = [i for i in items if isinstance(i.get("model_answer"), dict)
             and all(i["model_answer"].get(k) == i["visual_label"][k] for k in ("forklift", "person"))]
    both_yes = [i for i in items if all(i["visual_label"][k] == "yes" for k in ("forklift", "person"))]
    audit = summary["e_description_audit"]["clear_factual_errors"]
    shelves = [i for i in dev["items"] if i["visual_label"]["forklift"] == "no"]
    forklifts = [i for i in dev["items"] if i["visual_label"]["forklift"] == "yes"]
    v3_missed = [i for i in forklifts if not confirmed(i.get("look_v3"))]
    when = {k: datetime.strptime(summary[k] if k != "test_run_utc" else summary[k]["start"], "%Y-%m-%dT%H:%M:%SZ")
            for k in ("frozen_utc", "labels_recorded_utc", "test_run_utc")}
    source = f"`{EVAL}/look/results.json`"
    setting = (f"`look/FREEZE.json`: prompt {results['prompt_version']} `{results['prompt_sha256'][:12]}`, Qwen2-VL-2B Q4_K_M "
               f"`{results['model_sha256'][:12]}`, temperature {show(results['temperature'])}, seed {results['seed']}")
    cohorts = {c: sum(1 for i in items if i["cohort"] == c) for c in ("primary", "routine")}
    seconds = [i["seconds"] for i in items]
    threads = int(results["threads"])
    return [
        Fact("look.crops", [Decimal(len(items))], f"{len(items)} test crops", source, setting,
             f"{cohorts['primary']} primary and {cohorts['routine']} routine; every predicted pair of the three profiles, pairs with IoU ≥ 0.9 share a crop."),
        Fact("look.tp_confirmed", [Decimal(sum(confirmed(i["model_answer"]) for i in tp)), Decimal(len(tp))],
             f"{sum(confirmed(i['model_answer']) for i in tp)} of {len(tp)} crops of annotation-matched pairs confirmed", source, setting,
             "Confirmed = forklift yes and person yes. The pairs are matched against the simulator annotations, not the visual labels."),
        Fact("look.fp_flagged", [Decimal(sum(flagged(i["model_answer"]) for i in fp)), Decimal(len(fp))],
             f"{sum(flagged(i['model_answer']) for i in fp)} of {len(fp)} crops of unmatched (gold-FP) pairs flagged", source, setting,
             "All gold-FP crops show a real person and forklift; they are box-localization misses."),
        Fact("look.agreement_both", [Decimal(len(agree)), Decimal(len(items))],
             f"{len(agree)} of {len(items)} crops agree with the visual labels on both fields", f"{source}, `{EVAL}/look/test_labels.json`", setting,
             "Visual labels: one labeling pass over contact sheets, no independent second labeler."),
        Fact("look.always_yes_both", [Decimal(len(both_yes)), Decimal(len(items))],
             f"{len(both_yes)} of {len(items)} crops would agree on both fields if every answer were forklift yes and person yes",
             f"`{EVAL}/look/test_labels.json`", "Baseline, no model: the visual labels alone",
             f"Only {len(items) - len(both_yes)} test crops lack a forklift or a person, so agreement on both fields cannot show "
             "that the check rejects false candidates."),
        Fact("look.median_s", [median(seconds), Decimal(threads)], f"{show(median(seconds))} s per crop (median) on {threads} threads", source, setting,
             f"Generation only, model loaded once ({show(results['load_seconds'])} s); range {show(min(seconds))}–{show(max(seconds))} s; shared workstation."),
        Fact("look.desc_errors", [Decimal(audit["numerator"]), Decimal(audit["denominator"])],
             f"{audit['numerator']} of {audit['denominator']} sampled descriptions with a clear factual error", f"`{EVAL}/look/summary.json`",
             setting, summary["e_description_audit"]["sampling"]),
        Fact("look.labels_after_freeze_s", [Decimal(int((when["labels_recorded_utc"] - when["frozen_utc"]).total_seconds()))],
             f"{int((when['labels_recorded_utc'] - when['frozen_utc']).total_seconds())} s from the prompt freeze to the recorded test labels",
             f"`{EVAL}/look/summary.json`", f"Freeze {summary['frozen_utc']}, labels {summary['labels_recorded_utc']}",
             f"The test run started {int((when['test_run_utc'] - when['labels_recorded_utc']).total_seconds())} s after the labels were recorded."),
        Fact("look.dev_shelf_flagged", [Decimal(sum(flagged(i.get("look_v2")) for i in shelves)), Decimal(len(shelves))],
             f"{sum(flagged(i.get('look_v2')) for i in shelves)} of {len(shelves)} development shelf/rack crops flagged by look-v2",
             f"`{EVAL}/look/dev.json`", "Development crops only (already used for calibration); prompt look-v2",
             "Unreadable answers count as not flagged."),
        Fact("look.dev_v3_unconfirmed", [Decimal(len(v3_missed)), Decimal(len(forklifts))],
             f"{len(v3_missed)} of {len(forklifts)} development crops with a real forklift not confirmed by look-v3",
             f"`{EVAL}/look/dev.json`", "Development crops only; the rejected prompt look-v3",
             f"{sum(1 for i in v3_missed if (i.get('look_v3') or {}).get('forklift') == 'no')} answered forklift no, "
             f"{sum(1 for i in v3_missed if (i.get('look_v3') or {}).get('person') == 'no')} person no, "
             f"{sum(1 for i in v3_missed if 'error' in (i.get('look_v3') or {}))} unreadable."),
    ]


def frame_timing(root: Path) -> list:
    """The 2026-11-06 erratum: which frame the published crops came from, and the Stage I check re-run on frame-exact crops."""
    base = f"{EVAL}/stage2/frame-timing"
    crops = load(root, f"{base}/stage1_look_crops.json")
    rerun = load(root, f"{base}/stage1_look_rerun.json")
    jobs = load(root, f"{base}/sample_jobs.json")
    published = {i["id"]: i for i in load(root, f"{EVAL}/look/results.json")["items"]}
    if {i["id"] for i in crops["items"]} != set(published) or {i["id"] for i in rerun["items"]} != set(published) \
            or any(i["published_sha256"] != published[i["id"]]["crop_sha256"] for i in crops["items"]):
        raise FactError(f"{base} does not cover the published look test crops")
    identical = sum(1 for i in crops["items"] if i["identical"])
    same = sum(1 for i in rerun["items"] if i["same_answer"])
    fast = next(run for run in jobs["runs"] if run["sample_fps"] == 5)
    late = sum(1 for row in fast["rows"] if row["old_offset"] != 0)
    exact = sum(1 for row in fast["rows"] if row["new_offset"] == 0)
    note = "Frame chosen by number (media.frame_at) against the frozen look.crop_frame / server.frame_image -ss seek; see docs/ERRATA.md."
    return [
        Fact("frame.stage1_identical", [Decimal(identical), Decimal(len(crops["items"]))],
             f"{identical} of {len(crops['items'])} Stage I look test crops identical to the frame-exact crop", f"`{base}/stage1_look_crops.json`",
             "`look/FREEZE.json` crop hashes", "Every test frame was a multiple of 30, a real keyframe of these clips. " + note),
        Fact("frame.stage1_rerun_same", [Decimal(same), Decimal(len(rerun["items"]))],
             f"{same} of {len(rerun['items'])} answers unchanged when look-v2 is re-run on the frame-exact crops", f"`{base}/stage1_look_rerun.json`",
             f"`look/FREEZE.json` model, prompt and decoding; {rerun['meta']['threads']} threads instead of {rerun['meta']['threads_published_run']}",
             "Resident CPU worker (scripts/look_worker.py)."),
        Fact("frame.sample_5fps_late", [Decimal(late), Decimal(len(fast["rows"])), Decimal(exact)],
             f"{late} of {len(fast['rows'])} evidence frames and crops were late before the fix, {exact} show the detector's frame now "
             f"(a job on the warehouse-1 sample at 5 frames/s)", f"`{base}/sample_jobs.json`", "Fast search settings at sample_fps 5 (a sensitivity setting)",
             "At the shipped 1 frame/s every sampled frame of these clips is a keyframe. " + note),
    ]


def drafting(root: Path) -> list:
    audit = load(root, f"{EVAL}/generation/independent-audit.json")
    source = f"`{EVAL}/generation/independent-audit.json`"
    setting = "`generation/FREEZE.json`: Qwen2.5-1.5B Q4_K_M, the earlier text-only prompt (replaced by draft-v2 on 2026-10-20)"
    pair = lambda a, b: [Decimal(a), Decimal(b)]
    return [
        Fact("gen.runs_drafts", pair(audit["runs"], audit["candidate_drafts"]), f"{audit['runs']} / {audit['candidate_drafts']} selected runs / drafts", source, setting),
        Fact("gen.checks_passed", pair(audit["automatic_checks_passed"], audit["candidate_drafts"]),
             f"{audit['automatic_checks_passed']} / {audit['candidate_drafts']} raw drafts passing the original automatic checks", source, setting),
        Fact("gen.unsupported", pair(audit["review_factual_addition_failures"], audit["candidate_drafts"]),
             f"{audit['review_factual_addition_failures']} / {audit['candidate_drafts']} drafts with an unsupported factual addition", source, setting,
             "Single rubric-based review against the supplied observations."),
        Fact("gen.observations", pair(audit["supplied_observations_covered"], audit["supplied_observations_total"]),
             f"{audit['supplied_observations_covered']} / {audit['supplied_observations_total']} supplied observations preserved", source, setting),
        Fact("gen.median_s", [dec(audit["generation_total_seconds_median"])], f"{show(audit['generation_total_seconds_median'])} s per draft, end to end (median)",
             source, setting, "Per-call process including model load, under concurrent workload."),
    ]


def baseline(root: Path) -> list:
    results = load(root, f"{EVAL}/results.json")
    spread = load(root, f"{EVAL}/uncertainty.json")
    frozen = load(root, f"{EVAL}/freeze.json")
    setting = (f"`freeze.json`: Grounding DINO tiny `{frozen['settings']['model_revision'][:8]}`, confidence {show(frozen['settings']['confidence'])}, "
               f"{show(frozen['settings']['sample_fps'])} frame/s; {frozen['heldout']} runs frozen before inference")
    return pair_facts("v1", results, f"`{EVAL}/results.json`", setting) + [
        Fact("v1.wall_s", [dec(spread["median_video_wall_s"])], f"{show(spread['median_video_wall_s'])} s per 10 s clip (median of 24)",
             f"`{EVAL}/uncertainty.json`", setting, "Includes model loading; shared workstation.")]


def device(root: Path, configs: dict) -> list:
    """The UGen300 records: benchmark, HEF accuracy, phase timings of a clip, the second-process guard."""
    base = f"{EVAL}/stage2/device"
    label = "UGen300 over USB at 0.9 A (reduced-performance mode)"
    bench = {name: load(root, f"{base}/bench/bench-{name}.json") for name in ("o0", "o2s", "o2s-phases")}
    clip, held = (load(root, f"{base}/{name}") for name in ("clip/clip-summary.json", "held-guard.json"))
    for name, record in (("clip", clip), ("held-guard", held)):
        if record.get("conditions") != label:
            raise FactError(f"{base}: the {name} record is not labelled {label!r}")
    if any(b["hailort_version"] != "5.3.2" for b in bench.values()):
        raise FactError(f"{base}/bench: a benchmark ran on another HailoRT release")
    o2s_sha = bench["o2s"]["hefs"]["detector"]["sha256"]
    if clip["hefs"]["detector"]["sha256"] != o2s_sha or load(root, f"{base}/hef-o2s/device_run.json")["hef_sha256"] != o2s_sha:
        raise FactError(f"{base}: the clip run and the HEF records name different detector HEFs")
    setting = f"{label}; HailoRT 5.3.2; detector HEF o2s `{o2s_sha[:12]}`"
    if bench["o2s-phases"]["hefs"]["detector"]["sha256"] != o2s_sha:
        raise FactError(f"{base}/bench/bench-o2s-phases.json: not HEF o2s")
    bench_source = f"`{base}/bench/` (three runs: `bench-o0.json`, `bench-o2s.json`, `bench-o2s-phases.json`)"
    bench_setting = f"{label}; HailoRT 5.3.2; HEFs o0 and o2s, confidence {show(bench['o2s']['settings']['confidence'])}"
    span = lambda values: [min(values), max(values)]
    fps = lambda batch: span([dec(run["fps"]) for b in bench.values() for run in b["detector"] if run["batch_size"] == batch])
    load_s = span([dec(b["load_s"]["vlm"]) for b in bench.values()])
    checks = [b["vlm"]["visual_check"] for b in bench.values()]
    speed = span([dec(c["ttft_s"]) for c in checks]) + span([dec(c["tokens_per_s"]) for c in checks])
    same = span([dec(b["concurrent"]["same_process"]["detector"]["fps"]) for b in bench.values()])
    if bench["o2s-phases"]["concurrent"]["second_process"].get("device_held_by_this_process") is not True:
        raise FactError(f"{base}/bench/bench-o2s-phases.json: the second-process run is not the expected refusal")
    stage1 = load(root, f"{base}/hef-o2s/stage1_device_eval.json")
    scored = load(root, f"{EVAL}/compact/v1/evaluation_fps1.json")
    if (stage1["n_scored_runs"], stage1["n_scored_frames"]) != (scored["n_scored_runs"], scored["n_scored_frames"]):
        raise FactError(f"{base}/hef-o2s/stage1_device_eval.json: not the scorable Stage I runs and frames of the ONNX scores")
    snr = min(dec(output["snr_db"]) for output in load(root, f"{base}/hef-o2s/acceptance.json")["outputs"])
    phases = clip["schedule"]["phases"]
    clip_values = [dec(phases[name]["seconds"]) for name in ("detection", "visual_check", "draft")] + [dec(clip["schedule"]["total_s"])]
    waits = [dec(v["max_s"]) for k, v in clip["poll_latency"].items() if k in ("visual_check", "draft")]
    second = held["second_process"]
    clip_source = f"`{base}/clip/clip-summary.json`"
    facts = [
        Fact("device.detector_fps_b1", fps(1), f"{show(fps(1)[0], 1)}–{show(fps(1)[1], 1)} FPS at batch 1 through TraceBrief's detector "
             "(letterboxed frames in, source-pixel boxes out, HailoRT's host NMS included)", bench_source, bench_setting,
             "`hailortcli run2` gave about 115 FPS for the same HEFs in our build check; batch 1 here is one frame per call."),
        Fact("device.detector_fps_b8", fps(8), f"{show(fps(8)[0], 1)}–{show(fps(8)[1], 1)} FPS at batch 8 (the server's default)",
             bench_source, bench_setting),
        Fact("device.vlm_load_s", load_s, f"{show(load_s[0], 1)}–{show(load_s[1], 1)} s to load Qwen2-VL-2B (3.26 GB HEF)", bench_source,
             bench_setting, "Hashing the HEF adds about 7 s before it."),
        Fact("device.vlm_speed", speed, f"{show(speed[0], 2)}–{show(speed[1], 2)} s to first token and {show(speed[2], 2)}–"
             f"{show(speed[3], 2)} tokens/s for one visual check", bench_source, bench_setting,
             "Hailo publishes 1.02 s and 7.04 tokens/s for this HEF on Hailo-10H."),
        Fact("device.same_process_fps", same, f"{show(same[0], 2)}–{show(same[1], 2)} FPS for the detector while the VLM generates "
             "in the same process", bench_source, bench_setting,
             "The genai binding holds the GIL while it reads tokens; the VLM kept its speed. A second process cannot open the device "
             "(`bench-o2s-phases.json`)."),
        *pair_facts("device", stage1, f"`{base}/hef-o2s/stage1_device_eval.json`", setting),
        Fact("device.snr_min", [snr], f"{show(snr, 1)} dB lowest per-head SNR, full precision against INT8, on the 64 acceptance frames",
             f"`{base}/hef-o2s/acceptance.json`", "DFC 5.3.0, optimization level 2, 1,024 calibration frames (seed 20261105)",
             "The build script's acceptance gate is 10 dB on every head."),
        Fact("device.clip_phases", clip_values, f"{show(clip_values[0], 1)} s detection, {show(clip_values[1], 1)} s visual check and "
             f"{show(clip_values[2], 1)} s draft, {show(clip_values[3], 1)} s in all, for `{clip['sample']}` "
             f"({phases['visual_check']['candidates']} candidate)", clip_source, setting,
             "Detection includes the host's evidence frames; the models were loaded before the job."),
        Fact("device.clip_switches", [Decimal(clip["model_switches_in_job"])],
             f"{clip['model_switches_in_job']} switch between the detector and the VLM in that job", clip_source, setting),
        Fact("device.poll_wait_s", [max(waits)], f"{show(max(waits), 1)} s longest wait for a job-status answer while the VLM generated "
             "(server in the same process)", clip_source, setting, "Answers took a few milliseconds during detection."),
        Fact("device.held_s", [dec(second["runtime"]["seconds"])], f"{show(second['runtime']['seconds'], 2)} s for a second process to "
             "be refused the device, with the reason", f"`{base}/held-guard.json`", label,
             "HailoRT answers HAILO_LIBUSB_FAILURE (LIBUSB_ERROR_ACCESS) on a Windows host; no HEF is hashed first."),
    ]
    return facts


def quantile(values, q) -> Decimal:
    """Linear interpolation between the closest ranks (numpy's default), as look-agreement/analyze.py computes it."""
    ordered = sorted(dec(v) for v in values)
    position = (len(ordered) - 1) * Decimal(str(q))
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def device_look(root: Path) -> list:
    """The device's look-v2 answers on the 148 Stage I crops against the published CPU answers, and its drafts beside the CPU's."""
    base = f"{EVAL}/stage2/device/look-agreement"
    label = "UGen300 over USB at 0.9 A (reduced-performance mode)"
    meta = load(root, f"{base}/device_meta.json")
    if meta["conditions"] != label or meta["hailort_version"] != "5.3.2":
        raise FactError(f"{base}/device_meta.json: not labelled {label!r} on HailoRT 5.3.2")
    published = {i["id"]: i for i in load(root, f"{EVAL}/look/results.json")["items"]}
    rows = [json.loads(line) for line in (root / base / "device_answers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    device = {r["id"]: r for r in rows}
    if len(rows) != len(device) or set(device) != set(published) \
            or any(device[k]["crop_sha256"] != published[k]["crop_sha256"] for k in published):
        raise FactError(f"{base}/device_answers.jsonl does not answer each published look test crop once")
    verdict = lambda answer: ("unavailable" if not isinstance(answer, dict) else
                              "confirmed" if answer.get("forklift") == "yes" and answer.get("person") == "yes" else "check-suggested")
    answer = lambda key: device[key].get("answer")
    same = sum(verdict(published[k]["model_answer"]) == verdict(answer(k)) for k in published)
    tp = [k for k in published if published[k]["gold_status"] == "tp"]
    fp = [k for k in published if published[k]["gold_status"] == "fp"]
    tp_confirmed = sum(verdict(answer(k)) == "confirmed" for k in tp)
    fp_flagged = sum(verdict(answer(k)) == "check-suggested" for k in fp)
    labels = sum(isinstance(answer(k), dict) and all(answer(k)[f] == published[k]["visual_label"][f] for f in ("forklift", "person"))
                 for k in published)
    unreadable = sum(not isinstance(answer(k), dict) for k in published)
    # The same crops again after HailoVLM stopped reading at the end of the JSON answer: every answer must be unchanged.
    fixed = {r["id"]: r for r in (json.loads(line) for line in
                                  (root / base / "device_answers_fixed.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    if set(fixed) != set(device) or any(fixed[k].get("answer") != answer(k) or fixed[k]["crop_sha256"] != device[k]["crop_sha256"]
                                        for k in device):
        raise FactError(f"{base}/device_answers_fixed.jsonl: an answer differs from device_answers.jsonl")
    seconds, before = [r["seconds"] for r in fixed.values()], [r["seconds"] for r in rows]
    vlm_sha = meta["hefs"]["vlm"]["sha256"]
    setting = f"{label}; HailoRT 5.3.2; Qwen2-VL-2B HEF `{vlm_sha[:12]}`, look-v2 `{meta['prompt_sha256'][:12]}`, greedy, seed {meta['seed']}"
    source = f"`{base}/device_answers.jsonl` against `{EVAL}/look/results.json`"
    drafts = {name: load(root, f"{base}/{name}_drafts.json")["drafts"] for name in ("device", "cpu")}
    events = load(root, f"{base}/draft_events.json")["events"]
    if any([d["key"] for d in runs] != [e["key"] for e in events] for runs in drafts.values()):
        raise FactError(f"{base}: the device and CPU drafts are not of the same candidates")
    if load(root, f"{base}/device_drafts.json")["hefs"]["vlm"]["sha256"] != vlm_sha:
        raise FactError(f"{base}: the drafts and the visual checks ran on different VLM HEFs")
    accepted = {name: sum(d["audit"]["method"] in ("ugen300-qwen2-vl-draft", "local-qwen-draft") for d in runs) for name, runs in drafts.items()}
    after = load(root, f"{base}/device_drafts_fixed.json")["drafts"]
    if [d["audit"]["method"] for d in after] != [d["audit"]["method"] for d in drafts["device"]]:
        raise FactError(f"{base}/device_drafts_fixed.json: a draft was accepted or rejected differently after the stream fix")
    draft_s = median([d["audit"]["model_output"]["generation_seconds"] for d in after])
    return [
        Fact("device.look_same_verdict", [Decimal(same), Decimal(len(published))],
             f"{same} of {len(published)} Stage I test crops given the same verdict on the device as in the published CPU run",
             source, setting, f"Verdicts: confirmed, check suggested, or unreadable. Device answers unreadable: {unreadable}. "
             "The device runs Hailo's compiled Qwen2-VL-2B on a padded 336×336 frame, the CPU the Q4_K_M GGUF."),
        Fact("device.look_tp_confirmed", [Decimal(tp_confirmed), Decimal(len(tp))],
             f"{tp_confirmed} of {len(tp)} crops of annotation-matched pairs confirmed with the device answers", source, setting,
             "The Stage I measure (`look.tp_confirmed`) recomputed with the device's answers."),
        Fact("device.look_fp_flagged", [Decimal(fp_flagged), Decimal(len(fp))],
             f"{fp_flagged} of {len(fp)} crops of unmatched (gold-FP) pairs flagged with the device answers", source, setting,
             "All gold-FP crops show a real person and forklift; they are box-localization misses."),
        Fact("device.look_agreement_both", [Decimal(labels), Decimal(len(published))],
             f"{labels} of {len(published)} crops agree with the visual labels on both fields with the device answers",
             f"{source}, `{EVAL}/look/test_labels.json`", setting, "Visual labels: one labeling pass over contact sheets."),
        Fact("device.look_seconds", [median(seconds), quantile(seconds, "0.95"), median(before), quantile(before, "0.95")],
             f"{show(median(seconds), 1)} s per crop (median) and {show(quantile(seconds, '0.95'), 1)} s at the 95th percentile on the device; "
             f"{show(median(before), 1)} s and {show(quantile(before, '0.95'), 1)} s while it still read the reply to the token limit",
             f"`{base}/device_answers_fixed.jsonl`, `{base}/device_answers.jsonl`", setting,
             "Generation only, the VLM resident. Since 2026-11-06 HailoVLM leaves the stream at the end of the JSON answer; "
             "the same 148 answers. The published CPU run's median is `look.median_s`."),
        Fact("device.draft_accepted", [Decimal(accepted["device"]), Decimal(len(events)), Decimal(accepted["cpu"]), Decimal(len(events))],
             f"{accepted['device']} of {len(events)} device drafts passed the source checks, against {accepted['cpu']} of {len(events)} "
             f"CPU drafts of the same candidates; {show(draft_s, 1)} s per device draft (median)",
             f"`{base}/device_drafts.json`, `{base}/cpu_drafts.json` (time: `device_drafts_fixed.json`)",
             f"{label}; draft-v2 with the device format hint (draft-v2-device), greedy; "
             "CPU: Qwen2.5-1.5B Q4_K_M with the JSON grammar, 4 threads", "Candidates fixed in `draft_events.json` before any device draft."),
    ]


def device_llm(root: Path) -> list:
    """Drafting with the Qwen2.5-1.5B-Instruct HEF instead of the VLM in text-only mode (pre-registered)."""
    base = f"{EVAL}/stage2/device/draft-llm"
    label = "UGen300 over USB at 0.9 A (reduced-performance mode)"
    prereg = load(root, f"{base}/PREREG.json")
    runs = {name: load(root, f"{base}/{name}.json") for name in ("llm_drafts", "llm_switch", "llm_drafts_device1_selection",
                                                                  "llm_drafts_device2_selection")}
    for name, record in runs.items():
        if record["conditions"] != label or record["hailort_version"] != "5.3.2" or record["hefs"]["llm"]["sha256"] != prereg["model"]["sha256"]:
            raise FactError(f"{base}/{name}.json: not the pre-registered LLM HEF at {label!r} on HailoRT 5.3.2")
    record = runs["llm_drafts"]
    events = load(root, f"{EVAL}/stage2/device/look-agreement/draft_events.json")["events"]
    if [d["key"] for d in record["drafts"]] != [e["key"] for e in events] or record["format"] != "draft-v2-device":
        raise FactError(f"{base}/llm_drafts.json: not draft-v2-device on the 20 candidates of draft_events.json")
    split = {e["key"]: e["split"] for e in events}
    passed = [d["key"] for d in record["drafts"] if d["audit"]["method"] == "ugen300-qwen2.5-draft"]
    held = sum(split[key] == "held-out" for key in passed)
    seconds = median([d["audit"]["seconds"] for d in record["drafts"]])
    speed = median([d["audit"]["model_output"]["tokens_per_s"] for d in record["drafts"]])
    hints = {name: sum(d["audit"]["method"] == "ugen300-qwen2.5-draft" for d in runs[f"llm_drafts_{name}_selection"]["drafts"])
             for name in ("device1", "device2")}
    cycles = [c for r in runs.values() for c in r["cycles"]]
    to_llm = median([c["vlm_release_s"] + c["llm_load_s"] for c in cycles if "llm_load_s" in c])
    # Swaps back: the VLM loaded again after the LLM was released, in the driver runs and the probes that had an LLM.
    probes = {name: load(root, f"{base}/swapback_{name}.json") for name in ("no_draft", "no_detector", "settle10", "no_llm")}
    after_llm = [p for p in probes.values() if p["variant"].get("llm", True)]
    back = [("vlm_load_s" in c) for c in cycles if "llm_release_s" in c] + [p["swap_back"] == "ok" for p in after_llm]
    failed = back.count(False)
    if probes["no_llm"]["swap_back"] != "ok":
        raise FactError(f"{base}/swapback_no_llm.json: the VLM did not load again without the LLM either")
    setting = (f"{label}; HailoRT 5.3.2; Qwen2.5-1.5B-Instruct HEF `{prereg['model']['sha256'][:12]}`, draft-v2-device, greedy, seed 19, "
               "frequency_penalty 1.0, 320 tokens")
    source = f"`{base}/llm_drafts.json`"
    return [
        Fact("device.llm_draft_accepted", [Decimal(len(passed)), Decimal(len(events)), Decimal(held), Decimal(sum(v == "held-out" for v in split.values()))],
             f"{len(passed)} of {len(events)} drafts passed the source checks ({held} of {sum(v == 'held-out' for v in split.values())} "
             "held out), drafted by the Qwen2.5-1.5B HEF", source, setting,
             f"Rule fixed in `PREREG.json` before the first draft: adopt only with at least 8 of 10 held out (the VLM's). The two stricter "
             f"hints passed {hints['device1']} and {hints['device2']} of 10 on the selection half."),
        Fact("device.llm_draft_seconds", [seconds, speed], f"{show(seconds, 1)} s per draft (median) at {show(speed, 2)} tokens/s",
             source, setting, "The VLM in text-only mode: 41.9 s (`device.draft_accepted`)."),
        Fact("device.llm_swap", [to_llm, Decimal(failed), Decimal(len(back))],
             f"{show(to_llm, 1)} s to release the VLM and load the LLM (median); the VLM failed to load again after the LLM in {failed} "
             f"of {len(back)} swaps back", f"`{base}/llm_*.json`, `{base}/swapback_*.json`", setting,
             "Each failure: HAILO_COMMUNICATION_CLOSED while the VLM HEF was sent, then LIBUSB_ERROR_IO, and the device re-enumerated; "
             "with or without the detector, a draft or a 10 s pause. Released and loaded again with no LLM between, the VLM loaded."),
    ]


def settings(root: Path, configs: dict, tests: int) -> list:
    rates = {name: dec(cfg["sample_fps"]) for name, cfg in configs.items()}
    threads = {name: int(cfg["threads"]) for name, cfg in configs.items()}
    if len(set(rates.values())) != 1 or len(set(threads.values())) != 1:
        raise FactError(f"The search profiles differ in sample_fps or threads: {rates} {threads}")
    names = ", ".join(f"`config/{name}`" for name in sorted(configs))
    manifest = load(root, "config/hailo-manifest.json")
    models = manifest.get("models") or {}
    recorded = sum(1 for entry in models.values() if entry.get("sha256"))
    return [
        Fact("config.sample_fps", [next(iter(rates.values()))], f"{show(next(iter(rates.values())))} frame per second in every search profile", names,
             "Matches every freeze record (see below)", "5 frames/s appears only as the Fast sensitivity run and as a UGen300 sizing scenario."),
        Fact("config.threads", [Decimal(next(iter(threads.values())))], f"{next(iter(threads.values()))} CPU threads for the detectors", names, "As frozen"),
        Fact("hailo.hailort", [dec(manifest["hailort"])], f"HailoRT {manifest['hailort']} (any {manifest['hailort']}.x is accepted)",
             "`config/hailo-manifest.json`", "The release ASUS ships for the UGen300",
             "The device code's HailoRT calls were checked against the 5.3, 5.4 and 5.5 Python API sources and ran on a UGen300 "
             "with HailoRT 5.3.2 (UGen300 section)."),
        Fact("hailo.hef_hashes", [Decimal(recorded), Decimal(len(models))], f"{recorded} of {len(models)} HEF SHA-256 values recorded",
             "`config/hailo-manifest.json`", "Device mode refuses a HEF without a recorded hash",
             "The detector HEF is built from calibration frames outside the repository, so this manifest records no hash; "
             "the device runs used a local manifest (`data/evaluation/stage2/device/README.md`)."),
        Fact("tests.collected", [Decimal(tests)], f"{tests} test cases", "`python -m pytest --collect-only -q tests`", "Current `tests/`",
             "Collected count; tests that need a UGen300, the real models (`TRACEBRIEF_TEST_MODELS=1`), a real browser "
             "(`TRACEBRIEF_TEST_BROWSER=1`), Python 3.11 or a recorded device buffer skip themselves."),
    ]


def prompt_sha256(root: Path) -> str:
    """SHA-256 of look_prompt.PROMPT, read as source so nothing from src/ is imported."""
    tree = ast.parse((root / "src" / "tracebrief" / "look_prompt.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROMPT" for t in node.targets):
            return hashlib.sha256(ast.literal_eval(node.value).encode()).hexdigest()
    raise FactError("src/tracebrief/look_prompt.py defines no PROMPT")


def freeze_checks(root: Path, configs: dict) -> list:
    same = lambda frozen, current, skip=(): all(current.get(k) == v for k, v in frozen.items() if k not in skip) and \
        all(k in frozen for k in current if k not in skip)
    compact = load(root, f"{EVAL}/compact/v1/FREEZE.json")
    v4 = load(root, f"{EVAL}/stage2/detector-v4/freeze/v4-S/FREEZE.json")
    v4_code = load(root, f"{EVAL}/stage2/detector-v4/PREREG.json")["external_code_sha256"]
    look = load(root, f"{EVAL}/look/FREEZE.json")
    generation = load(root, f"{EVAL}/generation/FREEZE.json")["code_sha256"]
    return [
        FreezeCheck("`compact/v1/FREEZE.json`", "`config/vision-compact.json` (all settings but the model path) and `src/tracebrief/compact_vision.py`",
                    same(compact["settings"], configs["vision-compact.json"], ("model_path",))
                    and compact["compact_vision_sha256"] == text_sha256(root / "src/tracebrief/compact_vision.py")),
        FreezeCheck("`stage2/detector-v4/freeze/v4-S/FREEZE.json`",
                    "`config/vision-compact-v4.json` (all settings but the model path), `weights/yolox-s-warehouse-v4.onnx` and "
                    "`src/tracebrief/compact_vision.py` (as registered in `stage2/detector-v4/PREREG.json`)",
                    same(v4["app_settings"], configs["vision-compact-v4.json"], ("model_path",))
                    and hashlib.sha256((root / configs["vision-compact-v4.json"]["model_path"]).read_bytes()).hexdigest() == v4["onnx_sha256"]
                    and v4_code["tracebrief/src/tracebrief/compact_vision.py"] == text_sha256(root / "src/tracebrief/compact_vision.py")),
        FreezeCheck("`freeze.json` and `phase2/comparator_v1/freeze.json`", "`config/vision-broader.json` and `config/vision.json`",
                    all(same(load(root, f"{EVAL}/{record}")["settings"], configs[name])
                        for record in ("freeze.json", "phase2/comparator_v1/freeze.json") for name in ("vision-broader.json", "vision.json"))),
        FreezeCheck("`phase2/freeze.json`", "`config/vision-stricter.json`",
                    same(load(root, f"{EVAL}/phase2/freeze.json")["settings"], configs["vision-stricter.json"])),
        FreezeCheck("`look/FREEZE.json`", "`scripts/look_local.py`, `src/tracebrief/look.py` and the prompt in `src/tracebrief/look_prompt.py`",
                    all(text_sha256(root / path) == digest for path, digest in look["files"].items())
                    and look["prompt_sha256"] == prompt_sha256(root)),
        FreezeCheck("`generation/FREEZE.json`", "`scripts/generate_local.py` and `src/tracebrief/generation.py`",
                    all(text_sha256(root / path.split("/", 1)[1]) == generation[path]
                        for path in ("tracebrief/scripts/generate_local.py", "tracebrief/src/tracebrief/generation.py")),
                    required=False, note="Superseded on purpose: draft-v2 (2026-10-20) and the later source checks; "
                                         "GENERATION_RESULTS.md reports the evaluated version as recorded."),
    ]


def collect(root: Path = ROOT, tests: int | None = None) -> Report:
    root = Path(root)
    configs = {path.name: load(root, f"config/{path.name}") for path in sorted((root / "config").glob("vision*.json"))}
    detection, timing = detector(root, configs)
    report = Report()
    report.sections = [
        ("Detector on the held-out test runs (Stage I models)", detection),
        ("Detector time on the CPU (Stage I models)", timing),
        ("Detector v4 (Stage II, pre-registered; Fast search since 2026-11-05)", detector_v4(root, configs)),
        ("Visual check (Qwen2-VL-2B, CPU)", visual_check(root)),
        ("Frame timing of evidence frames and crops (erratum, 2026-11-06)", frame_timing(root)),
        ("Stage I drafting evaluation (Qwen2.5-1.5B, CPU)", drafting(root)),
        ("Original baseline (v1, 24 runs)", baseline(root)),
        ("UGen300 (Stage II; USB at 0.9 A, reduced-performance mode)", device(root, configs)),
        ("UGen300 visual check and drafts against the CPU path (Stage II; USB at 0.9 A)", device_look(root)),
        ("UGen300 drafts with the Qwen2.5-1.5B HEF, pre-registered (Stage II; USB at 0.9 A)", device_llm(root)),
        ("Settings, device manifest and tests", settings(root, configs, count_tests(root) if tests is None else tests)),
    ]
    report.checks = freeze_checks(root, configs)
    keys = [fact.key for _, facts in report.sections for fact in facts]
    if len(keys) != len(set(keys)):
        raise FactError("duplicate fact keys")
    for fact in report.facts().values():  # the text must open with the values, as a document quotes them
        printed = NUMBER.findall(fact.text)[:len(fact.values)]
        if len(printed) != len(fact.values) or not all(agrees(p, v) for p, v in zip(printed, fact.values)):
            raise FactError(f"{fact.key}: text {fact.text!r} does not start with its values {[str(v) for v in fact.values]}")
    return report


def cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render(report: Report) -> str:
    lines = [
        "# Facts",
        "",
        "Generated by `scripts/build_facts.py` from the records in this repository; do not edit by hand. Each value names the file "
        "it is computed from and the frozen setting it depends on (freeze records are named relative to `data/evaluation/`). "
        "README.md and the files in `docs/` mark a quoted value with an HTML comment, `<!-- fact: key -->`, just before the number, "
        "and `tests/test_facts.py` checks every marked number against this list, rounded to the decimals the document prints.",
        "",
        "Accuracy figures are on NVIDIA's synthetic warehouse footage, except in the detector v4 section, which also reports two "
        "real-image test sets (LOCO and keremberke) and NVIDIA SmartSpaces (a re-used test). Times are CPU measurements on our "
        "Intel Core i7-10700 workstation, which was shared with other work, except in the UGen300 section: those were measured "
        "on the device over USB with a source advertising 0.9 A, so HailoRT ran it in reduced-performance mode; re-measure on "
        "a 3 A USB-C port before quoting them as the device's speed.",
    ]
    for title, facts in report.sections:
        lines += ["", f"## {title}", "", "| Key | Value | Source | Frozen setting | Note |", "|---|---|---|---|---|"]
        lines += [f"| `{f.key}` | {cell(f.text)} | {cell(f.source)} | {cell(f.setting)} | {cell(f.note)} |" for f in facts]
    lines += ["", "## Frozen records and the current code", "",
              "Each freeze record is compared with the settings and files in use now (SHA-256 of the text with LF line endings).", "",
              "| Freeze record | Compared with | Matches | Note |", "|---|---|---|---|"]
    lines += [f"| {c.record} | {cell(c.compared)} | {'yes' if c.matches else 'no'} | {cell(c.note)} |" for c in report.checks]
    return "\n".join(lines) + "\n"


def documents(root: Path) -> list:
    return [root / "README.md"] + sorted(p for p in (root / "docs").glob("*.md") if p.name != Path(FACTS_MD).name)


def check_text(name: str, text: str, facts: dict) -> tuple:
    """Problems with the marked numbers in one document, and how many marks were checked."""
    problems, checked = [], 0
    for match in TAG.finditer(text):
        key, line = match.group(1), text.count("\n", 0, match.start()) + 1
        fact = facts.get(key)
        if fact is None:
            problems.append(f"{name}:{line}: unknown fact '{key}'; the keys are listed in {FACTS_MD}.")
            continue
        rest = text[match.end():]
        end = SPAN_END.search(rest)
        quoted = NUMBER.findall(rest[:end.start()] if end else rest)
        if not quoted:
            problems.append(f"{name}:{line}: the mark for '{key}' is not followed by a number before the next mark, cell border or line end.")
            continue
        pairs = list(zip(quoted, fact.values))
        if not all(agrees(got, want) for got, want in pairs):
            problems.append(f"{name}:{line}: '{key}' is quoted as {' '.join(q for q, _ in pairs)}, but {FACTS_MD} has {fact.text}. "
                            f"Correct the document, or rerun scripts/build_facts.py if the record changed.")
        checked += 1
    return problems, checked


def check_documents(root: Path, report: Report) -> tuple:
    problems, checked, facts = [], 0, report.facts()
    for path in documents(root):
        found, count = check_text(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8"), facts)
        problems += found
        checked += count
    return problems, checked


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recompute the headline numbers and write docs/FACTS.md.")
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if FACTS.md is stale or a marked number differs")
    args = parser.parse_args(argv)
    report = collect(ROOT)
    text = render(report)
    target = ROOT / FACTS_MD
    problems = []
    if args.check:
        current = target.read_text(encoding="utf-8").replace("\r\n", "\n") if target.is_file() else ""
        if current != text:
            problems.append(f"{FACTS_MD} is out of date; run python scripts/build_facts.py")
    else:
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print(f"Wrote {FACTS_MD}: {len(report.facts())} facts.")
    found, checked = check_documents(ROOT, report)
    problems += found
    problems += [f"Frozen setting changed: {c.record} no longer matches {c.compared}." for c in report.checks if c.required and not c.matches]
    print(f"Checked {checked} marked numbers in README.md and docs/.")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
