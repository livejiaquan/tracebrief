# Frozen generation comparison

Every candidate from the first six run IDs in the corrected detector freeze was evaluated: five drafts and one no-candidate run. `FREEZE.json` fixes selection and code hashes; `*.py.txt` are byte-identical source snapshots from that evaluation. The current application includes a later question-format guard, so it intentionally differs from the frozen code.

`results.json` retains model text, deterministic baseline, original accepted output and automatic checks. `DRAFT_REVIEW.md` reports the rubric-based review, including one unsupported declaration accepted by the original checks. `post-audit-guard-recheck.json` is a later recheck on the same five outputs, not a new independent benchmark. It rejects the observed declaration and uses the template.

This portable export removes machine-local image paths and runtime frame URLs. `LINEAGE.json` records original and portable hashes. Audit references to the original results hash concern the original pre-portability file. Source frames are the corresponding NVIDIA synthetic footage, not human workplace recordings. See the project source manifest and third-party notices for dataset attribution.

The generator and the template see only selected detector observations. Coverage of those observations does not establish complete event coverage, detector correctness, physical safety, or operator time savings. No benefit over the template was established.
