# Local drafting results

## Current drafting design (2026-10-20)

The review below led to a change in what the language model is allowed to write. Since 2026-10-20 the local Qwen2.5-1.5B writes only the title, one scene sentence and open review questions, using the visual check's description when one is available (prompt `draft-v2`). Box relations are no longer paraphrased: `src/tracebrief/generation.py` appends the detector timeline exactly as recorded, a JSON grammar restricts citations to real observation IDs, and the existing rule checks and labeled template fallback still apply. The five-draft review below describes the earlier text-only prompt and is kept as recorded.

The local Qwen2.5-1.5B Q4_K_M model ran successfully, but this small evaluation did **not** establish an advantage over the deterministic template. A rubric-based review found an unsupported statement in one of five drafts that had passed automatic checks. The failure and the later guard repair are reported separately.

## Frozen evaluation

Every candidate from the first six baseline v1 held-out detector runs was included. Five runs yielded one candidate each; a no-candidate run stayed in the denominator. The input contained only selected, detector-derived timestamped box observations and stated unknowns. Simulator annotations were not supplied to generation. Model, prompt and source code were frozen before inspecting these outputs.

| Measure | Result |
|---|---:|
| Selected runs / actual drafts | 6 / 5 |
| Raw drafts passing original automatic checks | 5 / 5 |
| Drafts with unsupported factual additions (rubric review) | 1 / 5 |
| Supplied timestamp/box-state observations preserved | 12 / 12 |
| Drafts with citation sets matching supplied observations | 5 / 5 |
| Summary words: deterministic template / model, total | 152 / 161 |
| End-to-end CPU generation median | 37.581 s |
| End-to-end CPU generation range | 24.948–61.420 s |

Timings include startup under concurrent workload. These are not isolated throughput measurements. Word counts exclude title/questions. Factuality was assessed in a single rubric-based review against the supplied observations; it was not a human user study and did not check the full source video.

The failed draft, from run prefix `018461e5b5cf7a482d3f`, placed the statements “Another camera view is available” and “Site procedure applies” in the question field. The source supplied neither fact. Four other drafts had no definite new event-fact assertion identified, although three used imprecise or redundant wording. This is not a claim that four drafts were verified as true in the world.

## Post-review guard repair

The validation now rejects review-question entries that do not end with `?`, triggering an explicit deterministic-template fallback. Rechecking the **same five saved raw outputs** rejects the observed failed draft and leaves the other four accepted. This is a regression check of a structural repair, not a new held-out result and not proof of semantic correctness. An actual question can still contain a false presupposition.

The original results and review remain intact. No prompt was retuned on these six runs. Human review remains necessary; the template remains a valid default when the model does not add useful information.

## Evidence-selection limitation

In run prefix `005bc639df8c4a2b36e0`, the full candidate predictions contain a non-overlap state at 5 s, but the selected generation observations include only overlapping boxes at 0 s and 6 s. Neither the template nor the model can describe that omitted change. The 12/12 coverage count concerns supplied observations only. A future selector should retain state transitions and be evaluated separately.

## Provenance

The frozen records (`FREEZE.json`, `results.json`, `DRAFT_REVIEW.md`, `independent-audit.json`, `post-audit-guard-recheck.json`) are in `data/evaluation/generation/`. The frozen results SHA-256 is `d68c33cc05a20013bc9dce0cad77718225db924a9e9f6fb0d082ea9ab574bbb6` (original file before the portability export; see `LINEAGE.json`). Exact model identity and license are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Portable raw results, frozen code, protocol, review and subsequent guard recheck are included in [the generation evaluation bundle](../data/evaluation/generation/README.md).
