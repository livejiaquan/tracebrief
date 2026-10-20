# Rubric-based review: local generation

Reviewed 2026-10-04 against the [rubric](AUDIT_PROTOCOL.md) recorded before the outputs were read. This is a single rubric-based content review, not a human user study. The detector inputs, generation prompt and model were not changed during this review.

## Finding

**Local generation ran, but added value over the deterministic template is not established. One of five accepted model drafts added unsupported claims in its review-question field.** The template preserved the supplied facts without that error and had fewer total summary words. Valid IDs and numeric checks did not catch the semantic error.

The six selected run IDs exactly match `FREEZE.json`; all their candidate events were included. Five runs produced one candidate each; the other run produced none and was retained without substitution. The archived `generate_local.py`, `generation.py` and `server.py` all match the SHA-256 values in the freeze. Raw output, baseline, accepted output and prompt hashes are in `results.json`; its SHA-256 is `d68c33cc05a20013bc9dce0cad77718225db924a9e9f6fb0d082ea9ab574bbb6`.

## Results

| Measure | Frozen result |
|---|---:|
| Selected runs / generated drafts | 6 / 5 |
| Drafts passing automatic checks | 5 / 5 |
| Drafts with an unsupported factual addition identified in review | 1 / 5 |
| Drafts whose citation sets exactly match supplied observation IDs | 5 / 5 |
| Supplied timestamp/box-state observations retained in summaries | 12 / 12 |
| Template summary words, total | 152 |
| Model summary words, total | 161 |
| Model end-to-end generation time, median | 37.581 s |
| Model end-to-end generation time, range | 24.948–61.420 s |

Word counts use whitespace splitting and exclude titles/questions. Time includes model startup in these CPU runs under concurrent work; it is not a standalone throughput benchmark. A covered supplied observation is not proof that the detector was correct or that the event record is complete. Five drafts are too few for a broad reliability estimate.

## Case-by-case review

| Run prefix | Supplied evidence | Assessment |
|---|---|---|
| `001e53453441935632ae` | Close/no overlap at 3 s; overlap at 5 s; close/no overlap at 9 s | All three states and order preserved. “Again” is supported by the first and last states. Added sentence about chronological presentation is redundant. 35 vs 43 summary words. |
| `005bc639df8c4a2b36e0` | Overlap at 0 s and 6 s | Both states preserved. “Potential proximity between the person and forklift” is hedged but less precise than image-plane box overlap; retain as ambiguity, not proven physical proximity. 27 vs 22 words. |
| `00e0991fbc61ad24102b` | Overlap at 0 s and 6 s | Same wording/output as previous case from identical observation text. Same ambiguity. 27 vs 22 words. These duplicate inputs reduce task diversity. |
| `014f5a0f9461dfae29d5` | Overlap at 4 s and 7 s | Both states preserved. “Determine if there is a physical distance” is awkward and may imply more can be measured from visual review than is established; “supervisor is preparing an incident brief” restates supplied purpose, not an observed event. 38 vs 22 words. |
| `016c1396912bfc432997` | No candidate | No generation. Correctly retained as a no-candidate run; not counted as a successful draft. |
| `018461e5b5cf7a482d3f` | Close/no overlap at 4 s; overlap at 5 s; close/no overlap at 8 s | Summary preserves chronology. **Question field makes two unsupported assertions:** “Another camera view is available” and “Site procedure applies” followed by a made-up quoted procedure. Neither exists in supplied observations/unknowns. The second list item is also a statement rather than a question. Automatically accepted. 34 vs 43 summary words. |

Four drafts have no definite new event-fact assertion identified in this review; three of those have the wording concerns above. This does not make them proven accurate against the video. All drafts require review. These results do not support “5/5 factual”, “hallucination-free”, improved reviewer efficiency or template superiority.

## Upstream evidence-selection limitation

Run `005bc639df8c4a2b36e0` has an actual predicted close/non-overlap observation at 5 s inside the full candidate record, between overlap observations. The selected generation input contains only 0 s and 6 s. Therefore the language model cannot mention the intermediate change, and neither can the template. This is an evidence-selection omission, not a generation error. Completeness scores above deliberately concern **supplied** observations only. A future selector should preserve state transitions and be evaluated on new data; this frozen result stays as recorded.

## Product consequence

The recommended change is to keep deterministic review questions, or to reject non-question or assertive content before presentation, while retaining this frozen result. A later repair carries its own label and checks; it does not retroactively improve these five outputs. Generated summaries can remain an optional draft aid, but there is no demonstrated incremental efficiency benefit. Stage I therefore demonstrates a running local pipeline with exposed evidence, measured limitations and an explicit fallback.
