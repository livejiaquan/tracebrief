# Review protocol for real event drafts

Protocol recorded before final evaluation outputs, 2026-10-04.

## Sample and comparison

Use the first six held-out runs predeclared in the detector freeze, in frozen run order. Include every candidate selected by the predeclared rule from that subset, or the predeclared first candidate per run if cost requires that narrower rule. Record a run with no candidate rather than replacing it with a more favorable run. Inputs are actual detector-derived observations and known unknowns; simulator annotations are not generation inputs. Save prompt/hash, model hash, raw response, template, validation result, accepted output and timing for each.

The comparator is a **deduplicated deterministic template** retaining source timestamps and IDs. Concatenating duplicate sentences is an artificially weak baseline. Compare the generated raw draft with that template; separately report whether automatic validation rejected it and what the user ultimately saw.

## Audit questions

1. Is every asserted fact supported by the provided observation record? Split a compound sentence into atomic facts before counting. A statement like “worker moved behind the forklift” needs evidence for the actor, motion, direction and relative placement; proximity alone supports none of that trajectory.
2. Are temporal words justified by the order of observations? “Remained,” “approached,” “crossed,” “then” and “moved away” are stronger than co-occurrence. Do not give them credit without the required evidence.
3. Are all important distinct facts covered, including unknowns? Count omission of a change in box-overlap state separately from omission of repeated proximity observations.
4. Are citations valid, and do cited observations support the text? Existence of an ID only answers the first half.
5. Do review questions introduce a presupposition? “Why did the driver violate the rule?” asserts a violation even when formatted as a question.
6. Is the draft more concise without losing distinct facts? Compare words and duplicated propositions. Brevity alone does not establish reviewer efficiency.

## Reporting

For each case list unsupported facts verbatim, missing distinct facts, unjustified temporal links, unsupported question presuppositions, citations and automatic fallback outcome. Label this as a single rubric-based review, not a human study or clinical/safety validation. Inspect the actual output content; do not count schema success as factuality success. Report denominators for raw model outputs and accepted model outputs separately.

Useful narrow outcomes include “N/M drafts passed automatic checks” or “the review found no unsupported additions in N/M supplied-input drafts,” with exact sample scope. No human time saving, deployment safety or user preference can be inferred. If the template is equally clear or more faithful, report that result and explain why the model remains optional.

Wording revised for publication on 2026-10-20; questions and criteria unchanged from the 2026-10-04 version.
