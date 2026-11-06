# UGen300 drafts with the Qwen2.5-1.5B-Instruct HEF instead of the VLM in text-only mode (2026-11-06)

**Conditions: UGen300 over USB at 0.9 A (reduced-performance mode).** The same module, cable, host and HailoRT 5.3.2 as every record in [`../`](../README.md); HailoRT warned at every start that the module may run in reduced-performance mode. Every time here is what this setup gave, not the device's specification.

**Question.** On the device the brief is drafted by the Qwen2-VL-2B HEF in text-only mode (`frames=[]`), because only one GenAI model can be loaded at a time and the VLM is already resident for the visual check. The CPU path drafts with Qwen2.5-1.5B-Instruct, and Hailo compiles the same model for HailoRT 5.3.0 (`Qwen2.5-1.5B-Instruct.hef`, 2,464,582,493 bytes, SHA-256 `0856a10c2b804cfe1590b80aa5e28369eb7169a9b36de836420a0b8cdf5b8d05`, from `https://dev-public.hailo.ai/v5.3.0/blob/Qwen2.5-1.5B-Instruct.hef`). Would drafting with it, at the cost of swapping models (release the VLM, load the LLM, draft, release the LLM, load the VLM again), be better?

**Answer: no; the device keeps drafting with the VLM.** Under the rule fixed beforehand, the LLM HEF had to pass the source checks on at least 8 of the 10 held-out candidates and draft faster than the VLM; it passed 0 of 20, took longer per draft, and every time the VLM was loaded again after it, the UGen300 dropped off USB.

## The rule, fixed before the first draft

[`PREREG.json`](PREREG.json) was written after a probe (`probe_defaults.py`) had read the two HEFs' default generation settings ([`defaults.json`](defaults.json); nothing generated) and before the LLM drafted anything. In short:

- **Candidates and checks** as in [`../look-agreement`](../look-agreement/README.md): the 20 candidates of `draft_events.json`, with its own split (d01, d03, … selection; d02, d04, … held out), each drafted through `device_pipeline.generate` (parse, `generation.validate_draft`, compose, or the labeled template) and assessed by that study's analysis.
- **Generation**: `hailo_backend.HailoLLM`, the draft-v2 messages with the device format hint appended (as for the VLM), plain-text message content, greedy, seed 19, at most 320 tokens, `frequency_penalty` 1.0, context cleared before and after each draft, and the stream left at the end of the first JSON object. The HEF's own default is `frequency_penalty` 1.1 (Qwen2.5's `repetition_penalty`); the CPU worker applies no repetition penalty (llama.cpp `repeat_penalty` 1.0), and the VLM HEF's default, which the VLM drafts used, is 1.0. There is no grammar on the device.
- **Format hints**: draft-v2-device on all 20; the two stricter hints of the look-agreement study (draft-v2-device1, draft-v2-device2) on the selection half, chosen by that study's `SELECTION.json` rule with the LLM's draft-v2-device as the baseline; a chosen hint would then run on the held-out half.
- **Decision**: adopt the LLM for the draft phase only if, with the chosen hint, (1) at least 8 of the 10 held-out drafts pass the source checks (the VLM's 8 of 10), (2) the shown held-out briefs hold at most 2 unsupported sentences (the CPU's count), (3) the median seconds per draft is below 41.891 s (the VLM's, all 20), and (4) the round-trip swap (VLM release + LLM load + LLM release + VLM load) costs less than 9 × the saving per draft (9 = the drafts of a nine-candidate draft phase).

## Results

### Drafts of the 20 candidates

| | Device, Qwen2.5-1.5B HEF (draft-v2-device) | Device, Qwen2-VL-2B text-only (draft-v2-device) | CPU, Qwen2.5-1.5B Q4_K_M + JSON grammar |
|---|---|---|---|
| Passed the source checks | **0 / 20** (selection 0 / 10, held out **0 / 10**) | 17 / 20 (held out 8 / 10) | 20 / 20 (held out 10 / 10) |
| Reply not readable as JSON | 6 / 20 (all six ran to the 320-token limit) | 1 / 20 | 0 / 20 |
| Questions under `review_questions`, as asked | 2 / 20 (18 used `reviewQuestions`; 3 also `evidenceIds`) | 20 / 20 | 20 / 20 |
| Asks whether another camera view is available / which site procedure applies (under either key) | 0 / 20, 0 / 20 | 2 / 20, 0 / 20 | 20 / 20, 20 / 20 |
| Questions written as objects with an answer | 5 of the 14 readable replies | 0 | 0 |
| Titles naming neither a person nor a forklift (readable replies) | 2 / 14 | 13 / 19 | 0 / 20 |
| Unsupported sentences in the briefs shown | 0 (all 20 shown briefs are the template) | 1 sentence in 1 brief | 3 sentences in 2 briefs |
| Seconds per draft, median / p95 | **52.7** / 81.0 (held out 59.1) | 41.9 / 63.8 (held out 42.9) | 11.6 / 13.1 |
| Tokens per draft, median | 204 | 160 | 118.5 |
| Tokens per second, median; time to first token | 4.11; 3.4 s | 4.10; 3.2 s | — |

Why each was rejected: all 14 readable replies failed "Review questions must be a short list of questions": 12 put the questions under `reviewQuestions`, which the checks do not read, and the 2 that used `review_questions` wrote each question as an object carrying an answer ("Is the woman in the white shirt … facing the blue forklift?" with "answer": "Yes"; 5 of the 14 did that); 3 of the 14 also cited `evidenceIds` instead of `evidence_ids`. The 6 unreadable replies wrote question objects, mostly with a trailing comma, until the token limit, five of them repeating the same few questions. Many questions ask whether boxes were "close enough to be considered a collision" or a near miss, the judgment the prompt rules out; the source checks reject such drafts for their form before they reach that. The LLM writes indented JSON, so it spends more tokens on the same content, at the same 4.1 tokens/s as the VLM: at 0.9 A the smaller model is not faster on this device. Its replies are deterministic: a second run of all 20 (after the first ended at the swap back, below) gave the same reply to every candidate, byte for byte ([`llm_drafts_run1_replies.json`](llm_drafts_run1_replies.json)).

**Stricter hints (selection half).** draft-v2-device1: 0 / 10 passed, all 10 unreadable: it wrote the two spelled-out questions as a JSON object without values (`"reviewQuestions": {"Is another camera view available?", "Which site procedure applies?"}`). draft-v2-device2: 0 / 10, all unreadable: a trailing comma after the questions, or questions repeated to the token limit; 7 of its 10 replies still used `reviewQuestions` although the hint spells out `review_questions`. Both were eligible under the rule (no fewer passes than draft-v2-device's 0 and no unsupported sentences) but neither came before it in the rule's order, meeting none of the raw format checks against its 23, so draft-v2-device stayed and no hint ran on the held-out half (`comparison.json`, `format_selection`).

### Swapping models on one VDevice

In one process, with the detector HEF configured on the shared VDevice as the app keeps it:

| Step | Seconds (median of 4; range) |
|---|---|
| Release the VLM | 0.53 (0.52–0.55) |
| Load the LLM (2.46 GB HEF; hashing it adds about 5 s, once) | 7.62 (7.37–7.65) |
| Swap to the LLM, both together | **8.14** (7.93–8.17) |
| Time to first token of the first draft after the load | 2.78 (2.75–3.44; all 20 drafts of a run: 2.74–3.44) |
| Release the LLM | 0.46 (0.45–0.47) |
| Load the VLM again (3.26 GB HEF) | **failed every time: 4 of 4 runs, 3 of 3 probes, and the first run** |

Each time the VLM was loaded again, HailoRT lost the device while it sent the HEF (`Failed to write chunk file 11/42 … HAILO_COMMUNICATION_CLOSED`, then `LIBUSB_ERROR_IO` and `HAILO_SHUTDOWN_EVENT_SIGNALED`, about 8 s into the load), the detector could no longer be shut down, and the UGen300 came back under a new USB address (`usb/001:024` before the first failure, then `025`, `026`, …). The process had to end; a new process could open the device again. In `probe_defaults.py`, with no detector configured and nothing generated, the LLM had loaded and been released and the VLM had then loaded (10.1 s, 0.44 s, 10.5 s), but there the VLM had not been loaded before the LLM. A second probe (`probe_swap_back.py`) then changed one thing at a time, each in its own process: without the detector (`swapback_no_detector.json`), without any draft (`swapback_no_draft.json`) and with a 10 s pause before the VLM load (`swapback_settle10.json`) the VLM failed to load again in the same way, 8.4–8.6 s into the load; with no LLM in between (`swapback_no_llm.json`: VLM released and loaded again) it loaded in 13.6 s and answered the visual check. So on this setup a VDevice that has held the VLM and then the LLM cannot load the VLM again; whether that is the 0.9 A supply, the firmware or HailoRT 5.3.2 is open.

So no round trip completed, and rule (4) cannot be met; at the measured times it could not have been either, since the LLM drafts were slower.

### Decision

| Rule (PREREG.json) | Needed | LLM HEF | Met |
|---|---|---|---|
| (1) Held-out drafts passing the source checks | ≥ 8 / 10 | 0 / 10 | no |
| (2) Unsupported sentences in the held-out briefs shown | ≤ 2 | 0 (all templates) | yes |
| (3) Median seconds per draft | < 41.891 | 52.742 | no |
| (4) Round-trip swap below 9 × saving per draft | < 9 × (41.891 − 52.742) | no round trip completed | no |

**Not adopted.** The app is unchanged: the device drafts with the resident Qwen2-VL-2B in text-only mode, and no model is swapped during a job. `hailo_backend.HailoLLM` and the `draft_model` switch in `device_pipeline.generate` stay, so these records can be reproduced; `HailoRuntime` never loads the LLM.

## Files

| File | Contents |
|---|---|
| `PREREG.json` | The rule, written before the first LLM draft |
| `defaults.json` | The two HEFs' default generation settings, context capacity (2,048) and the LLM's chat template; read before the rule was written |
| `llm_drafts.json` | draft-v2-device on all 20 candidates (`device_pipeline.generate` audits), the swap times and the failed swap back |
| `llm_drafts_run1_replies.json` | The first run's replies, kept to show the second run reproduced them |
| `llm_switch.json` | The run meant to time three swaps (`--switch-only`): it ended at its first swap back |
| `llm_drafts_device1_selection.json`, `llm_drafts_device2_selection.json` | The stricter hints on the selection half |
| `swapback_*.json` | Which part of the swap back makes the device drop |
| `comparison.json` | Every number above, from the records here and `../look-agreement` |
| `MANIFEST.json` | SHA-256 of every file here |

## How they were made

On the device host's Windows interpreter (HailoRT 5.3.2, Python 3.10), with TraceBrief imported from this repository, a local manifest holding the detector and VLM hashes and the LLM HEF downloaded from Hailo's resource server; one process held the device at a time. The driver loaded the app's `HailoRuntime`, swapped the VLM for `HailoLLM` on the same VDevice and drafted every candidate of `../look-agreement/draft_events.json` through `device_pipeline.generate` (`--switch-only --cycles 3` for the swap timing run; `--format draft-v2-device1` / `draft-v2-device2 --split selection` for the hint trials); the swap-back probes ran one variant per process (`--no-draft`, `--no-detector`, `--settle 10`, and no LLM). The driver and probe scripts ran outside the repository and are not part of it; host paths in the records are shortened to `C:\tb-device`.

## Limits

- One device at one power condition (0.9 A). The swap-back failure may be a power effect; it has to be tried again on a 3 A USB-C port before it is called a property of the device or of HailoRT 5.3.2.
- 20 candidates from a few synthetic clips; passing the source checks is not a quality audit.
- Only the three device format hints of the look-agreement study were tried; a hint written for this model (its key naming, compact JSON) might do better, and was not part of the rule.
