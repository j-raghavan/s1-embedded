# Plan: distil the 26B teacher into a Gemma 3 1B student

## Goal

Take the s1 decision model — a LoRA fine-tune of Gemma 4 26B-A4B that scores 0.808 / 0.817 on the JevBench subset —
and see how much of that decision quality I can keep in a student small enough to run on a phone or a laptop:
`google/gemma-3-1b-it`, a dense, text-only decoder about 26× smaller.

Same teacher, same data, same eval, same one-forward-pass option-letter readout. The only thing that changes is the
student, and (because the student is a different model) the turn format it's prompted in.

## What I reuse unchanged

- **The teacher's soft targets.** The expensive step — the Gemma 4 26B teacher labelling ~90,000 decision rows with
  calibrated probability distributions over the answer options — is already done and published as
  [j-raghavan/s1-decision-data](https://huggingface.co/datasets/j-raghavan/s1-decision-data). Those targets are the
  labels, and they don't depend on the student, so I do **not** re-run the teacher.
- **The decoder machinery** (`s1/decoder.py`): render the prompt, one forward pass, softmax over the option-letter
  logits at the last position (`Readout`, `option_log_probs`), soft-target distillation loss (`decision_loss`).
- **The trainer** (`s1/train_decoder.py`): LoRA, per-example option shuffling, held-out-family validation, keep the
  adapter with the lowest unseen-family loss.
- **The eval suites, scorer and per-task calibration** under `eval/jevbench/`, and the `/v1/decisions` API.

## Code changes for Gemma 3 (done)

- `s1/decoder.py` `load()` now loads `AutoModelForCausalLM` (Gemma 3 1B is dense and text-only — no mixture-of-experts,
  no vision/audio towers, no `model.language_model.*` prefix), with a backbone-only missing-keys guard (lm_head is
  tied, so it never counts as missing).
- `LORA_TARGETS` now matches every `q/k/v/o_proj` and `gate/up/down_proj` (the Gemma-4 `language_model` path segment is
  gone).
- `run_decoder.py` / `train_decoder.py` default to `google/gemma-3-1b-it`; the Gemma-4-only `--experts-impl` is gone.

## Pre-flight checks before any paid run (free, local)

These are the real unknowns, and all of them are checkable on CPU/MPS with just the tokenizer or a tiny forward pass.
Nothing on Colab until these pass.

1. **Prompt format.** Render `google/gemma-3-1b-it`'s real `chat_template` with `add_generation_prompt=True` and set
   `GEMMA3_PROMPT` in `eval/jevbench/run_ollama.py` from it. The value committed there now is a hypothesis
   (`<start_of_turn>…<start_of_turn>model\n`, no thought channel) and is marked as such — it must be confirmed or
   corrected against the actual template, not assumed.
2. **Option-letter tokenization.** Confirm `" A".." Z"` are each a single token under Gemma 3's tokenizer (the whole
   readout depends on it; `Readout.__init__` already asserts this).
3. **`<bos>` behaviour.** `<bos>` was worth about 2.8 points for the 26B. Confirm Gemma 3 handles a prepended `<bos>`
   the same way and that the rendered prompt tokenizes cleanly.
4. **Smoke train.** A few steps locally to confirm the loop runs end-to-end and the loss drops on the 1B base.
5. **Colab dry run.** Exercise the session watcher, filters, cost guard and end condition before spending a unit — a
   silent watcher has cost units before.

## Success gate (fix before training)

A 1B student will land below the 26B's 0.808 — that's expected and fine. The bar that makes this worth shipping:

- Clearly beat the abandoned encoder line (JevBench 0.492).
- Hold the custom structured-decision usefulness (the 26B is at 0.981 calibrated).
- Chosen on the dev splits only, by a rule fixed before the test set is touched — same discipline as the parent.

## The one paid Colab run

Train the 1B LoRA on the published targets (same policy / injection mix), dev-select the snapshot, score the JevBench
test subset and the custom set once. A 1B dense model fits comfortably on the cheaper A100 40GB, so this is far lighter
than any 26B run in the parent's compute log. Rough estimate ~3–6 units; pinned after the smoke test and confirmed
before launch.

## Open decisions

- **LoRA vs full fine-tune.** For a 1B student a full fine-tune is feasible and may distil better than LoRA. I'll run
  LoRA first (reuses the machinery, directly comparable to the 26B) and treat full FT as the next lever.
- **A dedicated decision head** (score each option from the final hidden state) instead of the letter-logit readout is
  the fallback if the letter-readout 1B misses the gate — it's more robust for a small model but breaks apples-to-apples
  comparison and needs its own serving path.
- **Gemma 3 270M** as a cheaper A/B in the same session, if there's budget headroom.
