# s1-embedded

I built [s1](https://github.com/j-raghavan/s1-decision-model) as a calibrated System One decision model: fast typed
decisions (`choice`, `score`, `yes/no`) over text or JSON, each with a calibrated probability, read from the model's
logits with no text generated — one forward pass for questions with up to 26 options, and a few passes more for the
rare question with dozens of options, which is scored digit by digit. That model is a LoRA fine-tune of Gemma 4
26B-A4B — accurate, but far too big to run on a phone or a laptop.

This repo is where I shrink it. I keep the same teacher and the same training data, and swap the student for a tiny
on-device decoder: **`google/gemma-3-1b-it`** (about 26× smaller than the teacher), with **`google/gemma-3-270m-it`**
as a second, even smaller arm. The question is simple: how much of the 26B model's decision quality survives
distillation into a student small enough to run anywhere.

The answer, measured with the same held-out, dev-selected discipline as the parent project: **the 1B student clears the
bar; the 270M does not.**

## Results

Held-out test, scored with the parent project's scorer and per-task calibration — JevBench (1,700 cases) and my custom
structured-decision set (1,200). The bar to beat is **0.492** on JevBench, the score of the encoder line I abandoned
before moving to decoders.

| Student | JevBench | custom decisions | over the bar? |
| --- | ---: | ---: | :---: |
| `gemma-3-1b-it`, untuned | 0.375 | 0.428 | — |
| **`gemma-3-1b-it`, distilled** | **0.507** | **0.630** | ✅ |
| `gemma-3-270m-it`, untuned | 0.351 | 0.351 | — |
| `gemma-3-270m-it`, distilled | 0.424 | 0.372 | ❌ |
| *Gemma 4 26B-A4B teacher, for scale* | *~0.81* | — | — |

Distillation lifts the 1B student's structured-decision usefulness by **+20 points** and carries its JevBench from 0.375
to 0.507, over the bar. It helps the 270M too (+7 points on JevBench) but not enough: at that capacity and training
budget the 270M stays under the line. Both students reached these numbers at well under one epoch of training. The 1B is
published; the 270M stands as the honest negative half of the A/B. The teacher still scores far higher — so this is a
tiny student that beats the baseline, not one that matches the teacher.

The published 1B adapter lives at
**[j-raghavan/s1-gemma3-1b-decision](https://huggingface.co/j-raghavan/s1-gemma3-1b-decision)**.

## The idea in one paragraph

The expensive half of the original project — having the Gemma 4 26B teacher label ~90,000 decision rows with calibrated
probability distributions over the answer options — is already done and
[published](https://huggingface.co/datasets/j-raghavan/s1-decision-data). Those soft targets are the labels; they don't
care what the student is. So "repeat the teacher ↔ student exercise" here means: keep the teacher's targets, point the
same option-letter-logit decoder machinery at a 1B student, train, and measure it on the same JevBench and custom
held-out sets. The HTTP surface stays a decision endpoint (`POST /v1/decisions`); only the model behind it shrinks.

## What's here

| Path | Purpose |
| --- | --- |
| `s1/decoder.py` | The decision model: load a Gemma 3 decoder, read the option-letter logits in one pass (`Readout`, `option_log_probs`), soft-target distillation loss (`decision_loss`) |
| `s1/train_decoder.py` | LoRA-train the 1B student on the teacher's soft targets, validating on held-out task families |
| `eval/jevbench/` | The JevBench + custom runners, scorer and per-task calibration, carried over from the parent project |
| `api/` | The `/v1/decisions` server (the model behind it is swappable) |
| `pipeline/` | Training-data tooling, reused for any re-mix |
| `colab/` | Colab session machinery (busy-kernel launcher, cost guard, watcher) |
| `docs/plan.md` | The full plan: what's reused, the Gemma 3 code changes, the free pre-flight checks, the success gate, and the one paid Colab run |

## Status

Trained and measured. The 1B student is distilled, selected on held-out dev sets, tested and published; the 270M arm is
trained and measured too, and lands under the bar. See [docs/plan.md](docs/plan.md) for the full plan and
[COMPUTE_LOG.md](COMPUTE_LOG.md) for every paid run that got here.

## Relationship to s1

- **Published 1B student:** [j-raghavan/s1-gemma3-1b-decision](https://huggingface.co/j-raghavan/s1-gemma3-1b-decision)
- **Parent project:** [j-raghavan/s1-decision-model](https://github.com/j-raghavan/s1-decision-model)
- **Teacher model:** [j-raghavan/s1-gemma4-26b-decision](https://huggingface.co/j-raghavan/s1-gemma4-26b-decision)
- **Training data (shared):** [j-raghavan/s1-decision-data](https://huggingface.co/datasets/j-raghavan/s1-decision-data)

Origin and licence of every model and dataset: [LINEAGE.md](LINEAGE.md).

## Licence

Apache 2.0.
