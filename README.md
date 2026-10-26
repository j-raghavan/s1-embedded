# s1-embedded

I built [s1](https://github.com/j-raghavan/s1-decision-model) as a calibrated System One decision model: fast typed
decisions (`choice`, `score`, `yes/no`) over text or JSON, each with a calibrated probability, from a single forward
pass with no text generated. That model is a LoRA fine-tune of Gemma 4 26B-A4B — accurate, but far too big to run on a
phone or a laptop.

This repo is where I try to shrink it. I keep the same teacher and the same training data, and swap the student for
**`google/gemma-3-1b-it`** — a dense, text-only, on-device decoder about 26× smaller than the teacher. The question I
want to answer is simple: how much of the 26B model's decision quality survives distillation into a 1B student small
enough to run anywhere.

This is early, hands-on research. There are **no benchmark results yet** — when there are, they'll live here with the
same held-out, dev-selected discipline as the parent project.

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

Scaffolding and the Gemma 3 code path are in place; the prompt format and tokenizer behaviour are being verified against
the real `google/gemma-3-1b-it` before any training. See [docs/plan.md](docs/plan.md) for exactly where things stand.

## Relationship to s1

- **Parent project:** [j-raghavan/s1-decision-model](https://github.com/j-raghavan/s1-decision-model)
- **Teacher model:** [j-raghavan/s1-gemma4-26b-decision](https://huggingface.co/j-raghavan/s1-gemma4-26b-decision)
- **Training data (shared):** [j-raghavan/s1-decision-data](https://huggingface.co/datasets/j-raghavan/s1-decision-data)

Origin and licence of every model and dataset: [LINEAGE.md](LINEAGE.md).

## Licence

Apache 2.0.
