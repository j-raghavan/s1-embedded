"""The decoder's training objective (s1/decoder.py decision_loss) is a proper scoring rule: it is minimised when
the predicted distribution matches the soft target, and the ordinal (score) term never makes a correct prediction
look worse than a wrong one."""

import torch

from s1 import decoder

CASES = ((2, "noul"), (3, "choice"), (5, "choice"), (4, "score"), (7, "score"))


def _draw(seed):
    g = torch.Generator().manual_seed(seed)
    logits, targets, types = [], [], []
    for n, typ in CASES:
        targets.append(torch.softmax(torch.randn(n, generator=g), -1))
        logits.append(torch.randn(n, generator=g))
        types.append(typ)
    return logits, targets, types


def test_loss_is_minimised_when_logits_match_target():
    _, targets, types = _draw(0)
    perfect = [t.clamp_min(1e-6).log() for t in targets]  # softmax(perfect) == target
    wrong = [t.flip(0).clamp_min(1e-6).log() for t in targets]
    assert decoder.decision_loss(perfect, targets, types) < decoder.decision_loss(wrong, targets, types)


def test_loss_is_finite_and_nonnegative_for_every_task_type():
    logits, targets, types = _draw(1)
    for z, t, typ in zip(logits, targets, types):
        loss = decoder.decision_loss([z], [t], [typ])
        assert torch.isfinite(loss) and loss.item() >= 0.0


def test_ordinal_term_penalises_distant_mistakes_more():
    # A 5-level score target concentrated at level 0: predicting level 1 must cost less than predicting level 4.
    target = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    near = torch.tensor([0.0, 5.0, 0.0, 0.0, 0.0])
    far = torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0])
    assert decoder.decision_loss([near], [target], ["score"]) < decoder.decision_loss([far], [target], ["score"])
