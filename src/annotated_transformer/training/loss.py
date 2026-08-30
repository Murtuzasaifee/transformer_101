"""Label smoothing loss and the glue that turns model output into a scalar loss."""

import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    """KL-divergence loss against a label-smoothed target distribution.

    Instead of a one-hot target, `confidence` mass goes to the true class and
    the remaining `smoothing` mass is spread uniformly over the other classes.
    This discourages the model from becoming over-confident (Szegedy et al.).
    """

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist: torch.Tensor = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # x: (n_tokens, vocab_size) predicted log-probabilities (one row per
        #    token in the flattened batch).
        # target: (n_tokens,) the correct vocab id for each token.
        assert x.size(1) == self.size

        # Build the "soft" target distribution one row at a time, matching
        # x's shape. We start from a clone purely to get the right shape/
        # dtype/device -- its values get fully overwritten below.
        true_dist = x.data.clone()

        # Step 1: give every class a small, equal share of probability mass,
        # except the true class and the padding class (2 classes reserved) --
        # this is the "smoothing" part: instead of the model being trained
        # to output 100% confidence on the correct word, it's trained to
        # leave a little probability on every other word too, which prevents
        # over-confident, poorly-calibrated predictions.
        true_dist.fill_(self.smoothing / (self.size - 2))

        # Step 2: overwrite the true class's probability with `confidence`
        # (1 - smoothing). scatter_(dim=1, index, value) writes `value` into
        # true_dist[row, target[row]] for every row -- i.e. "at the column
        # matching this row's correct answer, put most of the probability
        # mass there". target.unsqueeze(1) reshapes (n_tokens,) to
        # (n_tokens, 1) because scatter_ needs an index tensor with the same
        # number of dims as true_dist.
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # Step 3: the padding token is never a valid prediction, so it should
        # never receive any probability mass, even the small smoothing share.
        true_dist[:, self.padding_idx] = 0

        # Step 4: rows that are themselves padding (i.e. target == pad) don't
        # correspond to a real word at all -- zero out their entire target
        # distribution so they contribute exactly 0 to the KLDiv loss below.
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        # Stashed for inspection/visualization (e.g. in tests or notebooks).
        self.true_dist = true_dist

        # KL divergence between the model's predicted log-probs (x) and this
        # smoothed target distribution -- `.clone().detach()` ensures no
        # gradient flows into true_dist itself, only into x.
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    """Projects decoder output to vocab logits, computes the (normalized) loss."""

    def __init__(self, generator: nn.Module, criterion: nn.Module):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x: torch.Tensor, y: torch.Tensor, norm) -> "tuple[torch.Tensor, torch.Tensor]":
        # x: raw decoder output (batch, seq_len, d_model) -> project to vocab
        # log-probabilities (batch, seq_len, vocab_size).
        x = self.generator(x)

        # Flatten batch and seq_len into one dimension so LabelSmoothing can
        # treat every token position (across every sequence in the batch) as
        # an independent row: (batch * seq_len, vocab_size) vs (batch * seq_len,).
        # Divide by `norm` (token count) so the loss is a per-token average,
        # not a per-batch sum -- otherwise larger batches would automatically
        # report a larger loss with no change in translation quality.
        sloss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        )
        # Two return values serve two different purposes:
        #   sloss.data * norm -- a detached (no-grad) scalar, rescaled back to
        #       a total-loss magnitude, purely for logging/reporting.
        #   sloss -- the still-differentiable, per-token-normalized loss that
        #       run_epoch actually calls .backward() on.
        return sloss.data * norm, sloss
