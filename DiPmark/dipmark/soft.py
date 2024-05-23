#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import FloatTensor, LongTensor
from torch.nn import functional as F

from . import AbstractWatermarkCode, AbstractReweight, AbstractScore


class Soft_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, shuffle: LongTensor):
        self.shuffle = shuffle
        self.unshuffle = torch.argsort(shuffle, dim=-1)

    @classmethod
    def from_random(
        cls,
        rng: torch.Generator | list[torch.Generator],
        vocab_size: int,
    ):
        if isinstance(rng, list):
            batch_size = len(rng)
            shuffle = torch.stack(
                [
                    torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
        else:
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
        return cls(shuffle)


class Soft_Reweight(AbstractReweight):
    watermark_code_type = Soft_WatermarkCode

    def __init__(self, delta: float = 0.0, gamma: float = 0.5):
        self.delta = delta
        self.gamma = gamma
    def __repr__(self):
        return f"Soft_Reweight(delta={self.delta})"

    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        """
        \textbf{$soft$-reweight:}
        
        """
        # s_ means shuffled
        s_p_logits = torch.gather(p_logits, -1, code.shuffle)
        n = s_p_logits.shape[-1]
        s_p_logits[..., int(n*self.gamma):] = s_p_logits[..., int(n*self.gamma):] + self.delta
        return torch.gather(s_p_logits, -1, code.unshuffle)
        