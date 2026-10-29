"""Optimiser used by every training script in this repository.

`torch.optim` is deliberately not used. Importing it pulls in `torch._dynamo`,
which allocates more than the 2 GB memory limit of the container this project is
trained in, and training dies with a MemoryError before the first step.

The implementation below is plain Adam (Kingma & Ba, 2015) with the same
defaults, plus state_dict/load_state_dict so a run can be resumed exactly where
it stopped without the moment estimates resetting to zero.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch


class Adam:
    """Minimal Adam over a list of parameters."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Sequence[float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        if not self.params:
            raise ValueError("Adam received no parameters")
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.step_count = 0

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        bias1 = 1.0 - self.beta1**self.step_count
        bias2 = 1.0 - self.beta2**self.step_count
        for index, param in enumerate(self.params):
            grad = param.grad
            if grad is None:
                continue
            if self.weight_decay:
                grad = grad.add(param, alpha=self.weight_decay)
            self.m[index].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            self.v[index].mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            denom = (self.v[index] / bias2).sqrt().add_(self.eps)
            param.add_(self.m[index] / bias1 / denom, alpha=-self.lr)

    def state_dict(self) -> dict:
        return {
            "m": self.m,
            "v": self.v,
            "step_count": self.step_count,
            "lr": self.lr,
        }

    def load_state_dict(self, state: dict) -> None:
        self.m = state["m"]
        self.v = state["v"]
        self.step_count = state["step_count"]
        self.lr = state.get("lr", self.lr)
