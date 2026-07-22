import abc
from dataclasses import dataclass, field

import torch
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
)


class OptimizerConfig(abc.ABC):
    @abc.abstractmethod
    def make(self, params: ParamsT) -> Optimizer: ...


@dataclass
class AdamWConfig(OptimizerConfig):
    lr: float = 4e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def make(self, params: ParamsT) -> Optimizer:
        return torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            foreach=True,
            fused=False,  # does not work with AMP and gradient clipping
        )


@dataclass
class MuonConfig(OptimizerConfig):
    lr: float = 4e-4
    weight_decay: float = 0.1
    momentum: float = 0.95
    nesterov: bool = True
    ns_coefficients: str | None = "polar_express"
    eps: float = 1e-7
    adjust_lr_fn: str = "match_rms_adamw"
    polar_scale: bool = True
    auxiliary: OptimizerConfig = field(default_factory=AdamWConfig)

    def make(self, params: ParamsT) -> Optimizer:
        from genome_lm.train.models.optimizers.muon import Muon

        return Muon(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
            ns_coefficients=self.ns_coefficients,
            eps=self.eps,
            adjust_lr_fn=self.adjust_lr_fn,
            polar_scale=self.polar_scale,
        )


@dataclass
class DionConfig(OptimizerConfig):
    lr: float = 1e-3
    weight_decay: float = 0.1
    rank_fraction: float = 1.0
    auxiliary: str = "lion"
    auxiliary_lr: float = 4e-4

    def make(self, params: ParamsT) -> Optimizer:
        from dion import Dion

        process_group = (
            torch.distributed.group.WORLD if torch.distributed.is_initialized() else None
        )
        return Dion(
            params,
            replicate_mesh=process_group,
            weight_decay=self.weight_decay,
            rank_fraction=self.rank_fraction,
        )


@dataclass
class StableAdamWConfig(OptimizerConfig):
    lr: float = 4e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def make(self, params: ParamsT) -> Optimizer:
        from optimi import StableAdamW

        return StableAdamW(
            params,  # type: ignore
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=self.eps,
            triton=True,
        )
