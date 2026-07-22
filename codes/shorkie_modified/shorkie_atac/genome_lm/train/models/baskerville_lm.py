"""Lightning wrapper for the Baskerville architecture.

Adapts the Baskerville model to the genome-lm data pipeline.
Species identity is encoded shorkie-style: a one-hot species vector is
concatenated to the DNA one-hot encoding at every position, producing
a (seq_len, num_species + 6) input tensor.

The 6 DNA channels are: A, C, G, T, N, mask (matching DNALMPreprocessing
with num_classes=6).
"""

from __future__ import annotations

import json
from typing import Any

import torch
import torch.nn.functional as F

from genome_lm.train.models.base import GenomeLM
from genome_lm.train.models.components.baskerville import (
    BaskervilleTorchModel,
    build_torch_model_from_baskerville_params,
)
try:
    from genome_lm.train.models.components.fixed_species_vocab import (
        FixedSpeciesVocab,
        make_species_vocab,
    )
except ImportError:
    FixedSpeciesVocab = None  # type: ignore
    make_species_vocab = None  # type: ignore
from genome_lm.train.models.optimizers.config import AdamWConfig, OptimizerConfig


class BaskervilleLM(GenomeLM):
    """Baskerville architecture wrapped for the genome-lm training loop."""

    DNA_VOCAB_SIZE = 6  # A C G T N mask

    def __init__(
        self,
        baskerville_params_file: str,
        optimizer: OptimizerConfig = AdamWConfig(),
        warmup_steps: int = 10000,
        cooldown_steps: int = 0,
        species_vocab_tree_taxid: str | int | None = None,
        species_vocab_size: int | None = None,
        softmask_loss_factor: float = 1.0,
        gradient_clip_algorithm: str | None = "norm",
        gradient_clip_val: float | None = 1.0,
        accumulate_grad_batches: int = 1,
        torch_compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer_config = optimizer
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.softmask_loss_factor = softmask_loss_factor
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self._torch_compile = torch_compile

        with open(baskerville_params_file) as f:
            raw_params = json.load(f)
        self.baskerville_model_cfg = raw_params["model"]

        if species_vocab_tree_taxid is not None:
            self.species_vocab: FixedSpeciesVocab | None = make_species_vocab(
                species_vocab_tree_taxid
            )
        else:
            self.species_vocab = None

        self._resolved_num_species = species_vocab_size
        self.model: BaskervilleTorchModel | None = None

    def configure_model(self) -> None:
        super().configure_model()
        if self.model is not None:
            return

        num_species: int
        if self.species_vocab is not None:
            num_species = self.species_vocab.size
        elif self._resolved_num_species is not None:
            num_species = self._resolved_num_species
        else:
            num_species = 0

        cfg = dict(self.baskerville_model_cfg)
        cfg["num_features"] = self.DNA_VOCAB_SIZE + num_species
        head_key = next((k for k in cfg if k.startswith("head")), None)
        if head_key:
            cfg[head_key] = dict(cfg[head_key])
            cfg[head_key]["units"] = 4
            cfg[head_key]["activation"] = "linear"

        self.model = build_torch_model_from_baskerville_params(cfg)
        self.model.apply_tf_initialization()
        if self._torch_compile:
            self.model = torch.compile(self.model)  # type: ignore[assignment]
        self._num_species = num_species

    def _resolve_species(self, batch: dict[str, Any]) -> torch.Tensor:
        """Return integer species ids [B]."""
        if "species_id" in batch:
            return batch["species_id"]
        if self.species_vocab is not None:
            if "lineage" in batch:
                ids = self.species_vocab.map_lineages(batch["lineage"])
            elif "assembly" in batch:
                ids = self.species_vocab.map_assemblies(batch["assembly"])
            else:
                raise ValueError("Batch must contain 'species_id', 'lineage', or 'assembly'")
            return torch.tensor(ids, dtype=torch.long, device=self.device)
        return torch.zeros(batch["input_ids"].shape[0], dtype=torch.long, device=self.device)

    def _encode_input(self, batch: dict[str, Any]) -> torch.Tensor:
        """Build shorkie-style [B, L, DNA_VOCAB+num_species] input."""
        if "one_hot" in batch:
            dna_oh = batch["one_hot"].float()
            B, L, _ = dna_oh.shape
        else:
            input_ids = batch["input_ids"]
            B, L = input_ids.shape
            dna_oh = F.one_hot(input_ids, num_classes=self.DNA_VOCAB_SIZE).float()

        if self._num_species > 0:
            species_ids = self._resolve_species(batch)
            species_oh = F.one_hot(species_ids, num_classes=self._num_species).float()
            species_expanded = species_oh.unsqueeze(1).expand(B, L, self._num_species)
            return torch.cat([dna_oh, species_expanded], dim=-1)
        return dna_oh

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        assert self.model is not None
        x = self._encode_input(batch)
        logits = self.model(x).float()  # [B, L, 4]
        return batch | {"logits": logits}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        batch = self(batch)
        logits = batch["logits"]
        sequence = batch["sequence"]
        selection = batch["selection"]

        loss = F.cross_entropy(
            logits[selection],
            sequence[selection],
            reduction="mean",
        )

        if self.softmask_loss_factor != 1.0 and "softmask" in batch:
            softmask = batch["softmask"]
            sm_sel = selection & softmask
            if sm_sel.any():
                sm_loss = F.cross_entropy(
                    logits[sm_sel],
                    sequence[sm_sel],
                    reduction="mean",
                )
                loss = loss + (self.softmask_loss_factor - 1.0) * sm_loss

        self.log("ce", loss.detach())
        self.compute_throughput(batch)
        return batch | {"loss": loss}

    def configure_optimizers(self) -> Any:
        def lr_lambda(step: int) -> float:
            if step <= self.warmup_steps:
                return (step + 1) / (self.warmup_steps + 1)
            steps_remaining = self.trainer.max_steps - step
            if steps_remaining <= self.cooldown_steps:
                return steps_remaining / (self.cooldown_steps + 1)
            return 1.0

        optimizer = self.optimizer_config.make(self.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
