from __future__ import annotations

from pathlib import Path

from genome_lm.train.models.components.baskerville import (
    BaskervilleTorchModel,
    build_torch_model_from_baskerville_params,
)
from genome_lm.train.parsers.baskerville_config import build_normalized_params_from_files


def build_torch_model_from_files(
    *,
    params_file: str | Path,
    data_stats_file: str | Path | None = None,
    targets_file: str | Path | None = None,
) -> tuple[BaskervilleTorchModel, dict]:
    normalized = build_normalized_params_from_files(
        params_file=params_file,
        data_stats_file=data_stats_file,
        targets_file=targets_file,
    )
    model = build_torch_model_from_baskerville_params(normalized["model"])
    return model, normalized
