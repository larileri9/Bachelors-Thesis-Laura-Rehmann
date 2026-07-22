from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

# Mirrors SeqNN.set_defaults in baskerville.
_SEQNN_MODEL_DEFAULTS: dict[str, Any] = {
    "augment_rc": False,
    "augment_shift": [0],
    "strand_pair": [],
    "verbose": True,
    "num_features": 4,
}


class BaskervilleConfigError(ValueError):
    """Raised when a Baskerville-style params JSON is invalid."""


def _ensure_dict(value: Any, key_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BaskervilleConfigError(f"Expected '{key_name}' to be an object, got {type(value)!r}")
    return value


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return _ensure_dict(payload, "root")


def load_data_stats(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return load_json(path)


def load_strand_pairs_from_targets(targets_path: str | Path | None) -> list[list[int]]:
    if targets_path is None:
        return []

    df = pd.read_csv(targets_path, sep="\t", index_col=0)
    if "strand_pair" not in df.columns:
        return []

    series = df["strand_pair"].fillna(-1).astype(int)
    # Keep the same nesting shape as baskerville scripts: [np.array(...)]
    return [series.tolist()]


def normalize_baskerville_params(
    params: dict[str, Any],
    *,
    data_stats: dict[str, Any] | None = None,
    strand_pairs: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Normalize params with semantics aligned to baskerville scripts + SeqNN defaults."""
    normalized = copy.deepcopy(params)

    if "model" not in normalized or "train" not in normalized:
        raise BaskervilleConfigError("params.json must contain both 'model' and 'train' sections")

    model = _ensure_dict(normalized["model"], "model")
    train = _ensure_dict(normalized["train"], "train")

    # SeqNN default semantics.
    for key, value in _SEQNN_MODEL_DEFAULTS.items():
        model.setdefault(key, value)

    # Script-level overrides used repeatedly in baskerville scripts.
    if strand_pairs:
        model["strand_pair"] = strand_pairs

    model["num_features"] = int(model.get("num_features", 4))

    stats = data_stats or {}
    num_species = int(stats.get("num_species", 1))

    task = train.get("task")
    loss = train.get("loss")

    # Mirrors hound_train / hound_eval_mlm logic.
    if task == "fine-tune":
        num_species = 165
    if task in {"fine-tune", "self-supervised"} or loss == "mlm":
        model["num_features"] = num_species + 5

    return normalized


def build_normalized_params_from_files(
    *,
    params_file: str | Path,
    data_stats_file: str | Path | None = None,
    targets_file: str | Path | None = None,
) -> dict[str, Any]:
    params = load_json(params_file)
    data_stats = load_data_stats(data_stats_file)
    strand_pairs = load_strand_pairs_from_targets(targets_file)
    return normalize_baskerville_params(params, data_stats=data_stats, strand_pairs=strand_pairs)


def dump_normalized_params(
    *,
    output_file: str | Path,
    params_file: str | Path,
    data_stats_file: str | Path | None = None,
    targets_file: str | Path | None = None,
) -> None:
    normalized = build_normalized_params_from_files(
        params_file=params_file,
        data_stats_file=data_stats_file,
        targets_file=targets_file,
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, sort_keys=True, indent=2)
