from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass
class EvaluationBundle:
    """Container for serialized evaluation outputs."""

    preds: np.ndarray
    targets: np.ndarray
    metadata: list[dict[str, Any]]
    per_window_r: np.ndarray
    all_per_window_r: np.ndarray


def _to_numpy(arr: np.ndarray | list[Any]) -> np.ndarray:
    return np.asarray(arr)


def coerce_metadata(metadata: Any, n_samples: int) -> list[dict[str, Any]]:
    """Normalize metadata to a list of dictionaries with JSON-friendly values."""
    if metadata is None:
        return [{} for _ in range(n_samples)]

    if isinstance(metadata, np.ndarray):
        raw_items = metadata.tolist()
    elif isinstance(metadata, list):
        raw_items = metadata
    else:
        raw_items = [metadata]

    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            normalized_item: dict[str, Any] = {}
            for key, value in item.items():
                if isinstance(value, np.generic):
                    normalized_item[str(key)] = value.item()
                else:
                    normalized_item[str(key)] = value
            normalized.append(normalized_item)
        else:
            normalized.append({"value": item})

    if len(normalized) < n_samples:
        normalized.extend({} for _ in range(n_samples - len(normalized)))
    elif len(normalized) > n_samples:
        normalized = normalized[:n_samples]

    return normalized


def flatten_per_sample(arr: np.ndarray) -> np.ndarray:
    """Flatten every sample to 1D while keeping the sample axis (N, ...)."""
    arr = _to_numpy(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr.reshape(arr.shape[0], -1)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation that returns 0.0 for near-constant inputs."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    r, _ = pearsonr(x, y)
    if np.isnan(r):
        return 0.0
    return float(r)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation that returns 0.0 for near-constant inputs."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    r, _ = spearmanr(x, y)
    if np.isnan(r):
        return 0.0
    return float(r)


def compute_per_window_pearson(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute Pearson R per sample/window by flattening non-sample dimensions."""
    preds = _to_numpy(preds)
    targets = _to_numpy(targets)

    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds{preds.shape} vs targets{targets.shape}")

    pred_flat = flatten_per_sample(preds)
    target_flat = flatten_per_sample(targets)

    rs = [safe_pearson(pred_flat[i], target_flat[i]) for i in range(pred_flat.shape[0])]
    return np.asarray(rs, dtype=np.float32)


def calculate_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute summary metrics for prediction vs target arrays."""
    preds = _to_numpy(preds)
    targets = _to_numpy(targets)

    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds{preds.shape} vs targets{targets.shape}")

    per_window_r = compute_per_window_pearson(preds, targets)
    pred_flat = preds.reshape(-1)
    target_flat = targets.reshape(-1)

    metrics = {
        "mse": float(np.mean((preds - targets) ** 2)),
        "mae": float(np.mean(np.abs(preds - targets))),
        "pearson_r": safe_pearson(pred_flat, target_flat),
        "spearman_r": safe_spearman(pred_flat, target_flat),
        "mean_per_window_pearson": float(np.mean(per_window_r)) if per_window_r.size else 0.0,
        "median_per_window_pearson": float(np.median(per_window_r)) if per_window_r.size else 0.0,
    }
    return metrics


def select_export_indices(
    per_window_r: np.ndarray,
    best_k: int = 50,
    worst_k: int = 50,
    random_k: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Pick representative windows to save: best/worst/random by per-window R."""
    per_window_r = np.asarray(per_window_r)
    n = len(per_window_r)
    if n == 0:
        return np.asarray([], dtype=np.int64)

    sorted_idx = np.argsort(per_window_r)
    selected: list[np.ndarray] = []

    if best_k > 0:
        selected.append(sorted_idx[-min(best_k, n):])
    if worst_k > 0:
        selected.append(sorted_idx[: min(worst_k, n)])
    if random_k > 0:
        rng = np.random.default_rng(seed)
        random_idx = rng.choice(np.arange(n), size=min(random_k, n), replace=False)
        selected.append(np.asarray(random_idx, dtype=np.int64))

    if not selected:
        return np.arange(n, dtype=np.int64)

    return np.unique(np.concatenate(selected))


def save_metrics(metrics: dict[str, float], output_path: str | Path) -> None:
    """Persist metrics as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(metrics, handle, indent=4)


def save_evaluation_bundle(
    output_path: str | Path,
    preds: np.ndarray,
    targets: np.ndarray,
    metadata: list[dict[str, Any]] | np.ndarray | None,
    per_window_r: np.ndarray,
    all_per_window_r: np.ndarray | None = None,
) -> None:
    """Persist arrays and metadata into a single NPZ file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preds = _to_numpy(preds)
    targets = _to_numpy(targets)
    per_window_r = _to_numpy(per_window_r)
    metadata_norm = coerce_metadata(metadata, n_samples=preds.shape[0])

    if all_per_window_r is None:
        all_per_window_r = per_window_r

    np.savez(
        output_path,
        preds=preds,
        targets=targets,
        metadata=np.asarray(metadata_norm, dtype=object),
        per_window_r=per_window_r,
        all_per_window_r=np.asarray(all_per_window_r),
    )


def load_evaluation_bundle(input_path: str | Path) -> EvaluationBundle:
    """Load an evaluation NPZ bundle in a backward-compatible way."""
    input_path = Path(input_path)
    with np.load(input_path, allow_pickle=True) as data:
        preds = np.asarray(data["preds"])
        targets = np.asarray(data["targets"])
        per_window_r = (
            np.asarray(data["per_window_r"])
            if "per_window_r" in data.files
            else compute_per_window_pearson(preds, targets)
        )
        all_per_window_r = (
            np.asarray(data["all_per_window_r"])
            if "all_per_window_r" in data.files
            else per_window_r
        )
        metadata = coerce_metadata(
            data["metadata"] if "metadata" in data.files else None,
            n_samples=preds.shape[0],
        )

    return EvaluationBundle(
        preds=preds,
        targets=targets,
        metadata=metadata,
        per_window_r=per_window_r,
        all_per_window_r=all_per_window_r,
    )
