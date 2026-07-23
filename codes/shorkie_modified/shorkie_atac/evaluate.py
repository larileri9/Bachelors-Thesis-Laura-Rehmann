"""Evaluation script for ATAC benchmarking."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from atac_dataset import build_datasets_from_folds
#from atac_dataset_modified import build_datasets_from_folds_whole_chromosome
from atac_dataset_modified_2 import build_datasets_from_folds_modified

from evaluation_utils import (
    calculate_metrics,
    coerce_metadata,
    compute_per_window_pearson,
    save_evaluation_bundle,
    save_metrics,
    select_export_indices,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw!r}") from exc


def _env_optional_int(name: str) -> int | None:
    raw = _env(name, "")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw!r}") from exc


def _split_semicolon(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(";") if part.strip()]


def _parse_boolish(raw: str, *, name: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be one of 1/0/true/false/yes/no/on/off, got: {raw!r}")

species_token_dict = {
    "candida_albicans": "candida_albicans",
    "saccharomyces_cerevisiae": "_saccharomyces_cerevisiae",
    "schizosaccharomyces_pombe": "schizosaccharomyces_pombe",
    "aspergillus_niger": "aspergillus_niger",
    "aspergillus_oryzae": "aspergillus_oryzae",
    "neurasposa_crassa": "neurospora_crassa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against GT with per-species and cross-model outputs."
    )

    # Checkpoint-based model specs.
    train_output_dir = _env("TRAIN_OUTPUT_DIR", "/s/project/multispecies/atac_seq_pipeline/shorkie_atac/output")
    checkpoint_dir_default = _env("CHECKPOINT_DIR", f"{train_output_dir}/checkpoints")

    parser.add_argument("--checkpoint", type=str, default=_env("CHECKPOINT", "") or None, help="Single checkpoint path.")
    parser.add_argument(
        "--checkpoint-spec",
        action="append",
        default=[],
        help="Repeated model spec: label=/path/to/model.ckpt",
    )

    # Precomputed prediction model specs (for non-Shorkie models too).
    parser.add_argument(
        "--predictions-npz",
        type=str,
        default=_env("PREDICTIONS_NPZ", "") or None,
        help="Single precomputed NPZ (simple mode).",
    )
    parser.add_argument(
        "--predictions-spec",
        action="append",
        default=[],
        help="Repeated model spec: label=/path/to/predictions.npz",
    )
    parser.add_argument("--pred-key", type=str, default=_env("PRED_KEY", "preds"), help="Prediction key for NPZ models.")
    parser.add_argument("--target-key", type=str, default=_env("TARGET_KEY", "targets"), help="Target key for NPZ models.")
    parser.add_argument("--metadata-key", type=str, default=_env("METADATA_KEY", "metadata"), help="Metadata key for NPZ models.")

    # Dataset/inference config.
    parser.add_argument(
        "--bigwig-dir",
        type=str,
        default=_env("BIGWIG_DIR", "/data/nasif12/home_if12/greissl/atac-seq-pipeline/results/bigwig") or None,
        help="GT bigwig directory (checkpoint mode).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=_env("CONFIG", "/data/nasif12/home_if12/greissl/atac-seq-pipeline/config/config.yaml") or None,
        help="Species config YAML (checkpoint mode).",
    )
    parser.add_argument("--params", type=str, default=_env("PARAMS", "configs/baskerville/shorkie_params.json"), help="Model params.")
    parser.add_argument("--species", type=str, default=_env("SPECIES", "") or None, help="Comma-separated species list; default all.")
    parser.add_argument("--seq-length", type=int, default=_env_int("SEQ_LENGTH", 16384), help="Window length.")
    parser.add_argument("--num-samples", type=int, default=_env_int("NUM_SAMPLES", 2000), help="Total windows (split across species).")
    parser.add_argument(
        "--num-samples-per-species",
        type=int,
        default=_env_optional_int("NUM_SAMPLES_PER_SPECIES"),
        help="If set, use this many windows per species.",
    )
    parser.add_argument("--batch-size", type=int, default=_env_int("BATCH_SIZE", 8), help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=_env_int("NUM_WORKERS", 8), help="DataLoader workers.")
    parser.add_argument("--device", type=str, default=_env("DEVICE", "cuda") or None, help="Device override (cpu/cuda).")


    # here for input type
    parser.add_argument("--species-lm-embedded", action="store_true", default=False,
    help="Use speciesLM embeddings instead of one-hot.")
    parser.add_argument("--embedding-root", type=str, default=None,
    help="Root dir for the Species LM embeddings.")

# the is for the shuffled test if we've made a mistake
    parser.add_argument("--shuffle-input", action="store_true", default=False,
    help="Shuffle input along sequence length before passing to the model.")
    parser.add_argument("--original-input",action="store_true", default=False,
                        help="Use 6 DNA channels (A,C,G,T,N,mask) instead of 4. Closer to pretraining encoding but still no actual species token here.")
# wrong but scared to derail smh so leaving dead code in 
    parser.add_argument("--skip-conv-dna", action="store_true", default=False,
                        help="Replace conv_dna with a small linear adapter. --> embeddings already have context",)
    parser.add_argument("--skip-edges", type=int, default=0, help="Shift eval tiling to [skip_edges, chrom_len - skip_edges).")

    # Pipeline orchestration options.
    parser.add_argument(
        "--auto-benchmark",
        type=str,
        default=_env("AUTO_TWO_MODEL_BENCHMARK", "0"),
        help="Auto-discover yeast+multispecies checkpoints when no explicit model inputs were provided.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=checkpoint_dir_default, help="Checkpoint search directory for --auto-benchmark.")
    parser.add_argument("--yeast-model-label", type=str, default=_env("YEAST_MODEL_LABEL", "yeast"), help="Label for auto-selected yeast checkpoint.")
    parser.add_argument(
        "--multispecies-model-label",
        type=str,
        default=_env("MULTISPECIES_MODEL_LABEL", "multispecies"),
        help="Label for auto-selected multispecies checkpoint.",
    )
    parser.add_argument("--yeast-checkpoint", type=str, default=_env("YEAST_CHECKPOINT", "") or None, help="Optional explicit yeast checkpoint path.")
    parser.add_argument(
        "--multispecies-checkpoint",
        type=str,
        default=_env("MULTISPECIES_CHECKPOINT", "") or None,
        help="Optional explicit multispecies checkpoint path.",
    )

    # Output config.
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_env("OUTPUT_DIR", "/s/project/multispecies/atac_seq_pipeline/shorkie_atac/evaluation"),
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=_env_int("SEED", 42), help="Random seed.")
    parser.add_argument("--best-k", type=int, default=_env_int("BEST_K", 50), help="Best windows to keep in results bundle.")
    parser.add_argument("--worst-k", type=int, default=_env_int("WORST_K", 50), help="Worst windows to keep in results bundle.")
    parser.add_argument("--random-k", type=int, default=_env_int("RANDOM_K", 100), help="Random windows to keep in results bundle.")
    parser.add_argument(
        "--save-all-samples",
        action="store_true",
        default=_parse_boolish(_env("SAVE_ALL_SAMPLES", "0"), name="SAVE_ALL_SAMPLES"),
        help="Save all windows in results.npz.",
    )
    parser.add_argument(
        "--run-plotting",
        type=str,
        default=_env("RUN_PLOTTING", "auto"),
        help="auto/1/0/true/false/yes/no/on/off.",
    )
    parser.add_argument("--plots-dir", type=str, default=_env("PLOTS_DIR", "") or None, help="Plot output root. Default: <output-dir>/plots")
    parser.add_argument(
        "--comparison-track-top-n",
        type=int,
        default=_env_int("COMPARISON_TRACK_TOP_N", 5),
        help="Number of top windows per species for comparison track plots.",
    )
    parser.add_argument(
        "--comparison-species",
        type=str,
        default=_env("COMPARISON_SPECIES", "") or None,
        help="Optional comma-separated species filter for comparison plots.",
    )

    return parser.parse_args()


def sanitize_label(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return clean or "model"


def parse_model_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []

    env_checkpoint_specs = _split_semicolon(_env("CHECKPOINT_SPECS", ""))
    env_predictions_specs = _split_semicolon(_env("PREDICTIONS_SPECS", ""))

    if args.checkpoint:
        specs.append(
            {
                "label": sanitize_label(Path(args.checkpoint).stem),
                "kind": "checkpoint",
                "path": args.checkpoint,
            }
        )

    for raw in [*env_checkpoint_specs, *args.checkpoint_spec]:
        if "=" not in raw:
            raise ValueError(f"Invalid --checkpoint-spec '{raw}'. Expected label=/path/to/model.ckpt")
        label, path = raw.split("=", 1)
        specs.append(
            {
                "label": sanitize_label(label),
                "kind": "checkpoint",
                "path": path.strip(),
            }
        )

    if args.predictions_npz:
        specs.append(
            {
                "label": sanitize_label(Path(args.predictions_npz).stem),
                "kind": "npz",
                "path": args.predictions_npz,
            }
        )

    for raw in [*env_predictions_specs, *args.predictions_spec]:
        if "=" not in raw:
            raise ValueError(f"Invalid --predictions-spec '{raw}'. Expected label=/path/to/predictions.npz")
        label, path = raw.split("=", 1)
        specs.append(
            {
                "label": sanitize_label(label),
                "kind": "npz",
                "path": path.strip(),
            }
        )

    # Ensure unique labels.
    counts: dict[str, int] = {}
    for spec in specs:
        base = spec["label"]
        idx = counts.get(base, 0)
        if idx > 0:
            spec["label"] = f"{base}_{idx+1}"
        counts[base] = idx + 1

    return specs


def _loss_from_ckpt_name(path: Path) -> float | None:
    match = re.search(r"loss=([-]?[0-9]+\.[0-9]+)", path.name)
    if not match:
        return None
    return float(match.group(1))


def _is_yeast_ckpt(path: Path) -> bool:
    return re.search(r"-v[0-9]+\.ckpt$", path.name) is not None


def best_loss_ckpt_by_mode(checkpoint_dir: Path, mode: str) -> str | None:
    if not checkpoint_dir.exists():
        return None

    best_path: Path | None = None
    best_loss: float | None = None

    for ckpt in checkpoint_dir.rglob("*.ckpt"):
        if mode == "yeast" and not _is_yeast_ckpt(ckpt):
            continue
        if mode == "multispecies" and _is_yeast_ckpt(ckpt):
            continue

        loss = _loss_from_ckpt_name(ckpt)
        if loss is None:
            continue

        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_path = ckpt

    return str(best_path) if best_path is not None else None


def resolve_model_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    specs = parse_model_specs(args)
    if specs:
        return specs

    auto_benchmark = _parse_boolish(args.auto_benchmark, name="--auto-benchmark")
    if not auto_benchmark:
        raise ValueError("No model inputs specified. Use checkpoint and/or predictions specs.")

    checkpoint_dir = Path(args.checkpoint_dir)
    yeast_checkpoint = args.yeast_checkpoint or best_loss_ckpt_by_mode(checkpoint_dir, mode="yeast")
    multispecies_checkpoint = args.multispecies_checkpoint or best_loss_ckpt_by_mode(
        checkpoint_dir, mode="multispecies"
    )

    if not yeast_checkpoint or not multispecies_checkpoint:
        raise ValueError(
            "Auto-benchmark is enabled but required checkpoints were not found. "
            "Set YEAST_CHECKPOINT/MULTISPECIES_CHECKPOINT or provide explicit model specs."
        )

    logger.info("Auto-enabled two-model benchmark")
    logger.info("Yeast model: %s", yeast_checkpoint)
    logger.info("Multispecies model: %s", multispecies_checkpoint)

    auto_specs = [
        {
            "label": sanitize_label(args.yeast_model_label),
            "kind": "checkpoint",
            "path": yeast_checkpoint,
        },
        {
            "label": sanitize_label(args.multispecies_model_label),
            "kind": "checkpoint",
            "path": multispecies_checkpoint,
        },
    ]

    counts: dict[str, int] = {}
    for spec in auto_specs:
        base = spec["label"]
        idx = counts.get(base, 0)
        if idx > 0:
            spec["label"] = f"{base}_{idx+1}"
        counts[base] = idx + 1

    return auto_specs


def validate_args(args: argparse.Namespace, specs: list[dict[str, str]]) -> None:
    need_checkpoint_context = any(spec["kind"] == "checkpoint" for spec in specs)

    if need_checkpoint_context:
        required_for_inference = {
            "--bigwig-dir": args.bigwig_dir,
            "--config": args.config,
        }
        missing = [flag for flag, value in required_for_inference.items() if not value]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing required arguments for checkpoint inference: {joined}")

    for spec in specs:
        if not spec["path"]:
            raise ValueError(f"Empty model path for label '{spec['label']}'")
        if not Path(spec["path"]).exists():
            raise FileNotFoundError(f"Model input does not exist: {spec['path']}")


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_species_to_run(args: argparse.Namespace, species_cfg: dict[str, Any]) -> list[str]:
    if args.species:
        requested = [s.strip() for s in args.species.split(",") if s.strip()]
    else:
        requested = list(species_cfg.keys())

    if not requested:
        raise ValueError("No species requested for evaluation.")

    return requested


def samples_per_species(args: argparse.Namespace, n_species: int) -> int:
    if args.num_samples_per_species is not None:
        return max(1, int(args.num_samples_per_species))
    return max(1, int(args.num_samples) // max(1, n_species))


def _extract_batch_metadata(batch: dict[str, Any], batch_size: int, species_name: str) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for i in range(batch_size):
        row: dict[str, Any] = {"species": species_name}
        if "chrom" in batch:
            row["chrom"] = str(batch["chrom"][i])
        if "start" in batch:
            start = batch["start"][i]
            if torch.is_tensor(start):
                start = start.item()
            if isinstance(start, np.generic):
                start = start.item()
            row["start"] = int(start)
        metadata.append(row)
    return metadata


def load_arrays_from_npz_path(
    npz_path: str,
    pred_key: str,
    target_key: str,
    metadata_key: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    with np.load(npz_path, allow_pickle=True) as data:
        if pred_key not in data.files:
            raise KeyError(f"Missing prediction key '{pred_key}' in {npz_path}")
        if target_key not in data.files:
            raise KeyError(f"Missing target key '{target_key}' in {npz_path}")
        preds = np.asarray(data[pred_key])
        targets = np.asarray(data[target_key])
        metadata = coerce_metadata(
            data[metadata_key] if metadata_key in data.files else None,
            n_samples=preds.shape[0],
        )
    return preds, targets, metadata


def build_species_test_dataset(
    args: argparse.Namespace,
    species_name: str,
    species_item: dict[str, Any],
    n_samples: int,
) -> Dataset:
    fasta_path = species_item.get("fasta")
    folds_path = species_item.get("folds")
    if not fasta_path or not folds_path:
        raise ValueError(f"Species '{species_name}' is missing fasta/folds in config.")
  
    sp_token =  species_token_dict[species_name]
    sample_ids = [str(sample) for sample in species_item.get("samples", [])]

    if args.original_input:
        num_channels = 6 
    else:
        num_channels =  4
    # MODIFICATION EHRE used to be:  build_datasets_from_folds
    datasets = build_datasets_from_folds_modified(
        bigwig_dir=args.bigwig_dir,
        fasta_path=fasta_path,
        folds_path=folds_path,
        seq_length=args.seq_length,
        train_samples=n_samples,
        val_samples=n_samples,
        seed=args.seed,
        sample_ids=sample_ids,
        entire_chromosome=True,       # IN EVALUATION WE ALWAYS WANT TOUSE ENTIRE CHROMSOME SO NO FALG NEEDED 
        species_lm_embedded=args.species_lm_embedded,
        species=species_name,    # this is  "saccharomyces_cerevisiae" folder name
        species_token=sp_token,  # and this is "_saccharomyces_cerevisiae" token
        embedding_root=args.embedding_root,
        num_channels=num_channels, # number of channles 
        skip_edges=args.skip_edges, 
    )
    test_ds = datasets.get("test")
    if test_ds is None:
        raise RuntimeError(f"No 'test' split found for species '{species_name}'")
    return test_ds


def run_model_on_dataset(
    model: Any,
    dataset: Dataset,
    device: torch.device,
    args: argparse.Namespace,
    species_name: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:

    # smae problem as before with embeddings --> ACTUALLy now no need anymore
    actual_workers = args.num_workers
    #if args.species_lm_embedded:
       # actual_workers = 0  
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=actual_workers,
    )

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_metadata: list[dict[str, Any]] = []

    logger.info("Running inference on %d windows for species '%s'", len(dataset), species_name)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {species_name}"):

            # ------ CAREFUL this is called one hot but it could also be the embedding!!!!!!

            # ---- HERE change for shuffling --> one hot is the one hot sequence or embedding
            # --> this is what I need to shuffle i think

            #one_hot.shape[0] = 4 # batch size
            #one_hot.shape[1] = 16384 # seq length along genome
            #one_hot.shape[2] = 768 #embedding dimension /length sort of

            #randperm = random permutation
            #takes a number and returns list of all integers from 0 to n - 1 a random order
            # one_hot[:, idx, :] didnt quite understand 100% but takes all rows and orders them according to the new index --> switche aorund 


            one_hot = batch["one_hot"].to(device)

            # CAREFUL got a precision missmatch when trying to run evaluation
            # this gets dtype of first parameter / (next) in model --> adjusrs one_hot to same dtype (float32 in this case)
            one_hot = one_hot.to(next(model.parameters()).dtype)
            if args.shuffle_input:
                idx = torch.randperm(one_hot.shape[1], device=one_hot.device)
                one_hot = one_hot[:, idx, :]

            target = batch["target"].to(device)
            pred = model(one_hot)

            if pred.ndim == target.ndim + 1 and pred.shape[-1] == 1:
                pred = pred.squeeze(-1)

            if pred.shape != target.shape:
                raise ValueError(f"Prediction shape {pred.shape} does not match target shape {target.shape}")

            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_metadata.extend(_extract_batch_metadata(batch, target.shape[0], species_name))

    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0), all_metadata


def run_checkpoint_inference_by_species(
    args: argparse.Namespace,
    checkpoint_path: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]]:
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    species_cfg = cfg.get("species", {})
    if not isinstance(species_cfg, dict) or not species_cfg:
        raise ValueError(f"No species found in config: {args.config}")

    species_to_run = parse_species_to_run(args, species_cfg)
    n_per_species = samples_per_species(args, len(species_to_run))

    shorkie_module = importlib.import_module("shorkie_atac")
    ShorkieATAC = getattr(shorkie_module, "ShorkieATAC")

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)
    logger.info("Loading checkpoint: %s", checkpoint_path)
    model = ShorkieATAC.load_from_checkpoint(checkpoint_path, params_file=args.params)
    model.to(device)
    model.eval()

    outputs: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
    for species_name in species_to_run:
        species_item = species_cfg.get(species_name)
        if not species_item:
            logger.warning("Species '%s' not found in config, skipping.", species_name)
            continue
        dataset = build_species_test_dataset(args, species_name, species_item, n_per_species)
        outputs[species_name] = run_model_on_dataset(model, dataset, device, args, species_name)

    if not outputs:
        raise RuntimeError("No valid species datasets were evaluated.")
    return outputs


def split_arrays_by_species(
    preds: np.ndarray,
    targets: np.ndarray,
    metadata: list[dict[str, Any]],
) -> dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]]:
    buckets: dict[str, list[int]] = {}
    for idx, row in enumerate(metadata):
        species_name = str(row.get("species", "")) if isinstance(row, dict) else ""
        if not species_name:
            continue
        buckets.setdefault(species_name, []).append(idx)

    outputs: dict[str, tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}
    for species_name, idxs in buckets.items():
        arr_idx = np.asarray(idxs, dtype=np.int64)
        outputs[species_name] = (
            preds[arr_idx],
            targets[arr_idx],
            [metadata[i] for i in idxs],
        )
    return outputs


def write_outputs(
    output_dir: Path,
    preds: np.ndarray,
    targets: np.ndarray,
    metadata: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = calculate_metrics(preds, targets)
    per_window_r = compute_per_window_pearson(preds, targets)
    save_metrics(metrics, output_dir / "metrics.json")

    if args.save_all_samples:
        save_idx = np.arange(preds.shape[0], dtype=np.int64)
    else:
        save_idx = select_export_indices(
            per_window_r,
            best_k=args.best_k,
            worst_k=args.worst_k,
            random_k=args.random_k,
            seed=args.seed,
        )

    save_evaluation_bundle(
        output_path=output_dir / "results.npz",
        preds=preds[save_idx],
        targets=targets[save_idx],
        metadata=[metadata[i] for i in save_idx],
        per_window_r=per_window_r[save_idx],
        all_per_window_r=per_window_r,
    )
    return metrics


def write_species_metrics_table(
    output_path: Path,
    model_label: str,
    overall_metrics: dict[str, float],
    species_metrics: dict[str, dict[str, float]],
) -> None:
    fieldnames = [
        "model",
        "species",
        "mse",
        "mae",
        "pearson_r",
        "spearman_r",
        "mean_per_window_pearson",
        "median_per_window_pearson",
    ]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow({"model": model_label, "species": "__overall__", **overall_metrics})
        for species_name, metrics in sorted(species_metrics.items()):
            writer.writerow({"model": model_label, "species": species_name, **metrics})


def write_per_window_metrics_table(
    output_path: Path,
    metadata: list[dict[str, Any]],
    per_window_r: np.ndarray,
) -> None:
    """Write one row per evaluated window for exact cross-model matching by coordinates."""
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["species", "chrom", "start", "per_window_pearson"],
            delimiter="\t",
        )
        writer.writeheader()
        for idx, r in enumerate(np.asarray(per_window_r).reshape(-1)):
            row = metadata[idx] if idx < len(metadata) and isinstance(metadata[idx], dict) else {}
            writer.writerow(
                {
                    "species": row.get("species", ""),
                    "chrom": row.get("chrom", ""),
                    "start": row.get("start", ""),
                    "per_window_pearson": float(r),
                }
            )


def evaluate_arrays_for_model(
    args: argparse.Namespace,
    model_label: str,
    model_source: str,
    output_dir: Path,
    preds: np.ndarray,
    targets: np.ndarray,
    metadata: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    metadata = coerce_metadata(metadata, n_samples=preds.shape[0])

    species_outputs = split_arrays_by_species(preds, targets, metadata)
    species_metrics: dict[str, dict[str, float]] = {}
    for species_name, (sp_preds, sp_targets, sp_meta) in sorted(species_outputs.items()):
        species_dir = output_dir / "species" / species_name
        metrics = write_outputs(species_dir, sp_preds, sp_targets, sp_meta, args)
        sp_per_window_r = compute_per_window_pearson(sp_preds, sp_targets)
        write_per_window_metrics_table(species_dir / "per_window_metrics.tsv", sp_meta, sp_per_window_r)
        species_metrics[species_name] = metrics

        logger.info("Model '%s' | Species '%s' metrics:", model_label, species_name)
        for metric_name, metric_value in metrics.items():
            logger.info("  %s: %.4f", metric_name, metric_value)

    overall_metrics = write_outputs(output_dir, preds, targets, metadata, args)

    metrics_blob = {
        "model": model_label,
        "source": model_source,
        "overall": overall_metrics,
        "by_species": species_metrics,
    }
    with open(output_dir / "metrics_by_species.json", "w") as handle:
        json.dump(metrics_blob, handle, indent=4)

    write_species_metrics_table(
        output_path=output_dir / "metrics_by_species.tsv",
        model_label=model_label,
        overall_metrics=overall_metrics,
        species_metrics=species_metrics,
    )

    logger.info("Model '%s' | Overall metrics:", model_label)
    for metric_name, metric_value in overall_metrics.items():
        logger.info("  %s: %.4f", metric_name, metric_value)

    return overall_metrics, species_metrics


def evaluate_checkpoint_model(
    args: argparse.Namespace,
    model_label: str,
    checkpoint_path: str,
    output_dir: Path,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    species_outputs = run_checkpoint_inference_by_species(args, checkpoint_path=checkpoint_path)

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_meta: list[dict[str, Any]] = []
    for preds, targets, metadata in species_outputs.values():
        all_preds.append(preds)
        all_targets.append(targets)
        all_meta.extend(metadata)

    return evaluate_arrays_for_model(
        args=args,
        model_label=model_label,
        model_source=checkpoint_path,
        output_dir=output_dir,
        preds=np.concatenate(all_preds, axis=0),
        targets=np.concatenate(all_targets, axis=0),
        metadata=all_meta,
    )


def evaluate_npz_model(
    args: argparse.Namespace,
    model_label: str,
    npz_path: str,
    output_dir: Path,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    preds, targets, metadata = load_arrays_from_npz_path(
        npz_path=npz_path,
        pred_key=args.pred_key,
        target_key=args.target_key,
        metadata_key=args.metadata_key,
    )
    return evaluate_arrays_for_model(
        args=args,
        model_label=model_label,
        model_source=npz_path,
        output_dir=output_dir,
        preds=preds,
        targets=targets,
        metadata=metadata,
    )


def write_comparison_outputs(
    output_dir: Path,
    comparison: dict[str, dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "comparison_metrics.json", "w") as handle:
        json.dump(comparison, handle, indent=4)

    long_fieldnames = [
        "model",
        "species",
        "source",
        "mse",
        "mae",
        "pearson_r",
        "spearman_r",
        "mean_per_window_pearson",
        "median_per_window_pearson",
    ]
    with open(output_dir / "comparison_metrics.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fieldnames, delimiter="\t")
        writer.writeheader()
        for model_label, blob in sorted(comparison.items()):
            writer.writerow(
                {
                    "model": model_label,
                    "species": "__overall__",
                    "source": blob["source"],
                    **blob["overall"],
                }
            )
            for species_name, metrics in sorted(blob["by_species"].items()):
                writer.writerow(
                    {
                        "model": model_label,
                        "species": species_name,
                        "source": blob["source"],
                        **metrics,
                    }
                )

    species_names = sorted(
        {
            species_name
            for blob in comparison.values()
            for species_name in blob["by_species"].keys()
        }
    )
    matrix_fieldnames = ["model", "__overall__", *species_names]
    with open(output_dir / "pearson_r_matrix.tsv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=matrix_fieldnames, delimiter="\t")
        writer.writeheader()
        for model_label, blob in sorted(comparison.items()):
            row: dict[str, Any] = {
                "model": model_label,
                "__overall__": blob["overall"].get("pearson_r", np.nan),
            }
            for species_name in species_names:
                row[species_name] = blob["by_species"].get(species_name, {}).get("pearson_r", np.nan)
            writer.writerow(row)


def _should_run_plotting(mode: str, is_multi_model: bool) -> bool:
    mode_normalized = mode.strip().lower()
    if mode_normalized == "auto":
        return is_multi_model
    return _parse_boolish(mode_normalized, name="--run-plotting")


def run_comparison_plotting(args: argparse.Namespace, output_dir: Path) -> None:
    comparison_json = output_dir / "comparison_metrics.json"
    if not comparison_json.exists():
        logger.info("No comparison JSON found at %s; skipping plotting.", comparison_json)
        logger.info("Run multi-model evaluation to generate comparison plots.")
        return

    plots_root = Path(args.plots_dir) if args.plots_dir else output_dir / "plots"
    comparison_output_dir = plots_root / "comparison"
    comparison_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent / "plot_comparison.py")),
        "--comparison-json",
        str(comparison_json),
        "--evaluation-root",
        str(output_dir),
        "--output-dir",
        str(comparison_output_dir),
        "--track-top-n",
        str(args.comparison_track_top_n),
    ]
    if args.comparison_species:
        cmd.extend(["--species", args.comparison_species])

    logger.info("Running comparison plotting")
    subprocess.run(cmd, check=True)
    logger.info("Completed evaluation + comparison plotting. Results: %s, plots: %s", output_dir, comparison_output_dir)


def main() -> int:
    args = parse_args()
    specs = resolve_model_specs(args)
    validate_args(args, specs)

    output_dir = Path(args.output_dir)

    if len(specs) == 1:
        spec = specs[0]
        logger.info("Evaluating model '%s' (%s)", spec["label"], spec["kind"])
        if spec["kind"] == "checkpoint":
            evaluate_checkpoint_model(args, spec["label"], spec["path"], output_dir)
        else:
            evaluate_npz_model(args, spec["label"], spec["path"], output_dir)

        if _should_run_plotting(args.run_plotting, is_multi_model=False):
            run_comparison_plotting(args, output_dir)
        else:
            logger.info("Completed evaluation only. Results: %s", output_dir)
        return 0

    comparison: dict[str, dict[str, Any]] = {}
    for spec in specs:
        model_label = spec["label"]
        model_output_dir = output_dir / "models" / model_label
        logger.info("Evaluating model '%s' (%s)", model_label, spec["kind"])

        if spec["kind"] == "checkpoint":
            overall_metrics, species_metrics = evaluate_checkpoint_model(
                args, model_label, spec["path"], model_output_dir
            )
        else:
            overall_metrics, species_metrics = evaluate_npz_model(
                args, model_label, spec["path"], model_output_dir
            )

        comparison[model_label] = {
            "source": spec["path"],
            "kind": spec["kind"],
            "overall": overall_metrics,
            "by_species": species_metrics,
        }

    write_comparison_outputs(output_dir, comparison)
    logger.info("Cross-model comparison written to %s", output_dir)

    if _should_run_plotting(args.run_plotting, is_multi_model=True):
        run_comparison_plotting(args, output_dir)
    else:
        logger.info("Completed evaluation only. Results: %s", output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
