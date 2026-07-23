"""
Linear-regression evaluation workflow (sort of a mirror of train_benchmark_regression.py).

fold test chroms -> tile_chromosomes -> EvaluationPositionDataset -> predict per window
-> stack to (n_windows, seq_length) -> save results.npz the same way shorkie does --> all done then 

The model is the pickle we trainend (make sure to train before) 
We read feature_window + kmer_len out of it so evaluation uses the exact same setzp the model was fit with.

Training flattened all windows into one (X, y) because fitting is position by position and doesnt care about window boundaries. 
here we  cant flatten cuz we need per-window Pearson and that needs window i's preds vs window i's targets
--> so keep the windows as separate rows aka predict each window instead of all at once 
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

from evaluate_benchmark_dataset import tile_chromosomes, EvaluationPositionDataset

import sys
sys.path.append("/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline")

from shorkie_atac.evaluation_utils import (
    calculate_metrics,
    compute_per_window_pearson,
    save_evaluation_bundle,
    save_metrics,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Lin Reg baseline full chromosome.")
    p.add_argument("--feature", required=True, choices=["gc", "gc_kmer", "embeddings"])
    p.add_argument("--model", required=True, help="The fitted baseline pkl.")
    p.add_argument("--fasta", required=True)
    p.add_argument("--bigwig-dir", required=True)
    p.add_argument("--output-dir", required=True, help="Where results.npz and metrics.json go")
    p.add_argument("--chromosomes", nargs="*", default=None,
                   help="TEST chroms to evaluate on")
    p.add_argument("--seq-length", type=int, default=16384,
                   help="Window length, match the shorkie run (default 16384)")
    p.add_argument("--embedding-root", default=None)
    p.add_argument("--species", default=None)
	# here we need this because chrombpnet doesnt predict edges
    p.add_argument("--skip-edges", type=int, default=0, help="Shift tiling to [skip_edges, chrom_len - skip_edges). ")
    return p.parse_args()


def save_results(output_dir, preds, targets, metadata):
    """Write results.npz and metrics.json the same way shorkies write_outputs does"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # metrics over every window
    metrics = calculate_metrics(preds, targets)
    save_metrics(metrics, output_dir / "metrics.json")

    per_window_r = compute_per_window_pearson(preds, targets)

    # we keep all windows here (baselines are cheap), same npz keys as shorkie
    save_evaluation_bundle(
        output_path=output_dir / "results.npz",
        preds=preds,
        targets=targets,
        metadata=metadata,
        per_window_r=per_window_r,
        all_per_window_r=per_window_r,
    )

    logger.info("Saved results to %s", output_dir)
    for name, value in metrics.items():
        logger.info("%s: %.4f", name, value)


def main():
    args = parse_args()

    # load the fitted model and the geometry it was trained with
    with open(args.model, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    feature_window = saved["feature_window"]
    kmer_len = saved["kmer_len"]
    logger.info("Loaded model (feature=%s, feature_window=%d, kmer_len=%d)",
                args.feature, feature_window, kmer_len)

    bigwig_paths = sorted(str(p) for p in Path(args.bigwig_dir).glob("*.bw"))
    if not bigwig_paths:
        raise FileNotFoundError(f"No .bw files in {args.bigwig_dir}")

    # tile the test chromosomes into seq_length windows (same windows shorkie sees)
    logger.info("Tiling chromosomes")
    regions = tile_chromosomes(args.fasta, args.chromosomes, args.seq_length,skip_edges=args.skip_edges)
    logger.info("Got %d windows", len(regions))

    # build the eval dataset (per window features + targets)
    logger.info("Building dataset (feature=%s)", args.feature)
    dataset = EvaluationPositionDataset(
        regions=regions,
        feature=args.feature,
        bigwig_paths=bigwig_paths,
        fasta_path=args.fasta,
        seq_length=args.seq_length,
        feature_window=feature_window,
        kmer_len=kmer_len,
        embedding_root=args.embedding_root,
        species=args.species,
    )

    # predict each window on its own and keep the windows separate (do NOT flatten)
    all_preds = []
    all_targets = []
    all_metadata = []
    for features, target, (chrom, start) in dataset.windows:
        pred = model.predict(features)            # (seq_length,)
        all_preds.append(pred)
        all_targets.append(target)
        all_metadata.append({"species": args.species, "chrom": chrom, "start": int(start)})

    # stack along the window axis -> (n_windows, seq_length)
    preds = np.stack(all_preds, axis=0)
    targets = np.stack(all_targets, axis=0)
    logger.info("preds shape=%s, targets shape=%s", preds.shape, targets.shape)

    save_results(args.output_dir, preds, targets, all_metadata)
    logger.info("Done.")


if __name__ == "__main__":
    main()