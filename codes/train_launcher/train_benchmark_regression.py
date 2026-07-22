"""
This is the linear regression baseline workflow

bed files -> read_bed_regions -> Baseline Position Dataset -> stack  -> Linear Regression -> pickle.

Chrombpnet input sizes 1216 / 608 are hardcoded in main, so this needs to be changed by hand

Most of the stuff is constructed after the shorkie workflow like logger and dataset loader / builder
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from train_benchmark_dataset import read_bed_regions, BaselinePositionDataset
 
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# I think this is horroible for the memory but not sure how to shortcut efficiently without breaking it
def stack_windows(dataset):
    # Flatten the dataset per-window blocks into one (X, y) for sklearn
    # hope I got the stacking right here with _ we can document what we throw out and what we keep 
    X = np.concatenate([feats for feats, _target, _id in dataset.windows], axis=0)
    y = np.concatenate([target for _feats, target, _id in dataset.windows], axis=0)
    return X, y


def parse_args():
    p = argparse.ArgumentParser(description="Fit the linear regression baseline.")
    p.add_argument("--feature", required=True, choices=["gc", "gc_kmer", "embeddings"])
    p.add_argument("--fasta", required=True)
    p.add_argument("--bigwig-dir", required=True)
    p.add_argument("--peak-bed", required=True)
    p.add_argument("--nonpeak-bed", required=True)
    p.add_argument("--output", required=True, help="Where to save the model")
    p.add_argument("--chromosomes", nargs="*", default=None,
                   help="train chroms")
    p.add_argument("--kmer-len", type=int, default=3)
    p.add_argument("--embedding-root", default=None)
    p.add_argument("--species", default=None)
    # this is for using chrombp bias corrected bigwigs
    p.add_argument("--edge-trim", type=int, default=0, help="Skip windows within this many bp of each chromosome end.")
    return p.parse_args()


def main():
    args = parse_args()

    # hardcoded here 
    input_length_cbp = 1216       # chrombpnet nonpeak width
    output_length_cbp = 608    # local gc / k-mer window (-> central 608 predicted)
    
    # get all the bigwigs 
    bigwig_paths = sorted(str(p) for p in Path(args.bigwig_dir).glob("*.bw"))
    if not bigwig_paths:
        raise FileNotFoundError(f"No .bw files in {args.bigwig_dir}")

    logger.info("Reading bed regions")
    regions = read_bed_regions(
        peak_bed=args.peak_bed,
        non_peak_bed=args.nonpeak_bed,
        region_len=input_length_cbp,
        chromosomes=args.chromosomes,
    )
    logger.info("Got %d regions", len(regions))

    logger.info("Building dataset (feature=%s)", args.feature)
    dataset = BaselinePositionDataset(
        regions=regions,
        feature=args.feature,
        bigwig_paths=bigwig_paths,
        fasta_path=args.fasta,
        region_len=input_length_cbp,
        feature_window=output_length_cbp,
        kmer_len=args.kmer_len,
        embedding_root=args.embedding_root,
        species=args.species,
        edge_trim=args.edge_trim,
    )

    X, y = stack_windows(dataset)
    # just checking
    logger.info("X shape=%s, y shape=%s", X.shape, y.shape)

    logger.info("Fitting Linear Regression")
    model = LinearRegression()
    model.fit(X, y)

    # same data R^2 --> optimistic set of estimation--> checking if gc and gc kmer and embeddings actually exhibit this on trining data
    logger.info("Training R^2 (optimistic): %.4f", model.score(X, y))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({
            "model": model,
            "feature": args.feature,
            "region_len": input_length_cbp,
            "feature_window": output_length_cbp,
            "kmer_len": args.kmer_len,
        }, f)
    logger.info("Saved model to %s", out)


if __name__ == "__main__":
    main()