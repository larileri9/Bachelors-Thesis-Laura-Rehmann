from pathlib import Path

import numpy as np
import pyBigWig
from Bio import SeqIO
import torch
from pysam import FastaFile
from torch.utils.data import Dataset

# reuse the methods from the other dataset 
from train_benchmark_dataset import revcomp, kmer_embedding_dict, observed_signal


# ------ tile test chroms ---------

# this is the evaluation version of read_bed_regions
# here we tile the chromosome into the same windows shorkie produces since we are sort of dependent on that for like comparable window metrics

def tile_chromosomes(fasta_path, chromosomes, seq_length, skip_edges=0):
    regions = []

    # only need the lengths here so we just read the fasta index
    fasta = FastaFile(str(fasta_path))
    # the chrom names should match ehre
    chrom_lengths = dict(zip(fasta.references, fasta.lengths))
    fasta.close()

    # we always want test chroms rather no regions and error than some default 
    for chrom in chromosomes:
        if chrom not in chrom_lengths:
            continue
        chrom_length = chrom_lengths[chrom]

        # these are the bounds if we need to skip because of chrombpnet
        left = skip_edges
        right = chrom_length - skip_edges
        #too short for even one window, skip (same as shorkie) (shoudlnt ahppen tho)
        if right - left < seq_length:
            continue

        position = left
        while position < right:
            # normal case, window fits
            if position + seq_length <= right:
                regions.append((chrom, position))
                position = position + seq_length
            # last window would overhang, push it back so it still fits --> we do the same in shorkie
            # just so evrything si comaprable
            else:
                start = right - seq_length
                regions.append((chrom, start))
                position += seq_length

    return regions


# --------------- per position feature tracks ---------------

# same idea as the training gc_track but here we predict EVERY position in the window
# the seq passed in already has a half-window flank on each side --> this way we dont only comput in region 
# predict_start is where 
# predict_length = how many positions we predict (the whole seq_length window)
def gc_track(seq, feature_window, predict_start, predict_length):
    seq = seq.upper()
    half = feature_window // 2

    is_gc = np.array([1.0 if base in "GC" else 0.0 for base in seq], dtype=np.float32)

    gc_fraction = np.zeros((predict_length, 1), dtype=np.float32)
    # out row is just our i agahin
    for out_row in range(predict_length):
        center = predict_start + out_row  # this is the position inside the seq we predict for
        window_start = max(0, center - half)  # clamp to make sure we dont go over the edge of the chrom 
        window_end = min(len(is_gc), center + half) # here the thing is : is_gc is the sequence right, and if the chromosme is 
        local = is_gc[window_start:window_end]          # shorter than like the window we woudl compute this clamps it because sequence is too shrot then
        gc_fraction[out_row, 0] = local.mean()
    return gc_fraction


#same idea as the training kmer_track  --> agai seq already has that window added on top
def kmer_track(seq, feature_window, kmer_dict, kmer_len, predict_start, predict_length):
    seq = seq.upper()
    half = feature_window // 2
    num_kmers = len(kmer_dict)
    length = len(seq)

    # one hot which canonical kmer starts at each position again 
    counts = np.zeros((length, num_kmers), dtype=np.float32)
    for position in range(length - kmer_len + 1):
        current_kmer = seq[position:position + kmer_len]
        canonical = min(current_kmer, revcomp(current_kmer))
        column = kmer_dict.get(canonical)   
        if column is not None:
            counts[position, column] = 1.0 # append if column is not noen

    # average the kmer one hots over the feature window centered at each predicted position
    frequencies = np.zeros((predict_length, num_kmers), dtype=np.float32)
    for out_row in range(predict_length):
        center = predict_start + out_row
        window_start = max(0, center - half)
        window_end = min(length, center + half)
        local = counts[window_start:window_end]
        frequencies[out_row] = local.mean(axis=0)
    return frequencies


# ----------------- the dataset -----------------

# evaluation version of BaselinePositionDataset
# builds self.windows the same way
# but here we go over the tiled test windows and over the full seq_length window instead of the central slice and ony the peak centered oness
class EvaluationPositionDataset(Dataset):
    def __init__(self,
                regions,
                feature,
                bigwig_paths,
                fasta_path,
                seq_length,
                feature_window,
                kmer_len=3,
                log_transform=True,
                embedding_root=None,
                species=None):

        self.feature = feature
        self.kmer_dict = kmer_embedding_dict(kmer_len) if feature == "gc_kmer" else None

        # load the genome once so we can just liek slice sequences (same as the training dataset)
        chrom_seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")}
        bw_handles = [pyBigWig.open(str(p)) for p in bigwig_paths]

        # cachinga gain
        if feature == "embeddings":
            emb_dir = Path(embedding_root) / species
            emb_cache = {}

        half = feature_window // 2

        self.windows = []

        for chrom, start in regions:
            end = start + seq_length
            # safety check, tiling should already keep us inside the chromosome --> never too littltle checks (eliminate those updfront)
            if start < 0 or end > len(chrom_seqs[chrom]):
                continue

            if feature == "gc" or feature == "gc_kmer":
                # fetch the window PLUS a half-window flank on each side
                # so every predicted position has a full feature window of context becuase we dont predict just center so we extend sequence instead
                fetch_start = max(0, start - half) #safety / needed
                fetch_end = min(len(chrom_seqs[chrom]), end + half) # safety
                seq = chrom_seqs[chrom][fetch_start:fetch_end] # get the slice
                predict_start = start - fetch_start   # where inside the seq does the actual rpediciton begin?

                if feature == "gc":
                    features = gc_track(seq, feature_window, predict_start, seq_length)  # (seq_length, 1) as ouptut
                else:
                    features = kmer_track(seq, feature_window, self.kmer_dict, kmer_len,
                                          predict_start, seq_length)  # (seq_length, n) as output n dependent on kemr lengt but 32
            else:  # embeddings with caching on cpu
                if chrom not in emb_cache:
                    emb_cache[chrom] = torch.load(emb_dir / f"{chrom}.pt", map_location="cpu")
                emb = emb_cache[chrom]
                features = np.asarray(emb[start:end], dtype=np.float32)  # (seq_length, 768) as output

            target = observed_signal(bw_handles, chrom, start, end, log_transform)  # (seq_length,) (target is ehe observed signal again)

            # keep every window separate, one entry per window so do NOT stack here --> instead list of windows
            self.windows.append((features, target, (chrom, start)))

        for bw in bw_handles:
            bw.close()