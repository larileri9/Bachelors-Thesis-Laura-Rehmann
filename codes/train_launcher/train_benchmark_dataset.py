from itertools import product

import numpy as np
import pyBigWig
from Bio import SeqIO
from torch.utils.data import Dataset
from pathlib import Path
import torch


# ---- helpers here ------

# revcomp because we want to collapse canonical kmers+ 

def revcomp(seq):
    complement = str.maketrans("ATGC", "TACG")
    return seq.translate(complement)[::-1] 

# dict which makes sure the index of each kmer in its vector is standardized (canonical ofc)
def kmer_embedding_dict(kmer_length):
    kmer_dict = {}
    index = 0
    # makes all combos then get the alphabetically minimal one
    for kmer in ("".join(c) for c in product("ATGC", repeat=kmer_length)):
        canonical = min(kmer, revcomp(kmer))
        if canonical not in kmer_dict:
            kmer_dict[canonical] = index
            index += 1
    return kmer_dict


# read the bed region files
# region len is specified here we gat that from the main file
# pass chromosmes as a list when we do training (validation is left out here)
# and jjust append the regiuon starts
def read_bed_regions(peak_bed, non_peak_bed, region_len, chromosomes=None):
    regions = []
    for bed in (peak_bed, non_peak_bed):
        with open(bed) as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.split("\t")
                chrom = parts[0]
                if chromosomes is not None and chrom not in chromosomes:
                    continue
                summit = int(parts[1]) + int(parts[9])   #narrowPeak col 10 is summit offset
                region_start = summit - region_len // 2
                regions.append((chrom, region_start))
    return regions

# ------ per-position feature tracks -----

# gc content in each sub window of a certain length over a sequence
def gc_track(seq, window):
    seq = seq.upper()
    length = len(seq) # 1216
    half = window // 2  # 304
    predict_margin = (length - window) // 2 # thats teh offset

    is_gc = np.array([1.0 if base in "GC" else 0.0 for base in seq], dtype=np.float32)

    # only the central positions we actually predict are initialized
    gc_fraction = np.zeros((length - 2 * predict_margin, 1), dtype=np.float32)

    for out_row, position in enumerate(range(predict_margin, length - predict_margin)):
        window_start = position - half          
        window_end = position + half          
        local = is_gc[window_start:window_end] # get the slice
        gc_fraction[out_row, 0] = local.mean() # gte the mean
    return gc_fraction


def kmer_track(seq, window, kmer_dict, kmer_len):
    seq = seq.upper()
    length = len(seq) # 1216
    num_kmers = len(kmer_dict)
    half = window // 2  # 304
    predict_margin = (length - window) // 2 #offset, same as gc_track

    # have a 2 d matrix where len is kmer start and height is kmer index 
    counts = np.zeros((length, num_kmers), dtype=np.float32)
    # compute the kmer starting at each position
    for position in range(length - kmer_len + 1):
        current_kmer = seq[position:position + kmer_len]
        canonical = min(current_kmer, revcomp(current_kmer))
        column = kmer_dict.get(canonical)   # None if it contains an N
        if column is not None:
            counts[position, column] = 1.0 # append the kmer to its starting poisition

    # now the actual frequenceis are not seq length but the window we predict and num kmers
    frequencies = np.zeros((length - 2 * predict_margin, num_kmers), dtype=np.float32)
    for out_row, position in enumerate(range(predict_margin, length - predict_margin)):
        window_start = position - half
        window_end = position + half 
        local = counts[window_start:window_end]   # slice out the posiitons we need 
        frequencies[out_row] = local.mean(axis=0) # average each k-mer column / average along the sequence axis 
    return frequencies

# average replicates, log1p per base  --> like shorkie do internally here

def observed_signal(bw_handles, chrom, start, end, log_transform=True):
    #from shorkie same per position averaging
    signals = []
    for bw in bw_handles:
        vals = np.array(bw.values(chrom, int(start), int(end)), dtype=np.float32)
        vals = np.nan_to_num(vals, nan=0.0)
        signals.append(vals)
    signal = np.mean(signals, axis=0)     # average along sequence array 
    signal = np.clip(signal, 0, None) # clip signal here for tobias biwgiwgs (none is upper limit 0 ower one)
    if log_transform:
        signal = np.log1p(signal)
    return signal                 # array with averaged signal


# ----- dataset class ----

class BaselinePositionDataset(Dataset):
    def __init__(self, 
                regions,   
                feature, 
                bigwig_paths, 
                fasta_path,
                region_len, 
                feature_window, 
                kmer_len=3, 
                log_transform=True,
                embedding_root=None, 
                species=None,
                edge_trim=0):

        self.feature = feature # this is the inly thing that rly matters everything else is static
        self.kmer_dict = kmer_embedding_dict(kmer_len) if feature == "gc_kmer" else None # just if we need it


        chrom_seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")}
        bw_handles = [pyBigWig.open(str(p)) for p in bigwig_paths]

        # embeddings are tensor per chromosome, cache them if we need a chrom later
        # --> we only lod once
        if feature == "embeddings":
            emb_dir = Path(embedding_root) / species 
            emb_cache = {}

        margin = (region_len - feature_window) // 2  # 304, the central 608 winodw

        self.windows = []

        #loop over the regions not the best but yeah
        for chrom, region_start in regions:
            # current region
            region_end = region_start + region_len
            # sanity chekc
            if region_start < 0 or region_end > len(chrom_seqs[chrom]):
                continue
            


            # here is check for using bias corrected chrobpnet bw
            # edgce cant be predicted --> cut anything that overlaps that regin
            if edge_trim > 0:
                length = len(chrom_seqs[chrom])
                pred_start = region_start + margin          
                pred_end = region_end - margin              
                if pred_start < edge_trim or pred_end > length - edge_trim:
                    continue

            
            if feature == "gc":
                seq = chrom_seqs[chrom][region_start:region_end]
                features = gc_track(seq, feature_window) # (608, 1)
            elif feature == "gc_kmer":
                seq = chrom_seqs[chrom][region_start:region_end]
                gc = gc_track(seq, feature_window)   
                kmers = kmer_track(seq, feature_window, self.kmer_dict, kmer_len)  # (608, 32)
                features = np.concatenate([gc, kmers], axis=1)    # (608, 33)
            else:  # embeddings
                if chrom not in emb_cache:
                    #load if we need it and its not there
                    emb_cache[chrom] = torch.load(emb_dir / f"{chrom}.pt", map_location="cpu")
                
                emb = emb_cache[chrom]
                #features = np.asarray(emb[region_start:region_end], dtype=np.float32)  # (1216, 768) --> full seq
                #features = features[margin:region_len - margin] # (608, 768) crop to the cenetr window we want (could do other way round but to be sure)
                # di directly and save memory 
                features = np.asarray(emb[region_start + margin : region_end - margin], dtype=np.float32)

            target = observed_signal(bw_handles, chrom, region_start, region_end, log_transform)
            target = target[margin:region_len - margin]  # (608,) same window cropped pout of the tragets 

            # big list of windows taht are given back and learned
            self.windows.append((features, target, (chrom, region_start)))

        for bw in bw_handles:
            bw.close()