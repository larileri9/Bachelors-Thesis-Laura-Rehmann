from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyBigWig
import torch
from pysam import FastaFile
from torch.utils.data import Dataset
# maybe add this here soem day but for now 

logger = logging.getLogger(__name__)

# DNA encoding: A=0, C=1, G=2, T=3
_DNA_TO_IDX_4  = {
    "A": 0, "a": 0,
    "C": 1, "c": 1,
    "G": 2, "g": 2,
    "T": 3, "t": 3,
}


_DNA_TO_IDX_6 = {
    "A": 0, "a": 0,
    "C": 1, "c": 1,
    "G": 2, "g": 2,
    "T": 3, "t": 3,
    "N": 4, "n": 4,
    # channel 5 is mask and always 0 because we dont ask during fine tuning
}

def one_hot_encode(seq: str, num_channels: int = 4) -> np.ndarray: # we added num channels with default 4 as additional param
    """One-hot encode a DNA sequence. Unknown bases get all-zeros.

    Returns:
        ndarray of shape (seq_len, 4).
    """
    arr = np.zeros((len(seq), num_channels), dtype=np.float32)
    lookup = _DNA_TO_IDX_4 if num_channels == 4 else _DNA_TO_IDX_6
    for i, base in enumerate(seq):
        idx = lookup.get(base)
        if idx is not None:
            arr[i, idx] = 1.0
    return arr



# method for creating an embedding for one sequence (or more so one sample) --> moved outside i accidtentaly did insideinit first
def embed_sequence(seq, species, embedder, tokenizer):
    
    #species predefined in here loses flexibility  -->  also no species is technically explicetly passed so we need to pass it 
        # what not to do: species = "_saccharomyces_cerevisiae" 

        # this is taken form my linear regression its a bit ugly but gets job done --> need to rework
    tokenized_input = tokenizer(seq, species) # create one tokenized input
        #input_id = torch.stack(torch.tensor(tokenized_input["input_ids"])) # stack all those input ids ...? i dont eben think we need to stack
    input_id = torch.tensor(tokenized_input["input_ids"])
        #batch_size = 1
        #for i in range(0, 1, batch_size): # batch size steps --> its alll for nothing its one embedding
        #batch = input_ids[i:i+batch_size].to(embedder.device)
        # nevermind i think this should simply work like that? 
    with torch.no_grad():
        # try if this is okay
        emb = embedder.embed(input_id.to(embedder.device))
        emb = emb[0]
            # move back to CPU for d ataloader
        return emb.cpu()

# i will keep a modified dataset to call, with a flag i find this to be more organized

class ATACSeqDatasetModified(Dataset):
    """Dataset that yields (one_hot_dna, atac_signal) pairs.

    Args:
        bigwig_paths: List of paths to bigwig files.
        fasta_path: Path to the reference genome FASTA file.
        chromosomes: Which chromosomes to sample from.
        seq_length: Length of each genomic window (must match model input).
        samples_per_epoch: Number of random windows per epoch.
        seed: Random seed for reproducibility.
        log_transform: Whether to apply log1p transform to the signal.
    """

    def __init__(
        self,
        bigwig_paths: list[str | Path],
        fasta_path: str | Path,
        chromosomes: list[str],
        seq_length: int = 16384,
        samples_per_epoch: int = 10000,
        seed: int = 42,
        log_transform: bool = True,
        entire_chromosome = False, # initialize with false 
        species_lm_embedded = False, # initialize with false 
        species_token = None,
        species = None, # initialize with none 
        embedding_root = None, #initialized with none
        num_channels: int = 4, #ä default is just one hot encode
        peak_centered = False,
        peak_path = None,
        non_peak_path = None,
        edge_trim: int = 0,   # normally we use all
        skip_edges: int = 0,    # for eval
    ):
        super().__init__()
        self.bigwig_paths = [str(p) for p in bigwig_paths]
        self.fasta_path = str(fasta_path)
        self.chromosomes = chromosomes
        self.seq_length = seq_length
        self.samples_per_epoch = samples_per_epoch
        self.log_transform = log_transform
        self.entire_chromosome = entire_chromosome
        self.species_lm_embedded = species_lm_embedded
        self.species_token = species_token
        self.species = species
        self.embedding_root = embedding_root 
        self.num_channels = num_channels
        self.peak_centered = peak_centered
        self.peak_path = peak_path
        self.non_peak_path = non_peak_path
        self.edge_trim = edge_trim
        self.skip_edges = skip_edges

        # Build chromosome --> length mapping from the fasta
        #  ----- this thros out chroms shorter than window size
        fa = FastaFile(self.fasta_path)
        self.chrom_lengths: dict[str, int] = {}
        for chrom, length in zip(fa.references, fa.lengths):
            if chrom in chromosomes:
                if length >= seq_length:
                    self.chrom_lengths[chrom] = length
                else:
                    logger.warning(
                        "Chromosome %s (length=%d) is shorter than seq_length=%d, skipping.",
                        chrom, length, seq_length,
                    )
        fa.close()

        if not self.chrom_lengths:
            raise ValueError(
                f"No valid chromosomes found. Requested: {chromosomes}"
            )

       



        # Build sampling weights proportional to chromosome length

        # ---- chroms is a list of all chroms with chrom 1 at  index 0 and chrom x at index x - 1 
        # this is a list of only the chroms we are actually using in that specific split so train or test or val
        self._chroms = list(self.chrom_lengths.keys())
        print(self._chroms)
        # lengths are aslo sored in the same manner which is good for iteration later 
        lengths = np.array([self.chrom_lengths[c] for c in self._chroms], dtype=np.float64)
        self._chrom_weights = lengths / lengths.sum()

         # ------ get embedder upfron if needed
        if self.species_lm_embedded:
            if not self.species:
                raise ValueError(
                    f"No species found but embeddings requested -> pass species"
            )
            if not self.species_token:
                raise ValueError(
                    f"No species token found but embeddings requested -> pass species token"
            )
            # DONT use token here, token solely for creating embeddings --> token only relevant if we add a generate missing flag
            self.embedding_dir = Path(self.embedding_root) / self.species # this just wokrs this is fantastic not os 

            # contorl if dir for that species exist
            if not self.embedding_dir.exists():
                raise ValueError(f"irectory for embeddings does not exist: {self.embedding_dir}"
                )
            #check if we have all chromosomes for the split we are doing
            for chrom in chromosomes:
                emb_path = self.embedding_dir / f"{chrom}.pt"
                if not emb_path.exists():
                    raise FileNotFoundError(f"Missing embedding for this chromosome: {emb_path}")

            # we checked if all chrom embeddings exist
            # now load upfront ebcause oadding each time in get item wouldnt be smart
            #  create a dict like chrom lengths was created 
            self._chrom_embeddings: dict[str, torch.Tensor] = {}
            for chrom in self._chroms:
                #chrom string  maps to loaded tensor --> can be used in get item depending on crom 
                self._chrom_embeddings[chrom] = torch.load(
                    self.embedding_dir / f"{chrom}.pt",
                    map_location="cpu"
                )

            logger.info("Loaded embedding for all chroms") #i dint understand that % sign yet --> read up on that!

            #self._embedder = embedding_generator.SpeciesLMV1_embedder(model_path="/s/project/denovo-prosit/JohannesHingerl/BERTADN/final_models/species_upstream_1000_k1/",kmer_size=1,device="cuda")
            #self._tokenizer = self._embedder.tokenize_func()
            
        '''
        modification needs to hapen here 
        --> seed we can leave i think that wont change anything, we need to make a differrent methiod here
        '''
        # Pre-generate random positions for this epoch
        self._rng = np.random.RandomState(seed)
        #self._regenerate_positions()
        # ------------------- sampling change is here!!! -------------------

        if self.entire_chromosome:
            self._generate_chrom_covering_positions()
        elif self.peak_centered:
            self._generate_peak_centered_positions()
        else:
            self._regenerate_positions()

        # -------------------------------------------------------------------
        # reset the number of samples per ecpoch --> otherwise we are getting indexing errors ??
        # when we do this in intit i think the len method is still valid?
        self.samples_per_epoch = len(self._positions)

        # lazy opened file handles (perworker once)
        self._fa: FastaFile | None = None
        self._bws: list[pyBigWig.pyBigWig] | None = None
        # % logging should work copied this
        logger.info(
            "ATACSeqDatasetTest: %d bigwigs, %d chromosomes, seq_length=%d, samples=%d",
            len(self.bigwig_paths), len(self._chroms), seq_length, samples_per_epoch,
        )

        

    '''
    new method to jsut slide over it
    '''
    def _generate_chrom_covering_positions(self):
        # initae position list
        self._positions: list[tuple[str, int]] = []
        # go over just each chrom in the chrom list we made from the pssed chroms for each split but only rly for test
        for c in self._chroms:
            print("chrom: ", c)
            # change this to left 
            left = self.skip_edges
            right = self.chrom_lengths[c] - self.skip_edges
            # check if one full window
            if right - left < self.seq_length:
                continue
            i = left
            #stop for vhrom if were over seq length 
            while i < right:
                # if window would not overhang chrom length justa ppend and slide window
                if i + self.seq_length <= right:
                    chrom = c
                    start = i
                    i = i + self.seq_length
                    self._positions.append((chrom, start))
                # if it overhangs go back append taht last window
                elif i + self.seq_length > right:
                    chrom = c
                    start = right - self.seq_length
                    i = i + self.seq_length
                    print(f"we have ", len(self._positions), " psoitions right now")
                    self._positions.append((chrom, start))

    def _generate_peak_centered_positions(self):
        # initae position list
        self._positions: list[tuple[str, int]] = []
        # here we need the half to center on peaks
        half = self.seq_length // 2
        # go over just each chroim in the chrom list we made from the passed chroms for each split but only rly for test
        
        #loop over paths so we keep one open at a time and smply run lines 
        for bed_path in [self.peak_path, self.non_peak_path]:        
            with open(bed_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.split("\t")
                    chrom = parts[0]
                    # keep only chroms that belong to this split, so if they are in chrom lengths 
                    if chrom not in self._chroms:
                        continue
                    bed_start = int(parts[1])
                    summit_offset = int(parts[9])   # 10th column so the peak center
                    summit = bed_start + summit_offset # thats the genomic center
                    start = summit - half            #center the window on the summit (just save start seq len will do the rest)

                    # edge filtering, entire chrom doesnt need to filter starta nd the other one cuts it from the possible indices
                    if start < 0:
                        continue
                    if start + self.seq_length > self.chrom_lengths[chrom]:
                        continue

                    self._positions.append((chrom, start))



    def _regenerate_positions(self) -> None:
        """Pre-sample random genomic positions for the epoch."""
        chroms = self._rng.choice(
            len(self._chroms), size=self.samples_per_epoch, p=self._chrom_weights
        )
        self._positions: list[tuple[str, int]] = []
        for ci in chroms:
            chrom = self._chroms[ci]
            max_start = self.chrom_lengths[chrom] - self.seq_length
            start = self._rng.randint(0, max_start + 1)
            self._positions.append((chrom, start))

    def _open_files(self) -> None:
        """Open file handles (called lazily, safe for multi-worker DataLoader)."""
        if self._fa is None:
            self._fa = FastaFile(self.fasta_path)
        if self._bws is None:
            self._bws = [pyBigWig.open(p) for p in self.bigwig_paths]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._open_files()
        assert self._fa is not None
        assert self._bws is not None
        #likely since we want to acess 2000 samples this doesnt work ...
        print(f"i am trying to access ", idx, " right now")
        print("we have so many chroms: ", len(self._positions))
        chrom, start = self._positions[idx]
        end = start + self.seq_length

        # ------------------ big chnage of input is here -------------------

        # Get DNA sequence and one-hot encode
        seq = self._fa.fetch(chrom, start, end)
        if self.species_lm_embedded:
            #if not self.species:  ----> hcnage this to once at teh top so we dont create that object in here we might as well check early on
             #   raise ValueError(
              #      f"No species found. Add species token for speciesLM"
               # )
            #else:
            #input_matrix = embed_sequence(seq, self.species, self._embedder, self._tokenizer) #(L, 738) --> for linear regression we needed mean pooling for this we dont i believe 
            chrom_emb = self._chrom_embeddings[chrom]   # already loaded, just look up per chrom
            input_matrix = chrom_emb[start:end]  # simply slice it --> should stilla lready be tensor object i believe ? test it 
        elif self.num_channels == 6:
            input_matrix = one_hot_encode(seq, num_channels=self.num_channels) # same process btu this will be 6
            input_matrix = torch.from_numpy(input_matrix)
            # make 170 long
            species_padding = torch.zeros(input_matrix.shape[0], 164, dtype=torch.float32) # 164 zeros 
            input_matrix = torch.cat([input_matrix, species_padding], dim=1) # put them on end to simulate ...

        else: 
            # rename onehot to input matrix makes it a bit easier to track
            input_matrix = one_hot_encode(seq, num_channels=self.num_channels) # (L, 4) 
            input_matrix = torch.from_numpy(input_matrix) # embedding is already torch tensor --> chenge this ehre

        print(f"this is the input matrix shape: ", input_matrix.shape, " right now")



        #--------------------------------------------------------------------
        # Get ATAC-seq signal from all bigwigs, average across replicates
        signals = []
        for bw in self._bws:
            vals = bw.values(chrom, start, end)
            arr = np.array(vals, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0)
            signals.append(arr)

        # Average across all bigwig files (replicates)
        signal = np.mean(signals, axis=0)  # (L,)

        if self.log_transform:
            signal = np.clip(signal, 0, None) # clip for tobias
            signal = np.log1p(signal)

        out ={
            "one_hot": input_matrix,       # (L, 4)
            "target": torch.from_numpy(signal),          # (L,)
            "chrom": chrom,
            "start": start,
        }
        # this is for again chrombpnet bais corrected bw because we have raw count edges
        if self.edge_trim > 0:
            L = self.chrom_lengths[chrom]
            # arrange them 
            coords = np.arange(start, end)           
            mask = ((coords >= self.edge_trim) &
                    (coords < L - self.edge_trim)).astype(np.float32)  # 1=keep, 0=drop
            out["loss_mask"] = torch.from_numpy(mask)

        return out

    def __del__(self) -> None:
        if self._fa is not None:
            self._fa.close()
        if self._bws is not None:
            for bw in self._bws:
                bw.close()

# call this method differently to avoid mistakes
def build_datasets_from_folds_modified(
    bigwig_dir: str | Path,
    fasta_path: str | Path,
    folds_path: str | Path,
    seq_length: int = 16384,
    train_samples: int = 10000,
    val_samples: int = 2000,
    seed: int = 42,
    log_transform: bool = True,
    sample_ids: list[str] | None = None,
    species = None, # pass that
    species_token = None,
    species_lm_embedded = False, # careful faslse by default 
    entire_chromosome = False, # careful faslse by default --> change only in evaluate
    embedding_root=None, #careful none by default
    peak_centered=False,
    peak_bed=None,
    non_peak_bed=None,
    num_channels: int = 4, #--> normal default
    edge_trim: int = 0,   # normally we use all
    skip_edges: int = 0, 
) -> dict[str, ATACSeqDatasetModified]:
    """Build train/valid/test datasets from a folds.json file.
    Args:
        bigwig_dir: Directory containing bigwig files.
        fasta_path: Path to the reference genome FASTA.
        folds_path: Path to folds.json with train/valid/test chromosome splits.
        seq_length: Sequence window length.
        train_samples: Samples per epoch for training.
        val_samples: Samples per epoch for validation.
        seed: Random seed.
        log_transform: Apply log1p to ATAC-seq signal.
        sample_ids: If provided, only use bigwig files whose stem (filename
            without extension) matches one of these IDs. This is used to
            filter to the samples specified in config.yaml.

    Returns:
        Dictionary with "train", "valid", and optionally "test" datasets.
    """
    bigwig_dir = Path(bigwig_dir)
    bigwig_paths = sorted(bigwig_dir.glob("*.bw"))
    if not bigwig_paths:
        raise FileNotFoundError(f"No .bw files found in {bigwig_dir}")

    # Filter to only the requested sample IDs
    if sample_ids is not None:
        allowed = set(sample_ids)
        filtered = [p for p in bigwig_paths if p.stem in allowed]
        skipped = [p.name for p in bigwig_paths if p.stem not in allowed]
        if skipped:
            logger.info("Filtered out %d bigwig files not in sample list: %s",
                        len(skipped), skipped)
        bigwig_paths = filtered
        if not bigwig_paths:
            raise FileNotFoundError(
                f"No .bw files matched sample_ids {sample_ids} in {bigwig_dir}"
            )

    with open(folds_path) as f:
        folds = json.load(f)

    # modify here with new dataset
    datasets: dict[str, ATACSeqDatasetModified] = {}
    for split, chroms in folds.items():
        n_samples = train_samples if split == "train" else val_samples
        # here we actually call the non rnadomized dataset 
        datasets[split] = ATACSeqDatasetModified(
            bigwig_paths=bigwig_paths,
            fasta_path=fasta_path,
            chromosomes=chroms,
            seq_length=seq_length,
            samples_per_epoch=n_samples,
            seed=seed,
            log_transform=log_transform,
            species_token=species_token, # new flags added here 
            species=species,
            species_lm_embedded = species_lm_embedded, 
            entire_chromosome = entire_chromosome,
            embedding_root=embedding_root,
            peak_centered=peak_centered,
            peak_path=peak_bed,
            non_peak_path=non_peak_bed,
            num_channels=num_channels,
            edge_trim=edge_trim, 
            skip_edges=skip_edges,
        )
        logger.info("Built %s dataset: %d chroms, %d samples", split, len(chroms), n_samples)

    return datasets
