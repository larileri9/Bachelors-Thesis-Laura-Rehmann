# take one species and make one pt per chromosme --> used for shorkie then


"""Generate Species LM per-chromosome embeddings for one species.

run this in pjo_flash2 environment:
    python generate_species_embeddings.py \ # or whatever yfile is called
        --fasta /path/to/genome.fa \
        --species _saccharomyces_cerevisiae \ --> for example 
        --output-dir /path/to/embeddings/

Output:
    one .pt file per chromosome: output-dir/saccharomyces_cerevisiae/chrI.pt, chrII.pt, ... (same manmes as in genome (pass species_input_files for that)
    Each file is a tensor of shape (chrom_len, 768).
"""

import sys
sys.path.insert(0, "/data/nasif12/home_if12/rehm/baseline_codes") 

import argparse
import os
import torch
from pathlib import Path
from pysam import FastaFile
from baseline_codes import Species_lm_embeddings as embedding_generator



# the standard things

chunk_size = 1000   # maximum input size of SpeciesLM is 1024 so we can't exceed this
stride = 500         # how many positions we move forward each window
                     

context_window = (chunk_size - stride) // 2 # technically adjsutabel but its hardcoded so = (1000 - 500) // 2 = 250

# ---> 250 bp are cut off at each end / are not saved only center is saved for more stable context but to save some runtime 

parser = argparse.ArgumentParser()
parser.add_argument("--fasta", required=True,  help="Path to genome fasta file.")
parser.add_argument("--species", required=True,  help="Species token and for s.cer use: _saccharomyces_cerevisiae")
parser.add_argument("--output-dir", required=True,  help="Directory to save per chromosome .pt files.")
args = parser.parse_args()

# safety first
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# load the embedder and the tokenizer
embedder = embedding_generator.SpeciesLMV1_embedder(
    model_path="/s/project/denovo-prosit/JohannesHingerl/BERTADN/final_models/species_upstream_1000_k1/",
    kmer_size=1,
    device="cuda"
)
tokenizer = embedder.tokenize_func()
print("Embedder + mtokenizer loaded.")


fa = FastaFile(args.fasta)
# loop over chroms
for chrom in fa.references:
    out_path = output_dir / f"{chrom}.pt"
    # --> because we are regenrating in case i missed smth 
    if out_path.exists():
        print(f"  {chrom}: already exists, skipping.")
        continue

    chrom_len = fa.get_reference_length(chrom)
    # --> in case theres a problem
    print(f"\nProcessing {chrom})")

    # fetch the chromosome sequence as a string
    seq = fa.fetch(chrom, 0, chrom_len)

    # because we now use a context window we NEED the contxt window like border around it. 
    # because we always want 250 bp left and right at leas we need to add 250 NS to the start 
    # tradeoff --< start and end actually  get a bit les context now but everything else gets better context
    padded_seq = ("N" * context_window) + seq + ("N" * context_window)
    padded_len = len(padded_seq)
    # padded_len = chrom_len + 2 * context border

    all_embs = []  # will collect one tensor per window, then concatenate --> so like 500 x 768 and then concat

    # slide the window along the padded sequence
    # each step moves the stride forward
    # we stop when we have covered all real chromosome positions

    positions_kept = 0

    for i in range(0, padded_len - context_window, stride):
        # get the current window
        window_start = i
        window_end = i + chunk_size
        window = padded_seq[window_start:window_end]

        # if we are near the end and the window is shorter than chunk size,
        #pad with Ns on the right so speciesLM always receives exactly 1000 bases
        if len(window) < chunk_size:
            window = window + "N" * (chunk_size - len(window))

        # embed that window thta we have
        tokenized = tokenizer(window, args.species)
        input_id = tokenized["input_ids"].unsqueeze(0)  # add batch dim --> (1, 1000) need to simulate taht because we arent uskng batch

        with torch.no_grad():
            emb = embedder.embed(input_id.to(embedder.device))
        emb = emb[0].squeeze(0).cpu()  # squeeze removes abtch (1000, 768)

        # keep only the center 500 embedddings
        center_emb = emb[context_window : context_window +stride] 

        # figure out how many of those center positions correspond to real chrom positions
        # this only fires in the last window where the chrom was padded --> just check if any of what we append is not a real chromosome anymore
        # the center of this window corresponds to padded positions i+250 to i+749

        real_pos_start = i # first real position this window contributes
        real_pos_end = min(i + stride, chrom_len)  # tehn we see since we would append to 750 if that is still in chrom --> careful since thsi is actuall chrom not padded
        n_real = real_pos_end - real_pos_start #if not if its shorter we can see how much we can append

        if n_real <= 0:
            # we have gone past the end of the chromosome, stop
            break

        # here we append then the rest of the center
        all_embs.append(center_emb[:n_real])
        positions_kept += n_real

        # we can stop iof were over chrom 
        if positions_kept >= chrom_len:
            break

    # concatenate all kept embeddings into one tensor
    full_emb = torch.cat(all_embs, dim=0)  # (sum of n_real, 768)

    # test if smth goes wrong
    print(f"Final shape: {full_emb.shape}  (expected ({chrom_len}, 768))")

   # make tmp firts then once doen rename so it isnt incomplete and looks rgth if smth happens ...
    tmp_path = str(out_path) + ".tmp"
    torch.save(full_emb, tmp_path)
    os.rename(tmp_path, str(out_path))
    print(f"  Saved to {out_path}")

fa.close()
