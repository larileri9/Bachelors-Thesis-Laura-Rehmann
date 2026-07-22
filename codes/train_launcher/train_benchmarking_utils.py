from pathlib import Path
import os 

def construct_species_short(species):
    parts = species.split("_")
    start = parts[0]
    end = parts[1]
    shortened_start = start[:1]
    name = f"{shortened_start}_{end}"
    return name




# just adjusted this to work with both cases kfold and non kfold

def contstruct_peak_nonpeak_filepath(chrombpnet_data, species, fold=0):
 
    species_name = construct_species_short(species)
    base = Path(chrombpnet_data)

    model_name_base = (base / species_name / "params_2" / "models" / "chrombpnet")
	
    # theres only one model IN MY CASE so iterdir and next is fine 
    model_dir = next(p for p in model_name_base.iterdir() if p.is_dir())

    peaks = (model_dir / f"fold_{fold}" / "auxiliary" / "filtered.peaks.bed")

    non_peaks = (model_dir / f"fold_{fold}" / "auxiliary" / "filtered.nonpeaks.bed")

    return peaks, non_peaks


    
def contruct_embedding_dir_path(species, input_dir):
    return Path(input_dir) / "species_lm_embeddings" / species

def construct_bigwig_dir(species_input_files, bigwig, species):
    species_name = construct_species_short(species)
    return Path(species_input_files) / "bigwigs" / bigwig / species_name 

def construct_fasta_path(species_input_files, species):
    species_name = construct_species_short(species)
    genomes = Path(species_input_files) / "genomes"
    # extension varies because we have .fa for s_cerevisiae and .fna for the rest
    for ext in (".fna", ".fa"):
        candidate = genomes / f"{species_name}_genome{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No genome fna or fa for {species_name} in {genomes}")


def construct_folds_path(chrombpnet_data, species):
    species_name = construct_species_short(species)
    return Path(chrombpnet_data) / "folds" / species_name / "fold_0.json"


# ------ all extracted from the chrombpnet pipeline for better visibility and so we can reuse

# get th chrom sizes of the test sepcies
def construct_chrom_sizes_path(species_input_files, test_species):
    species_name = construct_species_short(test_species)
    return Path(species_input_files) / "chrom_sizes" / f"{species_name}_chrom_sizes.txt"

# this gets the single model dir for tobais essentially  for the normal one weve got a differnt method
def construct_chrombpnet_model_dir(chrombpnet_data, species):
    species_name = construct_species_short(species)
    model_root = Path(chrombpnet_data) / species_name / "params_2" / "models" / "chrombpnet"
    return next(p for p in model_root.iterdir() if p.is_dir())

# merged deduped shifted bam for tobais generation
def construct_species_bam(chrombpnet_data, species):
    species_name = construct_species_short(species)
    bam = Path(chrombpnet_data) / species_name / "results" / "05_merged" / "all_merged.sorted.shifted.bam"
    if not bam.exists():
        raise FileNotFoundError(f"Missing merged bam: {bam}")
    return bam


# peaks also for tobais sometimes they dont have a blacklist tho --> candidates are the two possible files
def construct_species_peaks(chrombpnet_data, species):
    species_name = construct_species_short(species)
    peaks_dir = Path(chrombpnet_data) / species_name / "results" / "06_peaks" 
    candidates = [
        "all_merged_relaxed_peaks_no_blacklist.bed",
        "all_merged_relaxed_peaks.narrowPeak",
    ]
    for name in candidates:
        peaks = peaks_dir / name
        if peaks.exists():
            return peaks
    raise FileNotFoundError(
        f"No peaks file found in {peaks_dir} (tried: {candidates})"
    )