#!/bin/bash
set -uo pipefail

# --- edit these if necessary ---


PYTHON="/opt/modules/i12g/anaconda/envs/rehnv_shorkie/bin/python"
CHROMBPNET_DATA="/s/project/multispecies/fungi_code/atac/data/benchmarking/chrombpnet_data"
SPECIES_INPUT_FILES="/s/project/multispecies/fungi_code/atac/data/benchmarking/species_input_files"
MODEL_OUTPUT="/s/project/multispecies/fungi_code/atac/data/benchmarking/models/cross_species"

BIGWIGS=(
    "replicate_bigwigs/raw_cutsite"
    "replicate_biwgigs/tobias"
)

MODELS=(gc gc_kmer embeddings shorkie shorkie_scratch)
SPECIES_LIST=(
    saccharomyces_cerevisiae
    schizosaccharomyces_pombe
    candida_albicans
    aspergillus_niger
    aspergillus_oryzae
    neurasposa_crassa   # misspelled on purpose 
)

mkdir -p logs
for BIGWIG in "${BIGWIGS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SPECIES in "${SPECIES_LIST[@]}"; do
            echo "submitting:  ${BIGWIG} | ${MODEL} | ${SPECIES}"
            "$PYTHON" train_benchmark_model.py \
                --species "${SPECIES}" \
                --model "${MODEL}" \
                --bigwig "${BIGWIG}" \
                --chrombpnet-data "${CHROMBPNET_DATA}" \
                --species-input-files "${SPECIES_INPUT_FILES}" \
                --model-output-folder "${MODEL_OUTPUT}"
        done
    done
done


