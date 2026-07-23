#!/bin/bash
set -uo pipefail

PYTHON="/opt/modules/i12g/anaconda/envs/rehnv_shorkie/bin/python"
BENCH="/s/project/multispecies/fungi_code/atac/data/benchmarking"

CHROMBPNET_DATA="$BENCH/chrombpnet_data"
SPECIES_INPUT_FILES="$BENCH/species_input_files"
MODEL_OUTPUT="$BENCH/models/cross_species"
EVAL_OUTPUT="$BENCH/evaluation_2/cross_species"

BIGWIG="replicate_bigwigs/raw_cutsite"

MODELS=(embeddings shorkie shorkie_scratch gc gc_kmer)

TRAIN_LIST=(
    saccharomyces_cerevisiae
    schizosaccharomyces_pombe
    neurasposa_crassa
    aspergillus_niger
    aspergillus_oryzae
    candida_albicans
)

# tested against everything
TEST_LIST=(
    saccharomyces_cerevisiae
    schizosaccharomyces_pombe
    candida_albicans
    aspergillus_niger
    aspergillus_oryzae
    neurasposa_crassa
)

mkdir -p logs


for MODEL in "${MODELS[@]}"; do
  for TRAIN_SP in "${TRAIN_LIST[@]}"; do

    # dont submit a job for a model that isnt there
    MDIR="$MODEL_OUTPUT/$MODEL/$TRAIN_SP/$BIGWIG"
    if [ ! -d "$MDIR" ]; then
      echo "NO MODEL $MODEL $TRAIN_SP ($MDIR)" 
      continue
    fi

    for TEST_SP in "${TEST_LIST[@]}"; do
      echo "===running: $MODEL | $TRAIN_SP -> $TEST_SP ==="
      "$PYTHON" evaluate_benchmark_model.py \
          --model "$MODEL" \
          --train-species "$TRAIN_SP" \
          --test-species "$TEST_SP" \
          --bigwig "$BIGWIG" \
          --chrombpnet-data "$CHROMBPNET_DATA" \
          --species-input-files "$SPECIES_INPUT_FILES" \
          --model-output-folder "$MODEL_OUTPUT" \
          --evaluation-output-folder "$EVAL_OUTPUT"

    done
  done
done


