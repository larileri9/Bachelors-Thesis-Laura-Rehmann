#!/bin/bash
#SBATCH --job-name=cbp_kfold_%a
#SBATCH --array=0-5
#SBATCH --partition=noninterruptive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/cbp_kfold_%A_%a.log

BENCH=/s/project/multispecies/fungi_code/atac/data/benchmarking

SPECIES=(saccharomyces_cerevisiae schizosaccharomyces_pombe candida_albicans aspergillus_niger aspergillus_oryzae neurasposa_crassa)

SP=${SPECIES[$SLURM_ARRAY_TASK_ID]}

export PYTHONPATH=$PYTHONPATH:.
export HDF5_USE_FILE_LOCKING=FALSE

# no done flag needed because --kfold forces test == train, so each array task only builds the averaged tobias track and model for its own species
for BW in "replicate_bigwigs/raw_cutsites" "average_bigwig/tobias"; do
    echo "doing: $SP | $BW "
    python chrombpnet_benchmarking_workflow.py \
      --kfold \
      --train-species "$SP" \
      --test-species "$SP" \
      --bigwig "$BW" \
      --chrombpnet-data "$BENCH/chrombpnet_data_kfold" \
      --species-input-files "$BENCH/species_input_files" \
      --output-folder "$BENCH/evaluation_2/kfold" \
      --seq-length 16384
done
