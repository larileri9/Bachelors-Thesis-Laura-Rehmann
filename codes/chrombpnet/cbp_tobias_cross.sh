#!/bin/bash
#SBATCH --job-name=cbp_tobias_%a
#SBATCH --array=0-5
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cbp_tobias_%A_%a.log

BENCH=/s/project/multispecies/fungi_code/atac/data/benchmarking

SPECIES=(saccharomyces_cerevisiae schizosaccharomyces_pombe candida_albicans \
         aspergillus_niger aspergillus_oryzae neurasposa_crassa)
TRAIN=${SPECIES[$SLURM_ARRAY_TASK_ID]}

export PYTHONPATH=$PYTHONPATH:.
export HDF5_USE_FILE_LOCKING=FALSE

# task 0 builds every averaged tobias track and tobias modle, since it walks all six test species
# the others cant run at the same time so we add a done flag 
DONE_FLAG=$BENCH/chrombpnet_data/.tobias_build_done

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    rm -f "$DONE_FLAG"
else
    echo "waiting for task 0 to finish building the averaged tracks"
    while [ ! -f "$DONE_FLAG" ]; do
        sleep 60
    done
    echo "task 0 done, startign to train others"
fi

# this can be run with replicate_bigwigs/raw_cutsite as well 

for TEST in "${SPECIES[@]}"; do
    echo "doing:   $TRAIN -> $TEST "
    python chrombpnet_benchmarking_workflow.py \
      --train-species "$TRAIN" \
      --test-species "$TEST" \
      --bigwig "average_bigwig/tobias" \
      --chrombpnet-data "$BENCH/chrombpnet_data" \
      --species-input-files "$BENCH/species_input_files" \
      --output-folder "$BENCH/evaluation_2/cross_species" \
      --seq-length 16384
done

# hold the others back
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    touch "$DONE_FLAG"
    echo "wrote $DONE_FLAG, tasks 1-5 running"
fi
