#!/bin/bash
#SBATCH --job-name=shorkie_multispecies
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-23 		# max 24 jobs to be run need to adjust based on combo list!
#SBATCH --output=/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/shorkie_atac/logs/multispecies/%x_%A_%a.out
#SBATCH --error=/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/shorkie_atac/logs/multispecies/%x_%A_%a.err



# this is one adaptable script that runs the wanted bigwig type or snorkel scratch / shorkie  or organisms. 
# script is flexible but changes need to be made hardcoded in here for model to sweep and launch everything in an array

cd $SLURM_SUBMIT_DIR
export HDF5_USE_FILE_LOCKING=FALSE
export TOKENIZERS_PARALLELISM=FALSE
export PYTHONPATH=$PYTHONPATH:.

# ---- check these paths once if they exist ? 
# careful again combined only means all replicates are in one folder for multispecies but same replicates as in single mode 
CONFIG_DIR="/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/config/cross_configs"
BW_BASE="/s/project/multispecies/fungi_code/atac/data/benchmarking/species_input_files/bigwigs/combined"
OUT_BASE="/s/project/multispecies/fungi_code/atac/data/benchmarking/models/multispecies"
PRETRAINED="model_best.h5"

# make the log dir
mkdir -p /data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/shorkie_atac/logs/multispecies



# ---- all the settings to sweep over ----
holdouts=(pombe niger albicans crassa cerevisiae oryzae)
bw_types=(raw_cutsite tobias)
modes=(pretrained scratch)

# ---- build a combo list for every combination that needs to be run ----
combos=()
for holdout in "${holdouts[@]}"; do
  for bw_type in "${bw_types[@]}"; do
    for mode in "${modes[@]}"; do
      combos+=("${holdout} ${bw_type} ${mode}")
    done
  done
done

# ---- pick the one combo for this array task based on the index of the array id in the combo list----
combo=${combos[$SLURM_ARRAY_TASK_ID]}
read holdout bw_type mode <<< "$combo"

echo ">>> task=${SLURM_ARRAY_TASK_ID}  holdout=no_${holdout}  bw=${bw_type}  mode=${mode}"

# ---- mode decides output folder and pretraining flags ----
if [ "$mode" = "pretrained" ]; then
    model_dir="shorkie"
    pretrain_args="--pretrained ${PRETRAINED} --freeze-backbone-epochs 5"
else
    model_dir="shorkie_scratch"
    pretrain_args="--freeze-backbone-epochs 0"
fi

# make output fp and run

output_fp="${OUT_BASE}/${model_dir}/${bw_type}/no_${holdout}/output"
mkdir -p "${output_fp}"

python train.py \
    --bigwig-dir "${BW_BASE}/${bw_type}" \
    --config "${CONFIG_DIR}/holdout_no_${holdout}.yaml" \
    --params configs/baskerville/shorkie_params.json \
    --seq-length 16384 \
    --batch-size 4 \
    --max-epochs 100 \
    --lr 2e-5 \
    --warmup-steps 5000 \
    --num-workers 8 \
    --precision bf16-mixed \
    --original-input \
    --peak-centered \
    --output-dir "${output_fp}" \
    ${pretrain_args} \
    --wandb \
    --wandb-project "shorkie_multispecies_holdout" \
    --wandb-run-name "${model_dir}_${bw_type}_no_${holdout}" \
    --wandb-entity "laura-rehmann-technical-university-of-munich"
