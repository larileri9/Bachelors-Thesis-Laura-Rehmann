#!/bin/bash
#SBATCH --job-name=shorkie_multisp_eval
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-23 # depending on how many to run
#SBATCH --output=/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/shorkie_atac/logs/multispecies_eval/%x_%A_%a.out
#SBATCH --error=/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/shorkie_atac/logs/multispecies_eval/%x_%A_%a.err

cd $SLURM_SUBMIT_DIR
export HDF5_USE_FILE_LOCKING=FALSE
export TOKENIZERS_PARALLELISM=FALSE
export PYTHONPATH=$PYTHONPATH:.
PYTHON_BIN="${PYTHON_BIN:-/opt/modules/i12g/anaconda/envs/rehnv_shorkie/bin/python}"

# ---- check these paths once ?

CONFIG="/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/config/cross_configs/holdout_singular.yaml"
PARAMS="configs/baskerville/shorkie_params.json"
BW_BASE="/s/project/multispecies/fungi_code/atac/data/benchmarking/species_input_files/bigwigs/combined"
MODELS_BASE="/s/project/multispecies/fungi_code/atac/data/benchmarking/models/multispecies"
EVAL_BASE="/s/project/multispecies/fungi_code/atac/data/benchmarking/evaluation_2/multispecies"
mkdir -p /data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/shorkie_atac/logs/multispecies_eval

# ---- same sweep as training and also same order ----
holdouts=(pombe niger albicans crassa cerevisiae oryzae)
bw_types=(tobias raw_cutsite)
modes=(pretrained scratch)

# ---- build one flat list of every combination ----
combos=()
for holdout in "${holdouts[@]}"; do
  for bw_type in "${bw_types[@]}"; do
    for mode in "${modes[@]}"; do
      combos+=("${holdout} ${bw_type} ${mode}")
    done
  done
done

# ---- pick the one combination for this evaluation array task with id again ----
combo=${combos[$SLURM_ARRAY_TASK_ID]}
read holdout bw_type mode <<< "$combo"
echo ">>> task=${SLURM_ARRAY_TASK_ID}  holdout=no_${holdout}  bw=${bw_type}  mode=${mode}"

# folder name uses the short name, --species needs the full config key  --> need to get the correct species so we know which one to evaluate
holdout_to_species() {
    case "$1" in
        pombe)      echo "schizosaccharomyces_pombe" ;;
        niger)      echo "aspergillus_niger" ;;
        albicans)   echo "candida_albicans" ;;
        crassa)     echo "neurasposa_crassa" ;;
        cerevisiae) echo "saccharomyces_cerevisiae" ;;
        oryzae)     echo "aspergillus_oryzae" ;;
        *)          echo "" ;;
    esac
}

#lowest val-loss checkpoint function translated form python to bash 
get_best_checkpoint() {
    local fold_dir="$1"
    local best_ckpt=""
    local best_loss=999999
    for epoch_dir in "${fold_dir}"/checkpoints/shorkie_atac-epoch=*-val; do
        [ -d "$epoch_dir" ] || continue
        ckpt_file=$(ls "${epoch_dir}"/loss=*.ckpt 2>/dev/null | head -1)
        [ -z "$ckpt_file" ] && continue
        loss=$(basename "$ckpt_file" | sed 's/loss=//; s/\.ckpt//')
        if awk "BEGIN {exit !($loss < $best_loss)}"; then
            best_loss=$loss
            best_ckpt=$ckpt_file
        fi
    done
    echo "$best_ckpt"
}

# ---- mode decides model_dir (same mapping as training except ofc training) 
if [ "$mode" = "pretrained" ]; then
    model_dir="shorkie"
else
    model_dir="shorkie_scratch"
fi

# --- get whcih species to run test for
test_species=$(holdout_to_species "$holdout")
if [ -z "$test_species" ]; then
    echo "Unknown holdout '${holdout}' check commands"
    exit 1
fi

# get the trained model the specie same is the one to test and then th checkpoint
output_train="${MODELS_BASE}/${model_dir}/${bw_type}/no_${holdout}/output"
best_ckpt=$(get_best_checkpoint "${output_train}")
if [ -z "$best_ckpt" ]; then
    echo "No checkpoint under ${output_train}/checkpoints, skipping this task."
    exit 1
fi

# last check + make output dir + run 
bw_dir="${BW_BASE}/${bw_type}"
if [ ! -d "$bw_dir" ]; then
    echo "Missing bigwig dir ${bw_dir}, skipping this task."
    exit 1
fi

output_eval="${EVAL_BASE}/${model_dir}/${bw_type}/no_${holdout}"
mkdir -p "${output_eval}"


# --original-inputrequired: training used it, so the model expects that input shape we only use that one now 
"$PYTHON_BIN" evaluate.py \
    --config "${CONFIG}" \
    --checkpoint "${best_ckpt}" \
    --bigwig-dir "${bw_dir}" \
    --params "${PARAMS}" \
    --species "${test_species}" \
    --seq-length 16384 \
    --original-input \
    --output-dir "${output_eval}" \
    --run-plotting 0 \
    --save-all-samples \
    --auto-benchmark 0

