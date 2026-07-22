"""
training script for ONE model --> can be wrapped with anpther script 

CAREFUL I JUST REALISED ITS NEUROSPORA CRASSA NOT NEURASPOSA???????
args:
--species                   # entire species anme of species to run for
--model                     # model type youw ish to train
--bigwig                    # bigwig type you wish to run for
--chrombpnet-data           # where to extarct training windows from
--species-input-files       # where data cna be found
--model-output-folder       # where to output models

model: gc_content, gc_kmer, embeddings, shorkie, shorkie_scratch




--> runs the specific model for that species and that input
--> can be wrapped my model species beigwig or all
--> does then run stat specific model + checks all inputs

use rehnv_shorkie to run this script!

this script was based very much on the training pipeline of shorkieATAC in style

THINGS THAT COULD NEED CHNAGING:

- wandb paths
- config here is hardcoded --> if folds change so does config but we keep it the same here




"""

# ölater add merged bigwig maybe
# structure of the benchamrking folder (just for reference) --> also for me to rename all things standardized 

'''
/benchmarking/
    /species_input_files/
        /genomes/
            {species}_genome.fna / .fa 
        /bigwigs/
            /replicate_bigwigs/
                /{species}/
                    {replicate}.bw
            /merged_replicates/
                /{species}/
        /chrom_sizes/
            {species}_chrom_sizes.txt
        /gtfs/

        /blacklists/
            {species}_blacklist.bed
        /dataframes/
            /{species}/
                {region}_df.parquet
        /species_lm_embeddings/
            /{entire_species_name}/
                /{chrom}.pt
    /models/
        /cross_species/
            /{model_type}/          #move chrombpnet manually here
                /{species}/
                    /{bigwig}/ 
    /evaluation/
        /cross_species/                #evaluate chrombpnet into here
            /{model_type}/
                /{species}/
                    /{bigwig}/
    //
/chrombpnet_data/
    /{species}/
        /params_2/models/chrombpnet/
                            /{whatever the model name is}/      # but only one in here so grab that
                                /fold_0/auxiliary/
                                            filtered.nonpeaks.bed
                                            filtered.peaks.bed          # --> training windows
''' 
import argparse
import logging
import sys
from pathlib import Path
import json
import loco_kfold

import train_benchmarking_utils as utils
import subprocess


"""
# firsty check if all data we need exists (ggf. embeddings , chrombpnet training windows , biwgigs , files etc)
# secondly create the input window df or some data structure to pass
"""


# these -> 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train models for ATAC seq benchmarking."
    )

    # Data arguments
    parser.add_argument(
        "--bigwig", type=str, required=True,
        help="Type of bigwig for input.",
    )
    parser.add_argument(
        "--chrombpnet-data", type=str, required=True,
        help="Directory path to the trained crhombpnet models",
    )
    parser.add_argument(
        "--species-input-files", type=str, required=True,
        help="Directory path to the data and folds",
    )
    parser.add_argument(
        "--chrombpnet-bias-corrected", action="store_true", default=False,
        help="Train on the chrombpnet bias corrected bigwig, the bigwig source is the chrombpnet_bias_corrected folder and ignores the 304 bp at each chromosome end that has no bias prediction from chrombpnet",
   )


    # Output argumnets
    parser.add_argument(
        "--model-output-folder", type=str, required=True,
        help="Directory path to where models will be saved.",
    )
    
    # Distinction arguments
    parser.add_argument(
        "--model", type=str, required=True,
        help="Which type of model to train.",
    )

    parser.add_argument(
        "--species", type=str, required=True,
        help="Which species to train on.",
    )

    parser.add_argument(
        "--seq-length", type=int, required=False, default=16384,
        help="Window to trian shorkie on, still testing 1280 so default is 16384.",
    )

    parser.add_argument("--kfold", action="store_true",
                    help="this runs kfold loco and also makes the config/jsons")
    
    parser.add_argument("--run-fold", type=int, default=None,
                    help="Run only a single k-fold split if somthing goes wrong with chrombpnet for one."
                    )



    return parser.parse_args()

# here we build the commands for running which are kater used in main

chrombpnet_edge_trim = (1216 - 608) // 2 # needs to go to shrokei as well so put that here

# fixed constants here
config_path = Path("/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/config/cross_configs/holdout_singular.yaml")
params_path = Path("/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/shorkie_atac/configs/baskerville/shorkie_params.json")
pretrained_path = Path("/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/shorkie_atac/model_best.h5")
baseline_models = ("gc", "gc_kmer", "embeddings")
shorkie_models = ("shorkie", "shorkie_scratch", "shorkie_embeddings")




def run_shorkie(args, bigwig_dir, output_dir, cfg_path, params_path, pretrained_path, peaks, non_peaks, fold=None):
    seq_tag = "" if args.seq_length == 16384 else f"_{args.seq_length}" # this si for explicetly seperating the test wandb runs from my actual ones
    fold_tag = f"_fold{fold}" if fold is not None else ""
    cmd = [
        sys.executable, "train.py",          
        "--species", args.species,
        "--bigwig-dir", str(bigwig_dir),    #be sure all these paths are passed as strings no matetr how i construct them 
        "--config", str(cfg_path),
        "--params", str(params_path),
        "--output-dir", str(output_dir),
        "--batch-size", "4",
        "--max-epochs", "100",
        "--lr", "2e-5",
        "--warmup-steps", "5000",
        "--num-workers", "8",
        "--seq-length", str(args.seq_length),                    
        "--precision", "bf16-mixed",        
        "--peak-centered",                  # INPORTANT
        "--peak-bed", str(peaks),           
        "--nonpeak-bed", str(non_peaks),
        "--wandb",
        "--wandb-project", f"{args.model}_{args.bigwig}".replace("/", "_"),         # ake a new project per model and bigwig ...
        "--wandb-run-name", f"{args.model}_{args.bigwig}_{args.species}{seq_tag}{fold_tag}".replace("/", "_"), # replace because of / in bigwig setup
        "--wandb-entity", "laura-rehmann-technical-university-of-munich",
    ]

    # the edge trim so we mask the loss for shorkie where the bigwig values will still be raw
    if str(args.bigwig).endswith("chrombpnet_bias_corrected"):
        cmd += ["--edge-trim", str(chrombpnet_edge_trim)]

    # here the swith based on requested model type 
    if args.model == "shorkie":
        cmd += [
            "--pretrained", str(pretrained_path),       #actually load pretrained shorkieLM weights
            "--original-input",                         # 170 channel vector input
            "--freeze-backbone-epochs", "5",            #freeze 5 epochs because backbone exists
        ]
    elif args.model == "shorkie_scratch":
        cmd += [
            "--original-input",                         #here we want original input but no pretraining!!!
        ]

    elif args.model == "shorkie_embeddings":
        # since we construct the part with species but shorkie does that too --> wed have double --> this never gets called anymore DEAD CODE
        embedding_root = utils.contruct_embedding_dir_path(args.species, args.species_input_files).parent
        cmd += [
            "--species-lm-embedded",                    #here use the falg but we override the backbone with next fla
            "--embedding-replaced-head",                # predict from embeddings, no trunk
            "--embedding-root", str(embedding_root),    # embeddign dir we made fp for
        ]

    # small test
    #print(" ".join(cmd))
    # here again with sbatch and this time gpu also for snorkel need to cd to the dir where shorkie train is located 
    sbatch_cmd = [
        "sbatch",
        "--job-name", f"{args.model}_{args.species}",
        "--partition", "standard",
        "--gres", "gpu:1",
        "--cpus-per-task", "8",
        "--mem", "64G",
        "--time", "4:00:00",
        "--output", f"logs/{args.model}_{args.species}{fold_tag}_%j.log",  # %j = SLURM job id
        "--wrap",
        "cd /data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/shorkie_atac; "
        "export HDF5_USE_FILE_LOCKING=FALSE; "
        "export TOKENIZERS_PARALLELISM=FALSE; "
        "export PYTHONPATH=$PYTHONPATH:.; "
        + " ".join(str(c) for c in cmd),  # the whole python command as  string
    ]
    print(" ".join(sbatch_cmd))
    subprocess.run(sbatch_cmd, check=True)

def run_baseline(args, bigwig_dir, output_dir, peaks, non_peaks, folds_path=None, fold=None):
    # fasta + folds from the dirs
    fasta_path = utils.construct_fasta_path(args.species_input_files, args.species)
    if folds_path is None:
        folds_path = utils.construct_folds_path(args.chrombpnet_data, args.species)

    if not folds_path.exists():
        raise FileNotFoundError(f"Missing folds: {folds_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"Missing folds: {fasta_path}")


    
    fold_tag = f"_fold{fold}" if fold is not None else ""

    with open(folds_path) as f:
        folds = json.load(f)
    train_chroms = folds["train"]  # Here we ONLY take train chroms val we dont use

    output_pkl = output_dir / "model.pkl"

    # only paste model as iput flag 
    cmd = [
        sys.executable, "train_benchmark_regression.py",
        "--feature", args.model,
        "--fasta", str(fasta_path),
        "--bigwig-dir", str(bigwig_dir),
        "--peak-bed", str(peaks),
        "--nonpeak-bed", str(non_peaks),
        "--output", str(output_pkl),
        "--chromosomes", *train_chroms,
    ] 

    if args.model == "embeddings":
        embedding_root = utils.contruct_embedding_dir_path(args.species, args.species_input_files).parent
        cmd += [
            "--embedding-root", str(embedding_root),
            "--species", args.species,
        ]
    # edge trim so we throw those windows away even tho there shoudlnt be any or very little ones
    if str(args.bigwig).endswith("chrombpnet_bias_corrected"):
        cmd += ["--edge-trim", str(chrombpnet_edge_trim)]
    
    # run with sbatch 
    sbatch_cmd = [
        "sbatch",
        "--job-name", f"{args.model}_{args.species}",
        "--partition", "standard",                 #  here just cpu because linear regression is cheap
        "--cpus-per-task", "4",
        "--mem", "64G",
        "--time", "04:00:00",
        "--output", f"logs/{args.model}_{args.species}_{fold_tag}%j.log",
        "--wrap", " ".join(str(c) for c in cmd),
    ]

    subprocess.run(sbatch_cmd, check=True)





def main() -> None:
    args = parse_args()
    Path("logs").mkdir(exist_ok=True)

    # ---- checks needed in both modes ----
    if args.model == "embeddings" or args.model == "shorkie_embeddings":
        embedding_path = utils.contruct_embedding_dir_path(species=args.species, input_dir=args.species_input_files)
        if not embedding_path.is_dir():
            raise ValueError(f"Missing embedding directory: {embedding_path}")

    bigwig_dir = utils.construct_bigwig_dir(args.species_input_files, args.bigwig, args.species)
    if not bigwig_dir.is_dir():
        raise FileNotFoundError(f"Missing bigwig directory: {bigwig_dir}")

    if args.model in shorkie_models:
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}")
        if not params_path.exists():
            raise FileNotFoundError(f"Missing params: {params_path}")
    if args.model == "shorkie":
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Missing pretrained weights: {pretrained_path}")

    # =========================== ///// single fold for the howldout ///// ===========================
    if not args.kfold:
        chrombpnet_dir = Path(args.chrombpnet_data)
        if not chrombpnet_dir.is_dir():
            raise FileNotFoundError(f"No chrombpnet data found: {chrombpnet_dir}")

        peaks, non_peaks = utils.contstruct_peak_nonpeak_filepath(
            chrombpnet_data=args.chrombpnet_data, species=args.species)
        if not peaks.exists():
            raise FileNotFoundError(f"Missing peaks file: {peaks}")
        if not non_peaks.exists():
            raise FileNotFoundError(f"Missing non-peaks file: {non_peaks}")

        output_dir = Path(args.model_output_folder) / args.model / args.species / args.bigwig
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.model in baseline_models:
            run_baseline(args, bigwig_dir, output_dir, peaks, non_peaks)
        elif args.model in shorkie_models:
            run_shorkie(args, bigwig_dir, output_dir, config_path, params_path, pretrained_path, peaks, non_peaks)
        else:
            raise ValueError(f"No model detected: {args.model}")
        return

    # =============================== /// k-fold /// ===============================
    chrom_sizes = utils.construct_chrom_sizes_path(args.species_input_files, args.species)
    if not chrom_sizes.exists():
        raise FileNotFoundError(f"Missing chrom_sizes: {chrom_sizes}")

    # check / make folds
    fold_paths = loco_kfold.ensure_folds(
        chrombpnet_data=args.chrombpnet_data,
        species=args.species,
        chrom_sizes=chrom_sizes,
    )

    k = len(fold_paths)
    #check if weve got config
    if args.model in shorkie_models:
        for i in range(k):
            loco_kfold.ensure_config(args.chrombpnet_data, config_path, args.species, i)

    # in case we only want to run one fold to correct it
    fold_indices = [args.run_fold] if args.run_fold is not None else range(k)
    submitted = 0

    for i in fold_indices:
        peaks, non_peaks = utils.contstruct_peak_nonpeak_filepath(
            chrombpnet_data=args.chrombpnet_data, species=args.species, fold=i)

        if not (peaks.exists() and non_peaks.exists()):
            print(f"[skip] {args.species} fold {i}: no peakfile yet")
            continue

        output_dir = Path(args.model_output_folder) / args.model / args.species / args.bigwig / f"fold_{i}"
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.model in baseline_models:
            folds_path = loco_kfold.construct_loco_fold_path(args.chrombpnet_data, args.species, i)
            run_baseline(args, bigwig_dir, output_dir, peaks, non_peaks, folds_path=folds_path, fold=i)
        else:
            cfg = loco_kfold.construct_loco_config_path(args.chrombpnet_data, args.species, i)
            run_shorkie(args, bigwig_dir, output_dir, cfg, params_path, pretrained_path, peaks, non_peaks, fold=i)
        submitted += 1

        
if __name__ == "__main__":
    main()












