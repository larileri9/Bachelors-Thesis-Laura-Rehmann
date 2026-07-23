"""
evaluation launcher for again one train_species, test_species, model, bigwig combo

evaluation partof of train_benchmark_model.py idea

reuse train_benchmarking_utils to build the data paths
check everything exists first then build command submit

--train-species   
--test-species    

model -> who runs it:
    gc / gc_kmer / embeddings -> evaluate_benchmark_regression.py   
    shorkie / shorkie_embeddings  -> evaluate.py from shorkie (full path in script)                      

folds:
    baselines read the chrombpnet fold_0.json test split 
    shorkie reads folds from the --config yaml inside evaluate.py

seq length:
    default 16384 = what shorkie was trained at right now
    passed to both evaluate.py and the baseline so all models tile the same windows
    --> if we wnat we can try the downsized variant in another folder

output dir is built the same way as the model dir, with the test species added because
one trained model is evaluated against several test species:
(i read this si correct annotation isntead of {})
    model dir = <model_output>/<model>/<train_species>/<bigwig>
    eval output = <eval_output>/<model>/<train_species>/<test_species>/<bigwig>

--> we will have to redo the benchmarking steps as well so new path anming thing wont hurt
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
 
import loco_kfold
import train_benchmarking_utils as utils
 
# the repo dir we cd into inside the sbatch job (same as train_benchmark_model.py)
workdir = "/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/shorkie_atac"
 
# same hardcoded paths for constants
config_path_default = "/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/config/cross_configs/holdout_singular.yaml"
params_path_default = "/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline/shorkie_atac/configs/baskerville/shorkie_params.json"
 
# slightly better than the if else or statemnts 
baseline_models = ("gc", "gc_kmer", "embeddings")
shorkie_models = ("shorkie", "shorkie_scratch", "shorkie_embeddings")
 
# constant for shifting --> CAN BE OVERRIDDEN WHEN KFOLD OR SMTH
chrombpnet_edge_trim = (1216 - 608) // 2 
# actually other way round
 # args 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one model on one test species (full-chromosome).")
 
    parser.add_argument("--train-species", type=str, required=True,
                        help="Species the model was trained on")
    parser.add_argument("--test-species", type=str, required=True,
                        help="Species to evaluate on.")
    parser.add_argument("--model", type=str, required=True,
                        help="gc, gc_kmer, embeddings, shorkie, shorkie_embeddings")
    parser.add_argument("--bigwig", type=str, required=True,
                        help="Bigwig type, avergae or replicate/ raw or tobias")
 
    parser.add_argument("--chrombpnet-data", type=str, required=True,
                        help="chrombpnet data dir for folds in this case.")
    parser.add_argument("--species-input-files", type=str, required=True,
                        help="root of species_input_files (genomes / bigwigs / embeddings)")
    parser.add_argument("--model-output-folder", type=str, required=True,
                        help="Where the trained models are saved")
    parser.add_argument("--evaluation-output-folder", type=str, required=True,
                        help="Where evaluation results.npz / metrics.json go")
 
    parser.add_argument("--seq-length", type=int, required=False, default=16384,
                        help="Window length, default 16384 because tahts safe for now.")
    parser.add_argument("--config", type=str, default=config_path_default,
                        help="Shorkie config yaml")
    parser.add_argument("--params", type=str, default=params_path_default,
                        help="shorkie_params.json")
    parser.add_argument("--skip-edges", type=int, default=chrombpnet_edge_trim,help="Shift eval windows to [skip_edges, chrom_len - skip_edges).")
        #loco args / kfold args
    parser.add_argument("--kfold", action="store_true",
                    help="Evaluate the kfold setup.")
    parser.add_argument("--run-fold", type=int, default=None,
                    help="Evaluate only a single fold if smth went wrong")
    parser.add_argument("--overwrite", action="store_true",
                    help="Rerun folds that already have a results.npz")
 
    args = parser.parse_args()

    if args.kfold:
        args.test_species = args.train_species
    return args




# find the lowest val-loss checkpoint, translated back from bash
# the {val/loss} filename makes a "...-val/" subdir holding loss=X.XXXX.ckpt, so search recursively

# the chekpoint dir is already dependant on the model --> gets constructed with species and model
# --> wwr can directly plug this path taht comes out into the run shrokie command
# this is essentisally the same method as shorkies evaluate
def find_best_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    best_ckpt = None
    best_loss = None
    for ckpt in checkpoint_dir.rglob("*.ckpt"):
        match = re.search(r"loss=([-]?[0-9]+\.[0-9]+)", ckpt.name)
        if not match:
            continue
        loss = float(match.group(1))
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_ckpt = ckpt
    # fall back to last.ckpt if e find nothing should not be the case
    if best_ckpt is None:
        last = checkpoint_dir / "last.ckpt"
        if last.exists():
            best_ckpt = last
    return best_ckpt
 

# build and submit the shorkie evaluation on GPU
def run_shorkie_eval(args, bigwig_dir, output_dir, best_ckpt, config_path, params_path, fold=None):

    cmd = [
        sys.executable, "evaluate.py",
        "--config", str(config_path),
        "--checkpoint", str(best_ckpt),
        "--bigwig-dir", str(bigwig_dir),
        "--output-dir", str(output_dir),
        "--params", str(params_path),
        "--species", args.test_species,        # full name
        "--seq-length", str(args.seq_length),
        "--save-all-samples",
        "--run-plotting", "0",
        "--auto-benchmark", "0",
        "--skip-edges", str(args.skip_edges),
    ]
 
    # input type flag must match how the model was trained / what model we wnat to evaluate
    if args.model in ("shorkie", "shorkie_scratch"):
        cmd += ["--original-input"]             # dataset makes 170-dim input
    elif args.model == "shorkie_embeddings":
        embedding_root = utils.contruct_embedding_dir_path(args.test_species, args.species_input_files).parent
        cmd += ["--species-lm-embedded", "--embedding-root", str(embedding_root)]
 
    fold_tag = f"_fold{fold}" if fold is not None else ""
    job_name = f"eval_{args.model}_{args.train_species}_to_{args.test_species}{fold_tag}"
    sbatch_cmd = [
        "sbatch",
        "--job-name", job_name,
        "--partition", "standard",
        "--gres", "gpu:1",                      # shorkie needs a GPU
        "--cpus-per-task", "8",
        "--mem", "128G",
        "--time", "24:00:00",
        "--output", f"logs/{job_name}_%j.log",
        "--wrap",
        f"cd {workdir}; "
        "export HDF5_USE_FILE_LOCKING=FALSE; "
        "export TOKENIZERS_PARALLELISM=FALSE; "
        "export PYTHONPATH=$PYTHONPATH:.; "
        + " ".join(str(c) for c in cmd),
    ]
    print(" ".join(sbatch_cmd))
    subprocess.run(sbatch_cmd, check=True)
 
 
# build and submit a baseline evaluation this is on CPU
def run_baseline_eval(args, bigwig_dir, fasta_path, output_dir, test_chroms, fold=None):
    # only add segment for model if we have kfold
    fold_seg = f"fold_{fold}" if fold is not None else ""
    model_dir = Path(args.model_output_folder) / args.model / args.train_species / args.bigwig / fold_seg
    model_pkl = model_dir / "model.pkl"
    if not model_pkl.exists():
        raise FileNotFoundError(f"Missing baseline pickle: {model_pkl}")
 
    cmd = [
        sys.executable, "evaluate_benchmark_regression.py",
        "--feature", args.model,
        "--model", str(model_pkl),
        "--fasta", str(fasta_path),
        "--bigwig-dir", str(bigwig_dir),
        "--output-dir", str(output_dir),
        "--seq-length", str(args.seq_length),
        "--skip-edges", str(args.skip_edges),
    ]
    if args.model == "embeddings":
        embedding_root = utils.contruct_embedding_dir_path(args.test_species, args.species_input_files).parent
        cmd += ["--embedding-root", str(embedding_root), "--species", args.test_species]
 
    # chromosomes last so nargs="*" grabs them all cleanly
    cmd += ["--chromosomes", *test_chroms]
 
    fold_tag = f"_fold{fold}" if fold is not None else ""
    job_name = f"eval_{args.model}_{args.train_species}_to_{args.test_species}{fold_tag}"
    sbatch_cmd = [
        "sbatch",
        "--job-name", job_name,
        "--partition", "standard",              # CPU only
        "--cpus-per-task", "4",
        "--mem", "256G",
        "--time", "04:00:00",
        "--output", f"logs/{job_name}_%j.log",
        "--wrap",
        "export PYTHONPATH=$PYTHONPATH:.; " + " ".join(str(c) for c in cmd),
    ]
    print(" ".join(sbatch_cmd))
    subprocess.run(sbatch_cmd, check=True)
 
 
def main() -> None:
    args = parse_args()
    mkdir_logs = Path("logs")
    mkdir_logs.mkdir(exist_ok=True)
 
    # test species data paths (train doesnt matter)
    bigwig_dir = utils.construct_bigwig_dir(args.species_input_files, args.bigwig, args.test_species)
    if not bigwig_dir.is_dir():
        raise FileNotFoundError(f"Missing test bigwig dir: {bigwig_dir}")
 
    # where results go is the same nesting as the model dir + the test species at the end (do this dor based this time not name absed)
    output_dir = (
        Path(args.evaluation_output_folder)
        / args.model / args.train_species / args.test_species / args.bigwig
    )
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # embeddings must exist for the test species (embeddings + shorkie_embeddings) 
    if args.model in ("embeddings", "shorkie_embeddings"):
        embedding_dir = utils.contruct_embedding_dir_path(args.test_species, args.species_input_files)
        if not embedding_dir.is_dir():
            raise FileNotFoundError(f"Missing test embeddings: {embedding_dir}")
 
    
    if not args.kfold:
        fold_indices = [None]
    elif args.run_fold is not None:
        fold_indices = [args.run_fold]
    else:
        chrom_sizes = utils.construct_chrom_sizes_path(args.species_input_files, args.test_species)
        if not chrom_sizes.exists():
            raise FileNotFoundError(f"Missing chrom_sizes: {chrom_sizes}")
        fold_paths = loco_kfold.ensure_folds(
            chrombpnet_data=args.chrombpnet_data,
            species=args.test_species,
            chrom_sizes=chrom_sizes,
        )
        fold_indices = list(range(len(fold_paths)))

    # if we have a fold --> we either run one or all --> then we make sure to  add the fold to output path
    for fold in fold_indices:
        fold_seg = f"fold_{fold}" if fold is not None else ""

        # here add eitehr nothing or the fold 
        output_dir = (Path(args.evaluation_output_folder)  / args.model / args.train_species / args.test_species / args.bigwig / fold_seg)

        # check if done and skip if not overwrite
        results_npz = output_dir / "results.npz"
        metrics_json = output_dir / "metrics.json"
        if results_npz.exists() and metrics_json.exists() and not args.overwrite:
            print(f"[have] {args.model} {args.test_species} fold {fold}, skipping")
            continue


        output_dir.mkdir(parents=True, exist_ok=True)
        
        

        

        
        # baseline run
        if args.model in baseline_models:
            # baseline needs the test species fasta and the chrombpnet fold test chroms

            # check if fatsa, fold and test chroms exist
            fasta_path = utils.construct_fasta_path(args.species_input_files, args.test_species)
            if not fasta_path.exists():
                raise FileNotFoundError(f"Missing test fasta: {fasta_path}")
            # if fold is none we wvaluate on the normal path / get the normal fold0 otherwise the correct one aka without fold
            if fold is None:
                folds_path = utils.construct_folds_path(args.chrombpnet_data, args.test_species)
            else:
                folds_path = loco_kfold.construct_loco_fold_path(args.chrombpnet_data, args.test_species, fold)

            if not folds_path.exists():
                raise FileNotFoundError(f"Missing test folds: {folds_path}")

            with open(folds_path) as f:
                folds = json.load(f)
            if "test" not in folds:
                raise KeyError(f"No 'test' split in {folds_path}. Keys present: {list(folds)}")
            test_chroms = folds["test"]
            # then run abseline with them
            run_baseline_eval(args, bigwig_dir, fasta_path, output_dir, test_chroms, fold=fold)
    
        # shorkie run
        elif args.model in shorkie_models:
            # tst if constants exist  // same thing as in baseline with folds
            if fold is None:
                config_path = Path(args.config)
            else:
                config_path = loco_kfold.construct_loco_config_path(args.chrombpnet_data, args.test_species, fold)
            params_path = Path(args.params)
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config: {config_path}")
            if not params_path.exists():
                raise FileNotFoundError(f"Missing params: {params_path}")
    
            # get the checkpoint we need 
            checkpoint_dir = Path(args.model_output_folder) / args.model / args.train_species / args.bigwig / fold_seg / "checkpoints"
            if not checkpoint_dir.is_dir():
                raise FileNotFoundError(f"Missing checkpoints dir: {checkpoint_dir}")
            best_ckpt = find_best_checkpoint(checkpoint_dir)
            if best_ckpt is None:
                raise FileNotFoundError(f"No checkpoint found under: {checkpoint_dir}")

            #finally run
            run_shorkie_eval(args, bigwig_dir, output_dir, best_ckpt, config_path, params_path, fold=fold)
    
        else:
            raise ValueError(f"No model detected: {args.model}")
 
 
if __name__ == "__main__":
    main()
 
