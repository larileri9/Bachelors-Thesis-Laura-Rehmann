"""
workflow to get to get a results npz like shorkie
take the same tiling / make different tiling predictions (ive decided on an extra dir for other seq-length i will trian over night)

--> we write a bed region file which completely tiles the chromosome (region that is supposed tio be predcited has to match output len)
--> we then construct which model to use
--> we then run on this region file
--> then we read the predicted bw back over the eval tiles and save per position values in the npz

pred_bw needs a gpu so submit this whole script as ONE sbatch job (it runs pred_bw blocking then saves)
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pyBigWig
import chrombpnet
# shared helpers i already have
import train_benchmarking_utils as utils
import loco_kfold  

# reuse the same machinery the baselines use so every model lands on the same per position grid
from evaluate_benchmark_dataset import tile_chromosomes
from train_benchmark_dataset import observed_signal

import sys
sys.path.append("/data/nasif12/home_if12/rehm/shorkie/atac-seq-pipeline")

from shorkie_atac.evaluation_utils import (
    calculate_metrics,
    compute_per_window_pearson,
    save_evaluation_bundle,
    save_metrics,
)


chrombpnet_input_len = 1216
chrombpnet_output_len = 608


# args
def parse_args():
    p = argparse.ArgumentParser(
        description="Predict with ChromBPNet over test chromosomes and save a shorkie-style results.npz"
    )
    p.add_argument("--train-species", required=True,
               help="species the chrombpnet model was trained on e.g. saccharomyces_cerevisiae")
    p.add_argument("--test-species", required=True,
               help="species whose genome we predict on and score against e.g. candida_albicans")
    p.add_argument("--bigwig", required=True,
                   help="bigwig column e.g. replicate_bigwigs/raw_cutsites or replicate/tobias")
    p.add_argument("--chrombpnet-data", required=True, help="benchmarking/chrombpnet_data")
    p.add_argument("--species-input-files", required=True, help="benchmarking/species_input_files")
    p.add_argument("--output-folder", required=True,
                   help="evaluation root, results go to <root>/chrombpnet/<species>/<bigwig>/")
    p.add_argument("--seq-length", type=int, default=16384,
                   help="eval tiling window. MUST match what shorkie/baselines used")
    # ---- new kfold args ----
    p.add_argument("--kfold", action="store_true",
                   help="Run every loco fold into its own dir")
    p.add_argument("--run-fold", type=int, default=None,
                   help="Run only a single fold (if one broke or smth).")


    # ---> simple safty net
    args = p.parse_args()        
    if args.kfold:
        args.test_species = args.train_species

    return args

# output path segment
def output_fold_seg(fold):
    return "" if fold is None else f"fold_{fold}"

# normal run reads fold_0 but kfold run reads fold_{i}
def data_fold_dir(fold):
    return "fold_0" if fold is None else f"fold_{fold}"

# the json holding the train/test chrom split
def fold_json_path(args, fold):
    if fold is None:
        return utils.construct_folds_path(args.chrombpnet_data, args.test_species)
    return loco_kfold.construct_loco_fold_path(args.chrombpnet_data, args.test_species, fold)

# just laod them fro prediciton sicne they arent fixed 
def get_test_chroms(args, fold):
    with open(fold_json_path(args, fold)) as f:
        return json.load(f)["test"]

# if we ran the per replicate atacorrect here are the genome beds to use
def construct_genome_bed_path(species_input_files, species):
    return Path(species_input_files) / "genome_beds" / f"{species}_genome.bed"

# which folds to run? same logic as eval launcher find the fold json for that fold in the chrombpnte_data_kfold
def compute_fold_indices(args, chrom_sizes_file):
    if not args.kfold:
        return [None]
    if args.run_fold is not None:
        return [args.run_fold]
    fold_paths = loco_kfold.ensure_folds(
        chrombpnet_data=args.chrombpnet_data,
        species=args.test_species,
        chrom_sizes=Path(chrom_sizes_file),
    )
    return list(range(len(fold_paths)))




# model path cosntruction fromt he train specjes
def construct_chrombpnet_model_path(args, fold=None):
    """
    Pick the right ChromBPNet .h5 for the bigwig column we are filling

      raw_cutsites = bigwig still has Tn5 bias -> chrombpnet.h5       (full signal)
      tobias = new worflow  with tobias in here 
    """
    bigwig = str(args.bigwig)
    species_name = utils.construct_species_short(args.train_species)
    base = Path(args.chrombpnet_data)

    # same as contstruct_peak_nonpeak_filepath cuz there is exactly one model in chrombpnet
    model_root = base / species_name / "params_2" / "models" / "chrombpnet"
    model_dir = next(p for p in model_root.iterdir() if p.is_dir())
    fold_models = model_dir / data_fold_dir(fold) / "models" #--> add fold insted of defult fold_0
    # model switch
    bigwig = str(bigwig)
    if bigwig.endswith("raw_cutsites"):
        model_file = "chrombpnet.h5"          
    elif bigwig.endswith("tobias"):
        return construct_tobias_model_path(args, fold)
    elif bigwig.endswith("chrombpnet_bias_corrected"):
        model_file = "chrombpnet_nobias.h5"  
    else:
        raise ValueError(f"Unknown bigwig type: {bigwig}")

    # find exact model
    model_path = fold_models / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"missing chrombpnet model: {model_path}")
    return model_path

# try to find a file in a folder 
def find_one(folder, pattern, label):
    import glob
    hits = sorted(glob.glob(str(Path(folder) / pattern)))
    if not hits:
        raise FileNotFoundError(f"didnt find {label} + (pattern '{pattern}') in {folder}")
    return Path(hits[0])



# beds live under species_input_files/pred_bw_beds/{species_short} of the TEST species so i can inspect them
# --> made the fold=None change to be able to run kfold
def construct_pred_bed_path(species_input_files, test_species, fold=None):
    species_name = utils.construct_species_short(test_species)
    foldname = "pred_regions.bed" if fold is None else f"pred_regions_fold{fold}.bed"
    return Path(species_input_files) / "pred_bw_beds" / species_name / foldname


# the ground truth chrombpnet is the merged data_unstranded.bw in fold_0/
# careful gotta use test specis here
def construct_chrombpnet_target_bw(chrombpnet_data, test_species, fold=None):
    species_name = utils.construct_species_short(test_species)
    base = Path(chrombpnet_data)
    # same model dir as construct_chrombpnet_model_path (exactly one inside)
    model_root = base / species_name / "params_2" / "models" / "chrombpnet"
    model_dir = next(p for p in model_root.iterdir() if p.is_dir())
    # fold added
    target_bw = model_dir / data_fold_dir(fold) / "auxiliary" / "data_unstranded.bw"
    if not target_bw.exists():
        raise FileNotFoundError(f"Missing chrombpnet target bigwig: {target_bw}")
    return target_bw


#make a path to the scaled bias model here to use it in predicitng a bias track
def construct_scaled_bias_model(chrombpnet_data, test_species, fold=None):
    species_name = utils.construct_species_short(test_species)
    base = Path(chrombpnet_data)
    model_root = base / species_name / "params_2" / "models" / "chrombpnet"
    model_dir = next(p for p in model_root.iterdir() if p.is_dir())
    path = model_dir / data_fold_dir(fold) / "models" / "bias_model_scaled.h5"
    if not path.exists():
        raise FileNotFoundError(f"Missing scaled bias model: {path}")
    return path


# this is useless and dead code. Would technically subtract the bias track from the  raw track
def build_corrected_target_bw(args, fasta_path, chrom_sizes_file, bed_path, output_dir, fold=None):
    """
    predict atn5 bias track fro only our test chroms.
    then substrackt from the data unstranded bigwig 
    """
    output_dir = Path(output_dir)
    corrected_bw = output_dir / "corrected_target.bw"
    if corrected_bw.exists():
        print(f"corrected target already there, skipping: {corrected_bw}")
        return corrected_bw

    # go over the smae files like in the other py script,make a temp bw in the output dir 
    bias_dir = output_dir / "bias_predbw_tmp"
    bias_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = bias_dir / "pred"
    bias_bw_path = Path(str(out_prefix) + "_bias.bw")

    #here check so we dont compute more often 
    if not bias_bw_path.exists():
        scaled_bias_model = construct_scaled_bias_model(args.chrombpnet_data, args.test_species, fold)
        cmd = [
            "chrombpnet", "pred_bw",
            "-bm", str(scaled_bias_model),
            "-r", str(bed_path),
            "-g", str(fasta_path),
            "-c", str(chrom_sizes_file),
            "-op", str(out_prefix),
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)  # BLOCKING needs the GPU
        if not bias_bw_path.exists():
            raise FileNotFoundError(f"bias pred_bw did not write: {bias_bw_path}")

# gte data unstranded
    observed_bw_path = construct_chrombpnet_target_bw(args.chrombpnet_data, args.test_species, fold)

    #test chroms + lengths / sizes
    folds_path = utils.construct_folds_path(args.chrombpnet_data, args.test_species)
    test_chroms = get_test_chroms(args, fold)
    chrom_len = {}
    with open(chrom_sizes_file) as f:
        for line in f:
            name, length = line.strip().split("\t")
            chrom_len[name] = int(length)

    observed = pyBigWig.open(str(observed_bw_path))
    bias = pyBigWig.open(str(bias_bw_path))
    obs_chroms = observed.chroms()
    bias_chroms = bias.chroms()

    write_chroms = [c for c in test_chroms if c in chrom_len and c in obs_chroms]

    out = pyBigWig.open(str(corrected_bw), "w")
    out.addHeader([(c, chrom_len[c]) for c in write_chroms])
    #observed minus bias clipped to 0
    for chrom in write_chroms:
        length = chrom_len[chrom]
        obs_vals = np.nan_to_num(
            np.array(observed.values(chrom, 0, length), dtype="float64"), nan=0.0)
        if chrom in bias_chroms:
            bias_vals = np.nan_to_num(
                np.array(bias.values(chrom, 0, length), dtype="float64"), nan=0.0)
        else:
            bias_vals = np.zeros(length, dtype="float64")
        corrected = obs_vals - bias_vals
        corrected[corrected < 0] = 0.0 # clip to 0
        out.addEntries(chrom, 0, values=corrected.tolist(), span=1, step=1)

    observed.close()
    bias.close()
    out.close()
    print(f"wrote corrected target bigwig: {corrected_bw}")
    return corrected_bw





# now here is the tiling of windows for the prediction run, not the results --> on test ofc
def make_chrombpnet_pred_regions(chrombpnet_data, test_species, chrom_sizes_file, out_bed, chroms=None,      
                                 input_len=chrombpnet_input_len,
                                 output_len=chrombpnet_output_len):
    """
    check firstif it already exists 
    build 10 column narrowPeak bed that tiles the test chromosomes so that chrombpnets output windows cover them
    step by output_len  -> consecutive output windows 
    each summit noods input_len // 2 of flank on both sides --> cant start at 0 
    """
    out_bed = Path(out_bed)
    if out_bed.exists():
        return out_bed

    # which chromosomes are the test set for test species --> chnaged so we first check for chroms otherwise default to fold_0
    if chroms is None:                                           
        folds_path = utils.construct_folds_path(chrombpnet_data, test_species)
        with open(folds_path) as f:
            chroms = json.load(f)["test"]

    # chromosome lengths so we know where each chromosome ends
    chrom_len = {}
    with open(chrom_sizes_file) as f:
        for line in f:
            name, length = line.strip().split("\t")
            chrom_len[name] = int(length)

    half_in = input_len // 2   
    half_out = output_len // 2   

    rows = []
    name_i = 0
    for chrom in chroms:
        if chrom not in chrom_len:
            continue
        length = chrom_len[chrom]
        last_summit = None
        # summit is  half of input length because it needs the entire input length and then. we center output on that 
        summit = half_in
        while summit + half_in <= length:
            start = summit - half_out # output window start
            end = summit + half_out # output window end
            # 10 cols: chr start end name score strand signal p q summit_offset
            # start + summit_offset = summit and akso add some place holders
            rows.append((chrom, start, end, name_i, 0, ".", 0, 0, 0, half_out))
            name_i += 1
            last_summit = summit # new variable for writing that alst window
            summit += output_len  # step to the next output window

        end_summit = length - half_in 
        if last_summit is not None and end_summit > last_summit: # if these are  not that same append this
            # pred bw knows hwo to handle overlapping regions it takes the predictions of the window where the position si closer to the center so no worry here
            start = end_summit - half_out 
            end = end_summit + half_out
            rows.append((chrom, start, end, name_i, 0, ".", 0, 0, 0, half_out))
            name_i += 1

    out_bed.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bed, "w") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    return out_bed

# here we add the thing for deciding which bw to build 
def resolve_target_bw(args, fasta_path, chrom_sizes_file, bed_path, output_dir, chroms, fold=None):
    bigwig = str(args.bigwig)
    if bigwig.endswith("raw_cutsites"):
        return construct_chrombpnet_target_bw(args.chrombpnet_data, args.test_species, fold)
    elif bigwig.endswith("tobias"):
        # aberage ones
        if using_averaged(args):
            return construct_averaged_tobias_bw_path(args, args.test_species, fold)
        # merged ones
        return construct_tobias_bw_path(args, args.test_species, fold)
    # wrong ones
    else:
        return build_corrected_target_bw(args, fasta_path, chrom_sizes_file, bed_path=bed_path, output_dir=output_dir, fold=fold)


# here we actually run it
def run_pred_bw(model_path, bed_path, fasta_path, chrom_sizes_file, output_dir):
    """
    run chrombpnet pred_bw on the tiled regions and write a predicted bigwig
    BLOCKING (subprocess.run waits) so the results step after only runs once pred_bw is done
    --> this is why the whole script has to sit on a gpu node
    """
    out_prefix = Path(output_dir) / "pred"

    # lets hope this works
    pred_bw = Path(output_dir) / "pred_chrombpnet.bw"
    if pred_bw.exists():
        print(f"prediction already there, skipping pred_bw: {pred_bw}")
        return pred_bw

    cmd = [
        "chrombpnet", "pred_bw",
        "-cm", str(model_path),  # the model we picked
        "-r", str(bed_path), # our whole-test-chromosome tiles
        "-g", str(fasta_path), 
        "-c", str(chrom_sizes_file),
        "-op", str(out_prefix), # and here the ouput prefix
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)  # BLOCKING HERE
    return pred_bw # just the path 


# make pred and target over the tiles with same window as shorkie 

"""
one thing: windows will line up actually
the bigwig chrombpnet puts out, teh per psoition signal simply has nans for the startand end it didnt predict
so that will kill the pearson r in the firsta nd last window
but its nop cooridinate missmatch
"""

def save_results_npz(args, fasta_path, pred_bw_path, target_bw, output_dir, test_chroms):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_paths = [str(target_bw)]

    # same tiling as shorkie/baselines
    regions = tile_chromosomes(fasta_path, test_chroms, args.seq_length, skip_edges=304)

    # is only one target but keep it
    pred_handle = [pyBigWig.open(str(pred_bw_path))]
    target_handles = [pyBigWig.open(p) for p in target_paths]

    # keep windows separate (one row per window) just like the baseline eval
    all_preds, all_targets, all_meta = [], [], []
    for chrom, start in regions:
        end = start + args.seq_length
        # raw values, no log transform like in shorkie and its just one bigwig so this wont do anything
        if using_averaged(args):
            pred = observed_signal(pred_handle, chrom, start, end, log_transform=True)
            target = observed_signal(target_handles, chrom, start, end, log_transform=True)
        else:
            pred = observed_signal(pred_handle, chrom, start, end, log_transform=False)
            target = observed_signal(target_handles, chrom, start, end, log_transform=False)
        all_preds.append(pred)
        all_targets.append(target)
        #append the test species for this 
        all_meta.append({"species": args.test_species, "chrom": chrom, "start": int(start)})

    for h in pred_handle + target_handles:
        h.close()

    # stack along the window axis -> (n_windows x seq_length) 
    # this si the input we need for the evLUtion utils
    preds = np.stack(all_preds, axis=0)
    targets = np.stack(all_targets, axis=0)
    print(f"preds shape={preds.shape}, targets shape={targets.shape}")

    # same npz keys + metrics as shorkies outputs (metrics also because its just ncie to have them ready)
    # we take all of the methods from shorkie directly and save it 100% the same way 
    metrics = calculate_metrics(preds, targets)
    save_metrics(metrics, output_dir / "metrics.json")
    
    per_window_r = compute_per_window_pearson(preds, targets)
    save_evaluation_bundle(
        output_path=output_dir / "results.npz",
        preds=preds,
        targets=targets,
        metadata=all_meta,
        per_window_r=per_window_r,
        all_per_window_r=per_window_r,
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

# ------------------ train model from tobias bigwig ---------------


# this checks if the tobias bw exists --> if it doent it makes it
# then cklips at  0
# bigwig is saved in auxiliary of each model as "tobias_unstranded.bw"
def construct_tobias_bw_path(args, species, fold=None):

    model_dir = utils.construct_chrombpnet_model_dir(args.chrombpnet_data, species)
    tobias_bw = model_dir / data_fold_dir(fold) / "auxiliary" / "tobias_unstranded.bw" # --> fold 
    if tobias_bw.exists():
        print(f"tobias bw already there, skipping: {tobias_bw}")
        return tobias_bw

    bam = utils.construct_species_bam(args.chrombpnet_data, species)
    fasta = utils.construct_fasta_path(args.species_input_files, species)
    peaks = utils.construct_species_peaks(args.chrombpnet_data, species)
    chrom_sizes = utils.construct_chrom_sizes_path(args.species_input_files, species)

    # run the maketobias replicates before
    genome_bed = construct_genome_bed_path(args.species_input_files, species)

    tmp = tobias_bw.parent / "tobias_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    prefix = "atac"

    # here we also have to add genome bed that not only peaks are given back 
    cmd = [
        "TOBIAS", "ATACorrect",
        "--bam", str(bam),
        "--genome", str(fasta),
        "--peaks", str(peaks),
        "--regions-out", str(genome_bed),
        "--read_shift", "0", "0",   # our bam is ALREADY shifted -> so we do not double shift
        "--outdir", str(tmp),
        "--prefix", prefix,
        "--cores", "4",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)   # BLOCKING (cpu tool, no gpu needed)

    corrected = tmp / f"{prefix}_corrected.bw"   # of the 4 outputs, this is the bias-removed one
    if not corrected.exists():
        raise FileNotFoundError(f"TOBIAS did not write corrected bw: {corrected}")

    clip_bigwig_negatives(corrected, tobias_bw, chrom_sizes)
    print(f"wrote tobias corrected+clipped track: {tobias_bw}")
    return tobias_bw



# this is essentially the like drawn out method from tobias script to like cut the vals at 0 so we dotn have negatives
def clip_bigwig_negatives(in_bw_path, out_bw_path, chrom_sizes_file):
    chrom_len = {}
    with open(chrom_sizes_file) as f:
        for line in f:
            name, length = line.strip().split("\t")
            chrom_len[name] = int(length)

    bw_in = pyBigWig.open(str(in_bw_path))
    in_chroms = bw_in.chroms()
    write_chroms = [c for c in chrom_len if c in in_chroms]

    out = pyBigWig.open(str(out_bw_path), "w")
    out.addHeader([(c, chrom_len[c]) for c in write_chroms])
    for chrom in write_chroms:
        length = chrom_len[chrom]
        vals = np.nan_to_num(np.array(bw_in.values(chrom, 0, length), dtype="float64"), nan=0.0)
        vals[vals < 0] = 0.0
        out.addEntries(chrom, 0, values=vals.tolist(), span=1, step=1)
    bw_in.close()
    out.close()

# this si chrombpnets OWN METHOD simply copy paste
# because we need all params this si how they compute it
#Median total signal across peak windows / 10, floored at 1.0 
def compute_counts_loss_weight(bw_path, peaks_bed, inputlen):
    import pandas as pd
    bw = pyBigWig.open(str(bw_path))
    peaks = pd.read_csv(peaks_bed, sep="\t", header=None)
    totals = []
    for _, row in peaks.iterrows():
        center = int(row[1]) + int(row[9])          # start + summit
        start, end = center - inputlen // 2, center + inputlen // 2
        if start < 0:
            continue
        vals = np.nan_to_num(np.array(bw.values(row[0], start, end)))
        totals.append(vals.sum())
    bw.close()
    return max(round(float(np.median(totals)) / 10, 2), 1.0)


# build a params tsv like chrombpnet --> reuse all the same aprameters eccept the copunts loss weight because that depends 
# on the input data and we change that only 

def build_tobias_params(args, species, tobias_bw, fold_path, out_params, fold=None):
    model_dir = utils.construct_chrombpnet_model_dir(args.chrombpnet_data, species)
    src = find_one(model_dir / data_fold_dir(fold) / "logs", "*chrombpnet_model_params.tsv", "model params")

    keep = {}
    for line in open(src).read().strip().split("\n"):
        key, value = line.split("\t")
        if key in ("filters", "n_dil_layers", "inputlen", "outputlen",
                 "max_jitter", "negative_sampling_ratio"):
            keep[key] = value

    # bias_model_path is dropped because bpnet_model.py doesnt use it
    # here recompute the count loss weight and save the fold path
    peaks_bed = find_one(model_dir / data_fold_dir(fold) / "auxiliary", "filtered.peaks*", "filtered peaks")
    keep["counts_loss_weight"] = str(compute_counts_loss_weight(
        tobias_bw, peaks_bed, int(keep["inputlen"])))
    keep["chr_fold_path"] = str(fold_path)   # this has to equal the -fl passed to train 

    with open(out_params, "w") as f:
        for key, value in keep.items():
            f.write(f"{key}\t{value}\n")
    return out_params


"""
This chess first if all files exists then makes distinction if average is inn bigwig flag that was passed

normal -> single on merged, model chrombpnet_nobias_tobias.h5
average -> averaged replicate tobias track, model chrombpnet_nobias_tobias_averaged.h5

""" 

def construct_tobias_model_path(args, fold=None):
    model_dir = utils.construct_chrombpnet_model_dir(args.chrombpnet_data, args.train_species)

    # pick training track + matching model/params names
    if using_averaged(args):
        tobias_bw = construct_averaged_tobias_bw_path(args, args.train_species, fold)
        model_file = "chrombpnet_nobias_tobias_averaged.h5"
        params_file = "tobias_averaged_model_params.tsv"
    else:
        tobias_bw = construct_tobias_bw_path(args, args.train_species, fold)
        model_file = "chrombpnet_nobias_tobias.h5"
        params_file = "tobias_model_params.tsv"

    model_path = model_dir / data_fold_dir(fold) / "models" / model_file
    if model_path.exists():
        print(f"tobias model already there: {model_path}")
        return model_path

    # the train/test split for THIS fold (train species)
    if fold is None:
        fold_json = utils.construct_folds_path(args.chrombpnet_data, args.train_species)
    else:
        fold_json = loco_kfold.construct_loco_fold_path(args.chrombpnet_data, args.train_species, fold)

    # reuse the edge filtered peaks/nonpeaks from the fold auxiliary
    fasta = utils.construct_fasta_path(args.species_input_files, args.train_species)
    aux = model_dir / data_fold_dir(fold) / "auxiliary"
    peaks = find_one(aux, "filtered.peaks*", "filtered peaks")
    nonpeaks = find_one(aux, "filtered.nonpeaks*", "filtered nonpeaks")

	
    # build the params file
    params = build_tobias_params(args, args.train_species, tobias_bw, fold_json, model_dir / data_fold_dir(fold) / "logs" / params_file, fold=fold)


    # get the bpnet architecture / the file and run command
    bpnet_arch = Path(chrombpnet.__file__).parent / "training" / "models" / "bpnet_model.py"
    out_prefix = model_path.with_suffix("")   # train.py already adds the .h5
    cmd = [
        "python", "-m", "chrombpnet.training.train",
        "-g", str(fasta),
        "-b", str(tobias_bw),
        "-p", str(peaks),
        "-n", str(nonpeaks),
        "-fl", str(fold_json),
        "-pf", str(params),
        "-a", str(bpnet_arch),
        "-o", str(out_prefix),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)            # BLOCKING here we ahve to make some time GPU training befoer i start evaluation script
    if not model_path.exists():
        raise FileNotFoundError(f"training did not write the model: {model_path}")
    return model_path

# "averaged replicates" mode is triggered by the bigwig label, e.g. --bigwig average_bigwig/tobias
def using_averaged(args):
    return "average" in str(args.bigwig)

# build once the averaged tobias track from the per-replicate tobias bigwigs
def construct_averaged_tobias_bw_path(args, species, fold=None):
    model_dir = utils.construct_chrombpnet_model_dir(args.chrombpnet_data, species)
    out_bw = model_dir / data_fold_dir(fold) / "auxiliary" / "tobias_averaged_unstranded.bw"
    if out_bw.exists():
        print(f"averaged tobias bw already there, skipping: {out_bw}")
        return out_bw

    # the per-replicate tobias bigwigs shorkie/baselines use --> MAKE SURE TO USE TOBAIS ONES
    rep_dir = utils.construct_bigwig_dir(args.species_input_files, "replicate_bigwigs/tobias", species)
    rep_bws = sorted(str(p) for p in Path(rep_dir).glob("*.bw"))
    if not rep_bws:
        raise FileNotFoundError(f"no repliate tobias bigwigs in {rep_dir}")

    chrom_sizes = utils.construct_chrom_sizes_path(args.species_input_files, species)
    out_bw.parent.mkdir(parents=True, exist_ok=True)
    average_bigwigs(rep_bws, out_bw, chrom_sizes)
    print(f"wrote averaged tobias track ({len(rep_bws)} replicates): {out_bw}")
    return out_bw



# average several bigwigs into one, per base then clip at 0
# this is the baseline averaging (np.mean + clip) without log1p, because chrombpnet trains on counts --> we log insresults instead

def average_bigwigs(in_bw_paths, out_bw_path, chrom_sizes_file):
    chrom_len = {}
    with open(chrom_sizes_file) as f:
        for line in f:
            name, length = line.strip().split("\t")
            chrom_len[name] = int(length)

    handles = [pyBigWig.open(str(p)) for p in in_bw_paths]

    # only keep chroms that exist in the sizes file AND in every replicate
    common = set(chrom_len)
    for h in handles:
        common &= set(h.chroms())
    write_chroms = [c for c in chrom_len if c in common]

    #add header and write 
    out = pyBigWig.open(str(out_bw_path), "w")
    out.addHeader([(c, chrom_len[c]) for c in write_chroms])
    for chrom in write_chroms:
        length = chrom_len[chrom]
        # one row per replicate, then mean down the replicate axis so we have the averaging then
        stack = []
        for h in handles:
            vals = np.nan_to_num(np.array(h.values(chrom, 0, length), dtype="float64"), nan=0.0)
            stack.append(vals)
        mean_vals = np.mean(stack, axis=0)   # same averaging as observed_signal
        mean_vals[mean_vals < 0] = 0.0   # same clip as observed_signal
        out.addEntries(chrom, 0, values=mean_vals.tolist(), span=1, step=1)

    for h in handles:
        h.close()
    out.close()

# this runs one fold

def run_one_fold(args, fasta_path, chrom_sizes_file, fold):
    # get test chroms
    test_chroms = get_test_chroms(args, fold)

    # construict the output path
    shared = Path(args.output_folder) / "chrombpnet" / args.train_species / args.test_species / args.bigwig
    out_dir = shared / output_fold_seg(fold)
    out_dir.mkdir(parents=True, exist_ok=True)

    # tile just test chroms
    bed_path = construct_pred_bed_path(args.species_input_files, args.test_species, fold)
    make_chrombpnet_pred_regions(args.chrombpnet_data, args.test_species,
                                 chrom_sizes_file, bed_path, chroms=test_chroms)

    # this fold model + predbw 
    model_path = construct_chrombpnet_model_path(args, fold)
    pred_bw_path = run_pred_bw(model_path, bed_path, fasta_path, chrom_sizes_file, out_dir)

    # target track
    target_bw = resolve_target_bw(args, fasta_path, chrom_sizes_file,
                                  bed_path, out_dir, test_chroms, fold)

    # score -> results.npz + metrics.json
    save_results_npz(args, fasta_path, pred_bw_path, target_bw, out_dir, test_chroms)


# make this one much shorter --> run it for just fold_0 if we dont have folds aka run for fold none that equates to it
def main():
    args = parse_args()
    chrom_sizes_file = utils.construct_chrom_sizes_path(args.species_input_files, args.test_species)
    fasta_path = utils.construct_fasta_path(args.species_input_files, args.test_species)

    for fold in compute_fold_indices(args, chrom_sizes_file):
        run_one_fold(args, fasta_path, chrom_sizes_file, fold)


if __name__ == "__main__":
    main()



