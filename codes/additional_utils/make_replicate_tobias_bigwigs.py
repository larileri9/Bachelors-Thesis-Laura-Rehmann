"""
This script generates the per replicate TOBIAS corrected bigwigs for all six species

Two things:

1. runs TOBIAS ATACorrect on that replicates shifted bam and uses merged peaks of the species and takes the "_corrected.bw" output
2. clip negative values of that corrected track at zero and writes genome wide bigwig into the folder the benchmrking workflow reads from

Some notes:

bams already +4/-5 shifted --> --read_shift is 0 0 and no second shifting

Usage
one species --> : python make_replicate_tobias_bigwigs.py --species n_crassa
all six --> : python make_replicate_tobias_bigwigs.py --species all
Small dry run flag just for testing: --dry-run
Overwriting flag: --force

"""

import argparse
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pyBigWig



# the two roots because we did move the chrom_bp_pipeline out of data cuz that wasnt the right place but the singular replicates live there

BENCHMARK_ROOT = Path("/s/project/multispecies/fungi_code/atac/data/benchmarking")
PIPELINE_ROOT = Path("/s/project/multispecies/fungi_code/atac/cross_species_benchmark/chrom_bp_pipeline")

# where the benchmark has genomes, chrom sizes and raw bigwigs
SPECIES_INPUT_FILES = BENCHMARK_ROOT / "species_input_files"

# this wis where merged peaks live
CHROMBPNET_DATA = BENCHMARK_ROOT / "chrombpnet_data"

# per species: the sub path of the singular replicate pipeline and the replicates.
# a replicate is either SSR or GSM and S.cer has both sometimes so we need to make sue the names are stated here
SPECIES = {
    "s_cerevisiae": {
        "subpath": "GSE66386/control_pipeline_11",
        "samples": [
            ("SRR1822137", "GSM1621323"), "GSM1621324", "GSM1621325", "GSM1621326", "GSM1621327", "GSM1621328", "GSM1621329", "GSM1621330", "GSM1621331", "GSM1621332"],
    },
    "s_pombe": {
        "subpath": "pombe/chrombpnet_all_reps_all_fold",
        "samples": [ "SRR1822148", "SRR1822149", "SRR1822150", "SRR1822151", "SRR1822152"],
    },
    "c_albicans": {"subpath": "c_albicans",
        "samples": ["SRR12487669", "SRR12487670", "SRR12487671"],
    },
    "a_niger": {
        "subpath": "a_niger",
        "samples": ["SRR10193421", "SRR10193422"],
    },
    "a_oryzae": {
        "subpath": "a_oryzae",
        "samples": ["SRR10403107", "SRR10403108"],
    },
    "n_crassa": {
        "subpath": "n_crassa",
        "samples": ["SRR12229299", "SRR12229300"],
    },
}


def replicates(species):
    # for each species gets the samples and makes the names
    for entry in SPECIES[species]["samples"]:
        if isinstance(entry, tuple):
            yield entry
        else:
            yield entry, entry


# build paths
def fasta_path(species):
    # s_cerevisiae is a .fa, the other five are .fna
    for extension in (".fna", ".fa"):
        p = SPECIES_INPUT_FILES / "genomes" / f"{species}_genome{extension}"
        if p.exists():
            return p
    # nothing found, return the .fna so the error message names a concrete path that we can then check
    return SPECIES_INPUT_FILES / "genomes" / f"{species}_genome.fna"


def chrom_sizes_path(species):
    return SPECIES_INPUT_FILES / "chrom_sizes" / f"{species}_chrom_sizes.txt"


def bam_path(species, bam_dir):
    root = PIPELINE_ROOT / SPECIES[species]["subpath"]
    return (root / "singular_replicates" / bam_dir / "results" / "05_merged"/ "all_merged.sorted.shifted.bam")


def species_bw_dir(species):
    #one .bw per replicate, because the workflow doesPath(rep_dir).glob("*.bw")
    return (SPECIES_INPUT_FILES / "bigwigs" / "replicate_bigwigs" / "tobias"/ species)


def out_bw_path(species, bw_name):
    return species_bw_dir(species) / f"{bw_name}.bw"


def genome_bed_path(species):
    return SPECIES_INPUT_FILES / "genome_beds" / f"{species}_genome.bed"


def build_genome_bed(species):
   
    #bed covering every chromosome end to end, built from the chrom sizes file for regions out
   
    bed = genome_bed_path(species)
    sizes = chrom_sizes_path(species)

    bed.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(sizes) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, length = line.split("\t")
            rows.append((name, 0, int(length)))

    with open(bed, "w") as f:
        for name, start, end in rows:
            f.write(f"{name}\t{start}\t{end}\n")

    # ATACorrect drops these names, so warn if one is in the bed 
    dropped = {"chrM", "chrMT", "M", "MT", "Mito"}
    hits = [r[0] for r in rows if r[0] in dropped]
    if hits:
        print(f"Careful ATACorrect drops these: {hits}")
    return bed


def peaks_path(species):
    # need to try different setups strain g with balcklist because some fps differ
    candidates = [
        CHROMBPNET_DATA / species / "results" / "06_peaks" / "all_merged_relaxed_peaks_no_blacklist.bed",
        CHROMBPNET_DATA / species / "06_peaks" / "all_merged_relaxed_peaks_no_blacklist.bed",
        CHROMBPNET_DATA / species / "results" / "06_peaks" / "all_merged_relaxed_peaks.narrowPeak",
        CHROMBPNET_DATA / species / "06_peaks" / "all_merged_relaxed_peaks.narrowPeak",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"no merged peak file found for {species}")


# run ATACorrrect finally 
def run_atacorrect(bam, fasta, peaks, regions_out, tmp_dir, prefix, cores, read_shift):

    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "TOBIAS", "ATACorrect",
        "--bam", str(bam),
        "--genome", str(fasta),
        "--peaks", str(peaks),
        "--regions-out", str(regions_out),
        "--read_shift", str(read_shift[0]), str(read_shift[1]),
        "--outdir", str(tmp_dir),
        "--prefix", prefix,
        "--cores", str(cores),
    ]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)   # blocking, CPU only, no GPU needed

    corrected = tmp_dir / f"{prefix}_corrected.bw"
    # check
    if not corrected.exists():
        raise FileNotFoundError(f"ATACorrect did not write {corrected}")
    return corrected


# clip negatives and write --> clip negatives because negative cut sites impossible

def clip_bigwig_negatives(in_bw_path, out_bw_path, chrom_sizes_file):

    chrom_len = {}
    with open(chrom_sizes_file) as f:
        for line in f:
            name, length = line.strip().split("\t")
            chrom_len[name] = int(length)

    bw_in = pyBigWig.open(str(in_bw_path))
    in_chroms = bw_in.chroms()
    write_chroms = [c for c in chrom_len if c in in_chroms]

    if not write_chroms:
        bw_in.close()
        raise ValueError(
            f"no chromosome of {chrom_sizes_file} present in {in_bw_path}. "
            "the chrom names of fasta and the chrom sizes file disagree."
        )

    out_bw_path.parent.mkdir(parents=True, exist_ok=True)
    out = pyBigWig.open(str(out_bw_path), "w")
    out.addHeader([(c, chrom_len[c]) for c in write_chroms])

    for chrom in write_chroms:
        length = chrom_len[chrom]
        vals = np.nan_to_num(
            np.array(bw_in.values(chrom, 0, length), dtype="float64"), nan=0.0
        )
        vals[vals < 0] = 0.0
        out.addEntries(chrom, 0, values=vals.tolist(), span=1, step=1)

    bw_in.close()
    out.close()

# function for each replicate
def do_one_replicate(species, bam_dir, bw_name, peaks, regions_out, cores, force,
                     dry_run, keep_tmp, read_shift):
    bam = bam_path(species, bam_dir)
    fasta = fasta_path(species)
    chrom_sizes = chrom_sizes_path(species)
    out_bw = out_bw_path(species, bw_name)

    print(f"\n[{species} / {bam_dir} -> {bw_name}.bw]")
    print(f"bam {bam}")
    print(f"peaks {peaks}")
    print(f"regions-out {regions_out}")
    print(f"shift forward {read_shift[0]}, reverse {read_shift[1]}")
    print(f"out {out_bw}")

    if out_bw.exists() and not force:
        print("already exists use --force to overwrite")
        return

    # collect problems instead of raising on the first one, so dry runshows everything 
    problems = []
    for p in (bam, fasta, chrom_sizes, peaks):
        if not Path(p).exists():
            problems.append(f"missing input: {p}")
    if not Path(str(bam) + ".bai").exists():
        problems.append(f"BAM index missing: {bam}.bai)")

    if dry_run:
        for msg in problems:
            print(f"PROBLEM: {msg}")
        if not problems:
            print("no problems")
        return

    if problems:
        raise FileNotFoundError("\n".join(problems))

    tmp_dir = out_bw.parent / f"tobias_tmp_{bw_name}"
    corrected = run_atacorrect(bam, fasta, peaks, regions_out, tmp_dir, "atac", cores, read_shift)

    clip_bigwig_negatives(corrected, out_bw, chrom_sizes)
    print(f" wrote {out_bw}")

    if not keep_tmp:
        shutil.rmtree(tmp_dir)

# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True, help="one species key or all")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="overwrite bigwigs that already exist")
    ap.add_argument("--dry-run", action="store_true", help="dry run")
    ap.add_argument("--keep-tmp", action="store_true", help="keep the four raw ATACorrect outputs")
    ap.add_argument("--read-shift", nargs=2, type=int, default=[0, 0], metavar=("FORWARD", "REVERSE"), help="shift Tobias should apply to reads")
    args = ap.parse_args()

    if args.species == "all":
        species_list = list(SPECIES)
    elif args.species in SPECIES:
        species_list = [args.species]
    else:
        raise SystemExit( f"unknown species")

    for species in species_list:
        try:
            # once for species
            peaks = peaks_path(species)    
        except FileNotFoundError as e:
            if args.dry_run:
                print(f"\n[{species}]\n  PROBLEM  {e}")
                continue
            raise

        # genome wide --regions-out built once per speices
        print(f"\n[{species}]")

        regions_out = build_genome_bed(species)

        for bam_dir, bw_name in replicates(species):
            do_one_replicate(
                species=species,
                bam_dir=bam_dir,
                bw_name=bw_name,
                peaks=peaks,
                regions_out=regions_out,
                cores=args.cores,
                force=args.force,
                dry_run=args.dry_run,
                keep_tmp=args.keep_tmp,
                read_shift=tuple(args.read_shift),
            )

    print("done.")


if __name__ == "__main__":
    main()
