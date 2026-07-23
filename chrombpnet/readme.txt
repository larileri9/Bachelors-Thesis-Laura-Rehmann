In here:


pipelines.py --> simply the args.plus_shift and args.minus_shift not hardcoded
reads_to_bigwig.py --> here it takes effect


4 scripts as examples for the versions of the chrombpnet data pipeline I used:

00: Data processing merged and singular replicates.
01: Chrombpnet training merged and singular replicates.

--> singular replicates were run to obtain shifted bigwigs 
--> peak and nonpeak windows are obtained form the merged run

--> the input file names for everything are sort of fixed and only change by species  



For everything besides schep it looks like this:


import os

folds_dir = "/s/project/multispecies/fungi_code/atac/data/benchmarking/chrombpnet_data_kfold/loco_folds/n_crassa"
folds = glob_wildcards(os.path.join(folds_dir, "fold_{fold}.json")).fold

folds = [0,1,2,3,4,5,6] # folds flexible per species


root_dir = "/s/project/multispecies/fungi_code/atac/data/benchmarking/chrombpnet_data_kfold/n_crassa"
input_root = "/s/project/multispecies/fungi_code/atac/data/benchmarking/chrombpnet_data/n_crassa" 
out_dir = "params_2"


BLACKLIST = "/s/project/multispecies/fungi_code/atac/data/benchmarking/species_input_files/blacklist/n_crassa_blacklist.bed"
CHROM_SIZES = "/s/project/multispecies/fungi_code/atac/data/benchmarking/species_input_files/chrom_sizes/n_crassa_chrom_sizes.txt"
FASTA = "/s/project/multispecies/fungi_code/atac/data/benchmarking/species_input_files/genomes/n_crassa_genome.fna"


--> all file paths are the same unit the short species name which has to be substituted e.g. c_albicans 