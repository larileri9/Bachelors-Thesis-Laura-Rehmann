# explantion of files + general

# workflow of baselines kind of modeled after shorkie 


# train_benchmark_model.py // call to launch model with parameters
# train_benchmark_dataset.py // dataset calls for baseline
# train_benchmark_regression.py // baseline architecture
# train_benchmark_utils.py // some utils for paths etc 



Flags that can be adjusted when running

--model: thats the model types that are valid  (gc, gc_kmer, embeddings, shorkie, shorkie_scratch)
--species: full name with underscore and shortened internally
--bigwig: folder name under species_input_files/bigwigs/ (can be merged or replicate_bigwigs and tobias or raw_cutsite at least thats how I did it) (can also be chrombpnet_bias_corrected)
--chrombpnet-data: root for the peak and non-peak beds, the fold jsons and the chrom sizes (need to run chrombpnet first)
--species-input-files: root for genomes, bigwigs and embeddings
--model-output-folder: output root, becomes root/model/species/bigwig and /fold_i appended if kfold is run
--seq-length: shorkie only can be changed default is 16384
--kfold: switches from one holdout fold to leave-one-chromosome-out and generates folds and configs, then submits one job per fold
--run-fold: a single fold index in case a fold didnt work or was cancelled or smth, oes nothing without --kfold
--chrombpnet-bias-corrected: old flag for the chrombpnet bass correctedd bigwig essentially dead --> instead we regulate this through the bigwig name

Internal hardcoded variables: 

chrombpnet_edge_trim: 304 because 1216 - 608 and passed as --edge-trim when the bigwig string matches
config_path: holdout_singular.yaml, hardcoded absolute path
params_path: shorkie_params.json, hardcoded absolute path
pretrained_path: model_best.h5, hardcoded absolute path
baseline_models: "gc", "gc_kmer", "embeddings"
shorkie_models: "shorkie", "shorkie_scratch", "shorkie_embeddings"
--max-epochs: 100 for every species 
--batch-size: 4
--lr: 2e-5
--warmup-steps: 5000
--num-workers: 8
--precision: bf16-mixed
--peak-centered: always on, so shorkie never trains on random windows
--freeze-backbone-epochs: 5 
shorkie sbatch: partition noninterruptive, gpu:1, 8 cpus, 64G, 4:00:00 
baseline sbatch: partition standard, no gpu, 4 cpus, 64G, 04:00:00
input_length_cbp: 1216,
output_length_cbp: 608
log_transform: True by default 
kmer_len: --kmer-len is in the regression script with default 3 



Launcher steps:

Always:

  1. parse_args runs and logs/ directory is created
  2. for embeddings and shorkie_embeddings species_input_files/species_lm_embeddings/full_species_name is checked 
  3. construct_bigwig_dir shotens species and builds species_input_files/bigwigs/bigwig/s_cerevisiae


1. Baselines (gc, gc_kmer, embeddings)

   Both:
     1. run_baseline is called
     2. get the fasta file
     3. fold json is loaded and only folds["train"] are kept (val chromosomes are discarded here)
     4. the command is assembled with feature, fasta, bigwig dir, both beds, output pkl and the train chromosomes
     5. for embedings --embedding-root and --species are appended
     6. --edge-trim 304 is appended if the bigwig string ends in chrombpnet_bias_corrected
     7. command is wrapped in sbatch + submitted
     8. In job: train_benchmark_regression.py sets 1216 and 608 and gets bws
     9. read_bed_regions reads beds computes start plus summit_offset
    10. BaselinePositionDataset puts whole genome into a dict, every bigwig opened, then per regoin it builds feature and reads the target, crops both to the central 608 and apends
    11. stack_windows concatenates into X and y
    12. Linear Regression and pkl is written

   A) Non kfold
     1. contstruct_peak_nonpeak_filepath is called with default fold=0 so it reads fold_0/auxiliary/ under chrombpnet data (which was trained on that specific holdout fold and only that fixed holdout fold))
     2. the outpt directory model_output_folder/model/species/bigwig is created
     3. run_baseline is called with no folds_path, so it defaults to chrombpnet_data/folds/{species}/fold_0.json
     4. fold_tag is empty, so no fold suffix appears in the log filename

   B) KFOLD
     1. construct_chrom_sizes_path builds species_input_files/chrom_sizes/{species}_chrom_sizes.txt and checks it exists
     2. loco_kfold.ensure_folds generates or finds the leave-one-chromosome-out fold JSONs and returns their paths
     3. k is set to the number of fold paths returned
     4. fold_indices is a single numbr list if --run-fold was given, otherwsie range(k)
     5. loop begins and everything below repeats once per fold
     6. construct_peak_nonpeak_filepath is called with fold=i, so it reads fold_i/auxiliary/
     7. output dir model_output_folder/model/species/bigwig/fold_i is created
     9. loco_kfold.construct_loco_fold_path returns the fold json
    10. fold_tag is _foldi and goes into the log filename


2. SHORKIE (shorkie, shorkie_scratch, shorkie_embeddings)

   Both:
     1. config_path and params_path are checked + if shorkie pretrained_path is checked
     2. run_shorkie is called
     3. seq_tag is empty at 16384 and otherwise _length
     4. the command for train.py get assembled and --edge-trim 304 is appended if the bigwig string ends in chrombpnet_bias_corrected
     5. Variant flags are appended: shorkie: --pretrained, --original-input, --freeze-backbone-epochs 5. shorkie_scratch: --original-input
     6. command is wrapped in sbatch and train.py is run

   A) NON KFOLD
     1. contstruct_peak_nonpeak_filepath is called with default fold=0 so it reads fold_0/auxiliary/ under chrombpnet data (which was trained on that specific holdout fold and only that fixed holdout fold))
     2. the outpt directory model_output_folder/model/species/bigwig is created
     3. run_shorkie iis called with hardcoded holdout_singular.yaml

   B) KFOLD
     1. construct_chrom_sizes_path builds species_input_files/chrom_sizes/{species}_chrom_sizes.txt and checks it exists
     2. loco_kfold.ensure_folds generates or finds the leave-one-chromosome-out fold JSONs and returns their paths
     3. k is set to the number of fold paths returned
     4. loco_kfold.ensure_config runs once per fold and makes a per-fold config from the stff in holdout_singular.yaml
     5. fold_indices is a single numbr list if --run-fold was given, otherwise range(k)
     6. the loop begins and everything below repeats once per fold
     7. contstruct_peak_nonpeak_filepath is called with fold=i, so it reads fold_i/auxiliary/
     9. the output directory model_output_folder/model/species/bigwig/fold_i is created
    10. loco_kfold.construct_loco_config_pathreturns the per fold conig from step 4
    11. fold_tag is _foldi and goes into log filename and wandb



