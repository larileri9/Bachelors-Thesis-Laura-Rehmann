# explantion of files + general

# workflow of evaluation for models trained


# eval_benchmark_model.py // call to launch evaluation with parameters
# eval_benchmark_dataset.py // dataset calls for baseline
# eval_benchmark_regression.py // baseline architecture  


Flags:


--train-species: which species model to score on
--test-species: sets which species the data is read, so bigwigs, fasta, folds, embeddings, chrom sizes and is set to train-species when kfold is on
--model: gc, gc_kmer, embeddings, shorkie, shorkie_scratch 
--bigwig: pass the folder under species_input_files/bigwigs/ that fits the trained model
--chrombpnet-data: where the fold json files are read from
--species-input-files: root for genomes, bigwigs and embeddings
--model-output-folder: where the trained model is loaded from
--evaluation-output-folder: where results.npz and metrics.json are written
--seq-length: window length used for tiling
--config: shorkie config yaml. Replaced by the per fold config when kfold is on
--params: shorkie_params.json
--skip-edges: tiling runs from skip_edges to chrom_len minus skip_edges (here fixed because chrombpnet doesnt predict them)
--kfold: evaluates the leave one chromosome out setup and sets test species to train species
--run-fold: evaluates one fold index instead of all of them
--overwrite: reruns folds that already have results.npz and metrics.json


Internal hardcoding:

config_path_default: holdout_singular.yaml
params_path_default: shorkie_params.json
chrombpnet_edge_trim: 304, the default value for skip-edges
baseline_models: gc, gc_kmer, embeddings
shorkie_models: shorkie, shorkie_scratch, shorkie_embeddings
--save-all-samples: always passed to evaluate.py
--run-plotting 0: always passed to evaluate.py
--auto-benchmark 0: always passed to evaluate.py
feature_window: read from the pkl
kmer_len: read from the pkl



Both:
  1. args are parsed and with kfold the test species is set to the train species
  2. logs dir creation + bigwig dir for the test species is built and checked
  3. for embeddings: embedding dir for test species is checked
  4. output dir built, skipped if results are already there and overwrite is off


1. Baselines (gc, gc_kmer, embeddings)

   Both
     1. test species genome found by trying .fna then .fa + fold json read for the test chroms
     2. get trained .pkl for this model and train species 
     3. command built from .pkl + genome + bigwigs + output dir + seq length + skip edges + test chroms
     4. for embeddings: embedding root and species added
     5. in the job: feature window and kmer length read back out of the .pkl so the features come out like they did in training
     6. test chroms tiled into seq length windows from skip edges to chrom length mnius skip edges
     7. all windows built up front
     8. targets read from the bigwigs, averaged over replicates, log transformed
     9. each window predicted on its own and kept as its own row, so per window pearson lines a window up against itself
    10. predictions and targets stacked into windows by positions
    11. metrics and results file written with the same keys shorkie uses

   A) Non kfold
     1. one pass and no fold anywhere in the paths
     2. fold json is fold_0 of the chrombpnet folds and test chroms are the test split of that fold

   B) kfold
     1. test species is the train species so this stays inside one speices
     2. with a given fold index that one fold runs alone (--run-fold), without one the chrom sizes are checked and the folds are checked and if they dontt exist are written
     3. fold json is the loco one for the fold
     4. test chroms are the single held out chrom and fold goes into the .pkl path and into the output dir


2. Shorkie (shorkie, shorkie_scratch)

   Both
     1. config and params checked
     2. checkpoint dir for this model and train species built and checked
     3. checkpoints searched recursively --> lowest loss wins
     4. command for evaluate.py bult from checkpoint + config + params + bigwigs + output dir + test species + seq length + skip edges
     5. 170 channel input flag added so the data matches what the model was trained on
     6.in the job: checkpoint loaded and model rebuilt from the hyperparameters saved inside it
     7. dataset built always with whole chromosome tiling so the windows match what the baselines get
     9. every window predicted+  metrics and results file written

   A) Non kfold
     1. one pass and no fold anywhere in the paths
     2. config is the one passed in (holdout_singular.yaml)and test chroms come out of that config instead of a fold json

   B) Kfold
     1. test species is the train species
     2. with a given fold index (--run-fold) that fold runs, without the chrom sizes are checked and the folds are checked and made if they dont exist
     3. config is the loco one for the fold so the held out chrom changes per fold
     4. fold number goes into the checkpoint path and into the output dir