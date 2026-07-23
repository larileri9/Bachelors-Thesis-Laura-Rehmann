# explanation

--> some utils that are needed or called in other scripts 

# ----- create chromsome embeddings --------- --> has to be run explicitely

per_chromosome_embeddings.py --> launch the per chromsome embedder, calling instructions in file
Species_lm_embeddings.py --> code I was given just the model class of the embedder wrapped


FLAGS, per_chromosme_embeddings.py

--fasta: genome fasta 
--species: the species token handed to the tokenizer (_saccharomyces_cerevisiae for cerevisiae)
--output-dir: one .pt per chromosome lands here (add species subduer in path)


FIXED INTERNALLY, per_chromosme_embeddings.py

chunk_size: 1000 = how many bases go into the model at once
stride: 500 = how far the window shifts
context_window: derived as (1000 - 500) // 2 = 250 --> changes with stride
model_path: absolute path to the species_upstream_1000_k1 checkpoint
kmer_size: 1 =  single bases rather than kmers
device: cuda --> needs a gpu node
padding base: N, 250 at each end of the chromosome and however many are needed to fill the last window
output name: {chrom}.pt matching the fasta entry name 

--> I did not change any configurations for the model only the tiling


WORKFLOW, one species

  1. call per_chromosme_embeddings.py once per species with its fasta, species token and output dir
  2. script makes the output dir + the model and the tokenizer once (reuses them for everything)
  3. inside: fasta is opened and every entry in it looped --> per chromosome if .pt is alerady there --> skip
  4. whole chromosome sequence is fetched as a string + 250 N are added at each end
  5. window slides along the padded sequence in steps of 500 each window is padded_seq[i : i+1000] and if sequence ends is filled with N to 1000
  6. the window and the species token go into the tokenizer (batch dimnsion is added by hand)
  7. the window is run through the model with a hook on the encoder block we want --> activations get captured as the forward pass goes through it
  8. the model output is discarded, only the captured activation is used (1003 rows come back, row 0 = classification token, row 1 = species token,
     last row = separator. --> all are cut leaving the 1000 base rows (768 x 1000)
  9. only the middle 500 rows are kept, emb[250:750]
 10. on the last window only we cut window so nothing past the chromosome end gets saved
 11. .pt written


# -------   make_replicate_tobias_bigwigs.py: ------------- --> has to be run explicitely

--species: one key of the species dict or all ( short names like s_cerevisiae)
--cores: how many cores 
--force: overwrite bigwigs that are already there
--dry-run: dry run
--keep-tmp: keep the four raw ATACorrect outputs instead of deleting the tmp dir
--read-shift: two integers, the shift Tobias should apply 



BENCHMARK_ROOT: absolute path to the benchmarking dir
PIPELINE_ROOT: absolute path to the singular replicate pipeline
SPECIES: the six species with their pipeline subpath and their replicate list using short names
bam path: species subpath / singular_replicates / replicate dir / results/05_merged/all_merged.sorted.shifted.bam
output dir: species_input_files/bigwigs/replicate_bigwigs/tobias/{species} --> one .bw per replicate
peaks: peak files get fetched
genome bed: species_input_files/genome_beds/{species}_genome.bed
ATACorrect prefix: atac, so the file read back is atac_corrected.bw
mito names warned about: chrM, chrMT, M, MT, Mito



Always:
  1. args parsed and species is either one key or all six --> loop over the species list

Per species:
  1. merged peaks fetched
  2. genome bed written from the chrom sizes 
  3. mito names warned about since ATACorrect drops them
  4. loop over the replicates of that species

Per replicate: 
  1. bam, genome, chrom sizes and output bigwig paths built (if the output bigwig is there and force is off, this replicate is skipped)
  2. all four inputs plus the bam index checked, every missing one collected --> with dry run the problems printed
  3. tmp dir made next to where the bigwig will land
  4. TOBIAS ATACorrect runs with the bam, the genome, the merged peaks, the genome bed as regions-out, the read shift and the core count
  5. atac_corrected.bw îs used --> then:  nans set to zero, negatives set to zero, written out as the final bigwig
  6. tmp dir deleted unless keep-tmp is on




# ---- creating the kfold jsons and yamls --> never has to be run explicitly is always run internally 


Only loco_kfold_splits.py is called loco_kfold.py just has helper methods.

Flags:

--chrom-sizes: path to chrom sizes file
--species-short: short name
--output-dir: folds land in output-dir/species-short/fold_i.json
--group-by-prefix: collapse I_1 and I_2 into one chromosome before splitting
--size-matched-k: pack into this many size matched folds instead of one chromosome per fold

Internally hardcoded:

ROMAN_1_20: the twenty roman numerals I to XX --> for dropping non chroms
valid split: always the next group after the test group (wrapping around at the end)
minimum k: 3 on the size matched path
maximum k: the number of units --> no more folds than there are chroms 
output filename: fold_i.json
fold shape: three keys, test, valid and train, each a sorted list of real chrom names





Workflow: --> a launcher runs with --kfold triggers it

  1. the launcher calls loco_kfold.ensure_folds with the chrombpnet data root, the species and the chrom sizes path
  2. ensure_folds looks in chrombpnet_data/loco_folds/{species}/ for anything called fold_*.json
  3. if it finds them --> early exit return them, otherwise:
  4. checks whether this species is one of the two exceptions, then calls generate_species_folds
  5. generate_species_folds reads the chrom sizes file into a name to length dict
  6. drops anything not roman numeral (scaffolds mito etc)
  7. group_units maps each unit to its real chromosome names and to its total bp (also collapses e.g. I_1 and I_2 collapse into one unit)
  8. then depending on size_matched_k:
     if set: k at least 3 and at most the number of units, then partition_size_matched sorts greedy by length
     if not: one_per_group wraps each unit in its own list 
  9. build_folds gets list of lists --> for group i (test) it takes valid as i+1 wrapped, and train as remaining group
 10. expand turns the unit names back into real chromosome names and sorts them (I becomes I_1 and I_2)
 11. sanity_check asserts the three splits do not overlap and that test and train are not empty
 12. one json per fold gets written to chrombpnet_data/loco_folds/{species}/fold_i.json
 13. ensure_folds hands the paths back and the launcher which loops over the folds
 14. for shorkie the launcher calls ensure_config once per fold:
 15. ensure_config looks for chrombpnet_data/loco_configs/{species}/fold_i.yaml --> early exit if there
 16. otherwise it loads the template yaml (holdout singular), repoints its folds entry at this fold json and writes a one species yaml
 17. the baselines get the fold json path and read the split out of it // shorkie gets the yaml path 

