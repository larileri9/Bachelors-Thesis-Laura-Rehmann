# explanation 

--> here is how to launch the training of chrombpnet tobias and evaluataion of the cross species experiments
--> tobias model gets trained automatically if not present



Scripts:

Launches Tobias training but can also launch replicate bigwigs (raw):

cbp_tobias_cross.sh
cbp_tobias_kfold.sh

Evaluation script and Tobias model trainer:
chrombpnet_benchmarking_workflow.py


Flags:

--train-species: picks which trained chrombpnet model gets used and which species the tobias track gets trained on
--test-species: sets which genome gets predicted on and scored against ( train species when kfold is on)
--bigwig: picks the model file and the target track (raw_cutsites, tobias or chrombpnet_bias_corrected) Anything with average in the string switches to the averaged replicate track
--chrombpnet-data: where the trained models, folds, bams and peaks are read from
--species-input-files: root for genomes, bigwigs, chrom sizes, genome beds and the pred region beds
--output-folder: evaluation root the reslts get written under
--seq-length: window length for the scoring tiles, has to match what shorkie and the baselines used
--kfold: runs every loco fold into its own dir. Sets test species to train speices
--run-fold: runs one fold index instead of all 

Hardcoded variables:

chrombpnet_input_len: 1216, used for the prediction tiling
chrombpnet_output_len: 608, used for the prediction tiling 
skip_edges: 304
model file per bigwig: raw_cutsites gets chrombpnet.h5, chrombpnet_bias_corrected gets chrombpnet_nobias.h5, tobias gets a model this script trains
tobias model files: chrombpnet_nobias_tobias.h5 (for merged) or chrombpnet_nobias_tobias_averaged.h5 for the averaged track
tobias track names: tobias_unstranded.bw  (merged) and tobias_averaged_unstranded.bw, both in auxiliary
TOBIAS read_shift: 0 0, since the bam is already shifted
TOBIAS cores: 4
params carried over from chrombpnet: filters, n_dil_layers, inputlen, outputlen, max_jitter, negative_sampling_ratio
counts_loss_weight: recomputed as median total signal over peak windows divided by 10, floored at 1 (from chrombpnet)
bpnet architecture: takn from the installed chrombpnet package
log transform in scoring: on only when average is in the bigwig string, off otherwise and only results



Workflow:

Both:
  1. args are parsed and with kfold the test species is set to the train species
  2. chrom sizes file and genome are built for the test species
  3. without kfold fixed holdout is run, with a given fold index the one fold index is run and with kfold alone all folds are run


Everything up to bigwig selection:
  1. test chroms read out of the fold json
  2. output dir built under the evaluation root with train species, test species and bigwig in it (and + fold)
  3. prediction region bed path built for the test species (tiles of 1216 that chrombpnet predicts)
  4. if that bed already exists it gets reused as is, otherwise it gets written
  5. chrom lengths out of the chrom sizes file
  6. per test chrom, the first center or like pseudosummit is placed at 608 because a summit needs a full input window around it and each row written as 10 narrowPeak columns, output window from summit minus 304 to summt plus 304
  7. Windows are sliding by 608, so conscutive output windows sit edge to edge with no gap and looping stops once summit plus 608 would run past chrom end
  8. one extra row appended at the last possible summit, chrom length minus 608, so the right edge is covered


After picking model and target:
  1. pred_bw skips if pred_chrombpnet.bw is already in the output dir
  2. chrombpnet pred_bw runs with the model, the bed, the genome, the chrom sizes and an output prefix
  3. the predicted bigwig and the target bigwig are opened and per tile and values pulled from btoh over the same coordinates, nans turned into zeros, clipped at 0, then log transform applied only when average is in the bigwig string
  4. each window is kept as its own row, plus a metadata entry with test species, chrom and start and everything stacked into windows by positions
  5. metrics are computed and written to metrics.json and per window pearson computed and everyhing written to results.npz with the same keys shorkie uses


1. Raw cutsite (no training)

   Model
     1. model root built from the train species short name and the single dir inside it is taken
     2. the fold dir is chosen, fold_0 without kfold and fold_i with it
     3. chrombpnet.h5 is picked out of models. this is the full model (predicts tn5 bias along with the signal)

   Target
     1. model root built again, this time from the test species short name + same fold dir is entered
     2. data_unstranded.bw is taken out of auxilary, that is the observed cut site track chrombpnet itself trained against, so it carries the bias too

   Notes:
     1. nothing is built and nothing is trained, both files already exist from the original chrombpnet run
     2. both tracks are raw counts eachfold is one pred_bw call


2. Tobias

   Target track / Training track
     1. target path is auxiliary/tobias_unstranded.bw of the train species in this fold dir (if it is already there everything below is skiped)
     2. merged deduped shifted bam, genome, peaks and chrom sizes collected for the train species from "raw cutsite" run
     3. a tmp dir is made next to where the track will land
     4. Tobias ATACorrect runs with the bam, the genome, the peaks and a genome bed so regions outside peaks come back too and read shift is set to 0 0 because the bam is already shifted
     5. of the four files TOBIAS writes, the corrected one is taken --> negatives clipped to zero and written out as the final track

   Params
     1. the original chrombpnet params tsv for this fold is taken frm logs
     2. six values copied out of it, filters, n_dil_layers, inputlen, outputlen, max_jitter, negative_sampling_ratio (bias_model_path is not copied)
     3. counts_loss_weight gets recomputed, because it depends on the signal scale and the signal has changed (filtered peaks bed is used, each peak centred on start plus summit offset, a window of inputlen taken around it and values summed. The median across all peak windows divided by 10, rounded to two decimals, floored at 1 is the counts_loss_weight)
     4. chr_fold_path set to the fold json, which has to match what gets passed to training
     5. written out as a tsv next to the original one

   Model
     1. model path is models/chrombpnet_nobias_tobias.h5 of the train species in this fold dir
     2. if it is already there everything below is skipped
     3. fold json picked for the train species, fold_0 without kfold and the loco one with the fold
     4. genome for the train species constructed and filtered peaks and filtered nonpeaks taken from auxiliary of this fold
     5. bpnet architecture file taken out of the installed chrombpnet package
     6. chrombpnet.training.train runs with the genome, the tobias track as the signal, the peaks, the nonpeaks, the fold json, the params tsv, the architecture and an output prefix. blocking, needs the gpu

   Target track for test species
     1. same seven steps as the train species track, run again with the test species bam, genome, peaks and chrom sizes (skipped if that track already exists)

   Notes
     1. the first run for a species pair is long, the track build and the training both block 
     2. every run after that goes straight to pred_bw, since the track, the params and the model each check for their own output file first
     3. log transform is off unless average is in the bigwig string, which also swaps in the averaged replicate track and chrombpnet_nobias_tobias_averaged.h5