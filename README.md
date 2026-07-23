# Bachelor's Thesis — Laura Rehmann

## Repository structure

### [codes/shorkie_modified](shorkie_modified)

Slightly tweaked version of Max's Shorkie model. Includes workflow for evaluating entire chromsomes, peak centered training and additional flags for input feature switches / training window lengths etc. 

### [codes/train_launcher](train_launcher)

Scripts to launch all model trainings except chrombpnet. Chrombpnet needs to be run first to obtain the windows for other models to train on. Can be used to get kfold or holdout trainings.

### [codes/eval_launcher](eval_launcher)

Scripts to evaulate all models except chrombpnet in either kfold or holdout setting. 

### [codes/additional_utils](codes/additional_utils)

Chromosme embedding generation, ATACorrect ec.

### [codes/chrombpnet](chrombpnet)

Workflow to evaluate chrombpnet on entire chromosome using region tiling and pred_bw command. Trains tobias model if model hasnt been trained yet. 

### [chrombpnet](chrombpnet)

Version of preprocessing files I used (merged and singular replicate) and of the training to obtain the bigwigs + results.


Workflow / Order of calling:


Dependency:

```
step 0    prerequisites     file preparation
step 0.5  kfold folds       made by whichever launcher runs first
step 1    run chrombpnet    gives the bigwigs, peaks and non peaks
step 2    run embeddings    only if embeddings are used
step 3    tobias bigwigs    needs 1
step 4    training          needs 0 to 3 depending on model and bigwig
step 5    evaluation        needs 4
step 6    chrombpnet        needs 3 for tobias
```

--> all models done



Step 0:

Prepare the fixed files and download data + set up the file structures (adjust in code if needed). 
- download genome.fna + index 
- make chrom sizes (only chroms we use + names need to fit)
- download SSR runs
- create blacklists
- fixed folds split + fixed holdout_singular.yaml (created by hand 70/15/15 size split and pointing to all files prepared)



Step 1:

Run chrombpnet on the single fold. Because the order is a bit off and to create the folds you will have to run any launcher preferably train or evaluate first which generates the folds and point the chrombpnet pipeline to them. 

Run the 00 and then the 01 pipelines for merged and singular replicates.

Merged is then used for the peak and non peak files as well as copying the params.tsv and the results are used for raw data. 

Singular replicates produce a uniform format cut site bigwig + bams which I simply hand copied once the step for copying them into an input dir i did not automate.

Run chrombpnet_benchmarking_workflow.py as specified to obtain raw result npzs.

Step 2:

Generate the embeddings using per_chromosme_embeddings.py

Step 3: 

With make_tobias_replicate_bigwigs.py you can create the necessary Tobias bigwigs for Tobias training. 

Step 4: 

Train with the launcher all models and species and bigwig combinations you want.

Step 5:

Evaluation of all models trained.

Step 6:

Run chrombpnet_benchmarking_workflow.py with the average bigwigs flag to train and evaluate a BPNet architecture for tobias.


Multispeceis step:

For multispecies holdout yamls were made by hand and contain chrombpnet peaks and non peaks for every species from chrombpnet step. Separate from the launcher, scripts and explanations are included in the modified_shorkie section.