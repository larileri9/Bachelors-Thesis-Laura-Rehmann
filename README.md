# Bachelor's Thesis — Laura Rehmann

In this git are codes and more in depth per experiment results. I am curerntly still cleaning up and uploading codes + writing some explanations.

.. working on it, will be added gradually in the two weeks after thesis deadline :( ...
## Repository structure

### [shorkie_modified](shorkie_modified)

Slightly tweaked version of Max's Shorkie model. Includes workflow for evaluating entire chromsomes, peak centered training and additional flags for input feature switches / training window lengths etc. 

### [train_launcher](train_launcher)

Scripts to launch all model trainings except chrombpnet. Chrombpnet needs to be run first to obtain the windows for other models to train on. Can be used to get kfold or holdout trainings.

### [eval_launcher](eval_launcher)

Scripts to evaulate all models except chrombpnet in either kfold or holdout setting. 

### [codes/additional_utils](codes/additional_utils)

Chromosme embedding generation, ATACorrect ec.

### [chrombpnet](chrombpnet)

Workflow to evaluate chrombpnet on entire chromosome using region tiling and pred_bw command. Trains tobias model if model hasnt been trained yet. 
### [results](results)

Markdown files with per experiment values.
