# explanation of files + general

- most of the code is taken from max, I have made some changes in workflow and input features 

Changes:

train.py --> bed file opening for peaks and non peaks
shorkie_atac.py --> idk grade
atac_dataset_modified.py --> optional tiling for chromosomes, input feature decision, peak centering
evaluate.py --> always tiles test chromosomes

evaluation_utils.py --> nothing

General information:

- dead code left in for the chrombpnet corrected bigwig and the skip conv and embedding head 
- I think I have restructured it here so it is runnable with filepath and dir changes obviously

Example scripts:

- xxx.sh --> example script for training multispecies
- xxx.sh --> example script for evaluating multispecies

--> training and evaluation of single species is conducted through the launchers 


Flag list:


Flags I use:

--pretrained (best model h5)
--freeze-backbone-epochs 
--seq-length (window length) 
--max-epochs
--checkpoint (I use this instead of autobenchamrk and added a function that gets me the best checkpoint per species)
--save-all-samples
--bigwig-dir 
--config 
--species



Flags I use that I have added:

--original-input (170 vector input without assigned species token)

--> if not present and not --species-lm-embedded --> uses one hot only first weights never loaded

--entire-chromosome (windows tiling the entire chromosome)
--peak-centered (windows centred on summits read from column 10 of the peak and non-peak bed files) 
Peak centred looses to entire chromosome if both are on , but one if for training one for evaluation.
--peak-bed / --nonpeak-bed (overrides the peaks and non_peaks entries in the config if in config)
--shuffle-input (test with shuffled input for leakage)




Dead flags of mine: (would need revising before reinstating not sure if they all still work didn't check)


--species-lm-embedded (uses embeddings as input)
Requires:
	--embedding-root (directory)
	--species (name, token gets assigned within careful is hardcoded)

--skip-conv-dna (would've skipped first layer without weights and projected embeddings into input format of res tower)
--embedding-replaced-head (linear head on embeddings)
--edge-trim (loss masking of beginning an end of chromosome)
