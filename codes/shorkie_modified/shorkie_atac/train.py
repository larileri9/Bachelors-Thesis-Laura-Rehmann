"""Training script for ShorkieATAC post-training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

import torch

try:
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger, WandbLogger
except ImportError:
    from lightning import Trainer
    from lightning.pytorch import seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import CSVLogger, WandbLogger

from torch.utils.data import DataLoader

from atac_dataset import build_datasets_from_folds
from shorkie_atac import ShorkieATAC
from atac_dataset_modified_2 import  build_datasets_from_folds_modified


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-train ShorkieLM on ATAC-seq data."
    )

    # Data
    parser.add_argument(
        "--bigwig-dir", type=str, required=True,
        help="Directory containing ATAC-seq bigwig files.",
    )


    # Model
    parser.add_argument(
        "--params", type=str, default="configs/baskerville/shorkie_params.json",
        help="Path to Shorkie model params JSON.",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pretrained ShorkieLM .h5 weights.",
    )
    parser.add_argument(
        "--num-tracks", type=int, default=1,
        help="Number of output tracks.",
    )

    # Training
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=5000, help="Warmup steps.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0,
                        help="Freeze backbone for N initial epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max training epochs.")
    parser.add_argument("--train-samples", type=int, default=10000,
                        help="Number of training samples per epoch.")
    parser.add_argument("--val-samples", type=int, default=2000,
                        help="Number of validation samples per epoch.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers.")
    parser.add_argument("--log-transform", action="store_true", default=True,
                        help="Apply log1p to ATAC-seq signal (default: True).")
    parser.add_argument("--no-log-transform", dest="log_transform", action="store_false",
                        help="Disable log1p transform.")
                        # new args here
    parser.add_argument("--species-lm-embedded", action="store_true", default=False,
                        help="Instead of using One hot, use species lm embeddings.")
    parser.add_argument("--entire-chromosome", action="store_true", default=False,
                        help="Generate windows spanning the entire chromosme. Disable random window sampling in evaluation.")
    parser.add_argument("--embedding-root", type=str, default=None,
                        help="Root dir / input dir of Species LM embeddings. Required when --species-lm-embedded. Were precomputed per species")
    parser.add_argument("--original-input",action="store_true", default=False,
                        help="Use 6 DNA channels (A,C,G,T,N,mask) instead of 4. Closer to pretraining encoding but no actual species token here.")

# this is an old one unused --> leaving this in in case I still do chrombpnet correction
    parser.add_argument("--skip-conv-dna", action="store_true", default=False,
                        help="Replace conv_dna with a small linear adapter. --> embeddings already have context",)
    parser.add_argument("--embedding-replaced-head", action="store_true", default=False,
                        help="Skip the backbone and predict ATAC seq directly from Species LM embeddings with FinalHead. CAREFUL NEEDS --species-lm-embedded.")
    # for chrombpnet bias corrected ONLY --> since chrombpnet doesnt predict edge 304 need to trim those in loss because they are 0 
    parser.add_argument("--edge-trim", type=int, default=0, help="Ignore this many bp at each chromosome end in the loss.")

    # idk if sequence needs to be hard coded lets check that: --> verdict doesnt 

    parser.add_argument("--seq-length", type=int, default=16384,
                    help="Input window length. Should be no problem if divisible by 128 (transformer output length so we can upsample 7 times).")

    parser.add_argument("--peak-centered", action="store_true", default=False,
                        help="Center the windows on peak summits instead of random sampling during training")

# these will be the ones from chrombpnet
    parser.add_argument("--peak-bed", type=str, default=None, help="Peak narrowPeak BED (needs 10 columns).")
    parser.add_argument("--nonpeak-bed", type=str, default=None,
                        help="Non-peak narrowPeak bed (also needs 10 columns).")

    # Infrastructure
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory for checkpoints and logs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--accelerator", type=str, default="auto",
                        help="PyTorch Lightning accelerator (auto, gpu, cpu).")
    parser.add_argument("--precision", type=str, default="32",
                        help="Training precision (32, 16-mixed, bf16-mixed).")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0,
                        help="Gradient clipping value.")

    # Config & logging
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--species", type=str, default=None,
        help="Specific species to train on. If None, trains jointly on all species.",
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="shorkie-atac",
        help="W&B project name (default: shorkie-atac).",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="W&B run name (default: auto-generated).",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None,
        help="W&B entity (team or user).",
    )

    return parser.parse_args()


# just hard code this in here, we could pass it as an extra flag but for my use it works its not user friendly though
# also we really have to correct neurospora crassa 
species_token_dict = {
    "candida_albicans" : "candida_albicans",
    "saccharomyces_cerevisiae": "_saccharomyces_cerevisiae",
    "schizosaccharomyces_pombe": "schizosaccharomyces_pombe",
    "aspergillus_niger": "aspergillus_niger",
    "aspergillus_oryzae": "aspergillus_oryzae",
    "neurasposa_crassa": "neurospora_crassa"}




def main() -> None:
    args = parse_args()

    # Set seed --> workers true means background processes also use it like data loading and stuff
    seed_everything(args.seed, workers=True)

    # open the config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # get the one species we run otherwise it takes all 
    # HERE WE TAKE ONLY THE ONE SPECIES FOR TESTING --> in train.sh define that species as train species
    species_to_run = [args.species] if args.species else list(cfg.get("species", {}).keys())
    
    #only train and val here 
    train_datasets = []
    val_datasets = []

    # number of worked does forking
    # while gpu trains it prepares other batches already like dataset wise on CPU !!!
    # but the embedder sits on gpu so we can for that down there, we could add it in thta open files thing like bigwigs
    # bvut we would have many computationally expensive copies --> just disable workeds
    actual_num_workers = args.num_workers
    # now we precompute embeddings dont need this anymore
    #if args.species_lm_embedded:
     #   actual_num_workers = 0   # CUDA model in datasetso we cant fork workers
      #  logger.info("species_lm_embedded=True: setting num_workers=0")


    for sp in species_to_run:
        # get species confg
        sp_cfg = cfg.get("species", {}).get(sp)
        if not sp_cfg:
            logger.warning("Species %s not found in config.", sp)
            continue
    
        sp_token = species_token_dict[sp]
        
        #samples are replicates 
        sample_ids = [str(s) for s in sp_cfg.get("samples", [])]
        fasta_path = sp_cfg.get("fasta")
        folds_path = sp_cfg.get("folds")

        if args.peak_bed is not None:
            peak_bed = args.peak_bed
            non_peak_bed = args.nonpeak_bed
        else:
            peak_bed = sp_cfg.get("peaks")
            non_peak_bed = sp_cfg.get("non_peaks")

        # if we asked for peak centering we really need those files, so fail early and clearly
        if args.peak_centered and (peak_bed is None or non_peak_bed is None):
            raise ValueError(
                f"peak_centered is on but {sp} has no peak files. "
                f"Add 'peaks' and 'non_peaks' under this species in the config."
            )

        logger.info("Building dataset for %s...", sp)
        # here we dont restrict ourselves to the complete chromsome --> more data for training??
        if args.original_input:
            num_channels = 6
        else:
            num_channels = 4
        
        ds = build_datasets_from_folds_modified(
            bigwig_dir=args.bigwig_dir,
            fasta_path=fasta_path,
            folds_path=folds_path,
            seq_length=args.seq_length,
            train_samples=args.train_samples // len(species_to_run),
            val_samples=args.val_samples // len(species_to_run),
            seed=args.seed,
            log_transform=args.log_transform,
            sample_ids=sample_ids,
            species=sp,
            species_token=sp_token, # new flags added here + token 
            species_lm_embedded = args.species_lm_embedded, 
            entire_chromosome = args.entire_chromosome,
            embedding_root=args.embedding_root,
             peak_centered=args.peak_centered,
            peak_bed=peak_bed,
            non_peak_bed=non_peak_bed,
            num_channels=num_channels, 
            edge_trim=args.edge_trim, 
        )
         # this is essentially per split and per fold datastes as far as i can see it but idk if thats truly it?
         # also we only have one fold usually 
        train_datasets.append(ds["train"])
        val_datasets.append(ds["valid"])

    #concat the species datasets
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets)
    valid_dataset = ConcatDataset(val_datasets)

    # train and val loader + model 
    # just data loaders to batch feed data
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=actual_num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=actual_num_workers,
        pin_memory=True,
    )


    # this herebuilds the model with params
    logger.info("Building model...")
    model = ShorkieATAC(
        params_file=args.params,
        pretrained_h5=args.pretrained,
        num_tracks=args.num_tracks,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        weight_decay=args.weight_decay,
        species_lm_embedded=args.species_lm_embedded,
        original_input=args.original_input,     #--> pass these
        skip_conv_dna=args.skip_conv_dna, 
        embedding_replaced_head=args.embedding_replaced_head, 
    )

    #callback explanation
    #lightning calls at specific like methods in training
    # a callback is essentially one of these methods modified 
    # examples are 
    # ModelCheckpoint: Saves the best model weights automatically
    # EarlyStoppingStops training if metric stops improving
    # LearningRateMonitorLogs LR to TensorBoard/WandB
    # RichProgressBarNicer progress bar
    # here: callbacks=[checkpoint_cb, lr_monitor] --> so we have these methods sort of modified and then with callback we change that 

    #Adjustments here
    # monitor="val/loss" mode="min" --> watches valdiation loss and lower values is better
    # save_top_k=3 -->  3 best checkpoints plus last.ckpt are saved
    # Callbacks --> this si where we save stuff essentially
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="shorkie_atac-{epoch:03d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Loggers
    loggers = [CSVLogger(save_dir=str(output_dir), name="logs")]
    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            save_dir=str(output_dir),
            log_model=False,
        )
        # Log all hyperparameters to W&B
        wandb_logger.log_hyperparams(vars(args))
        loggers.append(wandb_logger)
        logger.info("W&B logging enabled (project=%s).", args.wandb_project)

    # Trainer
    # trainer.fit() does the full training loop that is predefined by lighning

    '''

    does smth like this: (i hope)
    for epoch in range(max_epochs):
    model.on_train_epoch_start()          
    for batch in train_loader:
        batch onto GPU
        loss = model.training_step(batch, idx) --> trainign step here
        loss.backward()                   ----> this is gradients
        clip_gradients(gradient_clip_val)
        optimizer.step()                  update weights
        scheduler.step()                   update learning rate
        optimizer.zero_grad()             clear the gradients again
        log metrics every 10 steps

    for batch in val_loader:
        with torch.no_grad():               -----> automatically so we cant make that mistake
            model.validation_step(batch, idx)
    get metrics
    ModelCheckpoint.on_validation_end()   -->save if improved
    '''
    
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        callbacks=[checkpoint_cb, lr_monitor],
        logger=loggers,
        log_every_n_steps=10,
        default_root_dir=str(output_dir),
    )

    # make sure that i find this with --> just print out that im 100% sure everything is correct
    # should be using % correctly here
    logger.info("x" * 60) # some separation bar to find it
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info(" species: %s", args.species)
    logger.info(" species_lm_embedded: %s", args.species_lm_embedded)
    logger.info(" original_input: %s", args.original_input)
    logger.info(" skip_conv_dna: %s", args.skip_conv_dna)
    logger.info(" num_channels: %d", num_channels)
    logger.info(" freeze_backbone: %d epochs", args.freeze_backbone_epochs)
    logger.info(" num_features in model: %d", model.num_features)
    logger.info("x" * 60)


    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    logger.info("Training complete. Checkpoints saved to %s", output_dir / "checkpoints")


if __name__ == "__main__":
    main()
