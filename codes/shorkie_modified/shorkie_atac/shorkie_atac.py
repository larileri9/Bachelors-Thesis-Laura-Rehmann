from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_lightning import LightningModule
except ImportError:
    from lightning import LightningModule

from genome_lm.train.models.components.baskerville import (
    BaskervilleTorchModel,
    build_torch_model_from_baskerville_params,
    FinalHead, # Added this to be able to make the head only modification
)
from genome_lm.util.baskerville_weight_loader import load_tf_weights

logger = logging.getLogger(__name__)


# this is the actual model
#LightningModule is the pytorch wrapper aournd nn.Module
#It kind of round sup the model, the training/validation steps, and the optimizer configuration alkl together
# with this you dont ahve to write an actual training loop
class ShorkieATAC(LightningModule):
    """Shorkie model fine-tuned for ATAC-seq coverage prediction.

    Args:
        params_file: Path to shorkie_params.json (model architecture config).
        pretrained_h5: Path to pretrained ShorkieLM .h5 weights (TF format).
        num_tracks: Number of output tracks (e.g., 1 for averaged signal,
            or N for per-sample prediction).
        learning_rate: Learning rate for fine-tuning.
        warmup_steps: Number of warmup steps.
        freeze_backbone_epochs: Freeze the backbone for this many initial epochs.
        weight_decay: AdamW weight decay.
    """

    def __init__(
        self,
        params_file: str,
        pretrained_h5: str | None = None,
        num_tracks: int = 1,
        learning_rate: float = 2e-5,
        warmup_steps: int = 5000,
        freeze_backbone_epochs: int = 0,
        weight_decay: float = 0.0,
        species_lm_embedded = False,
        original_input = False,    # NEW — use 6 channels instead of 4
        skip_conv_dna = False,
        embedding_replaced_head = False, # this will NOT replace species lm embed we keep that but its additional to skip backbone
    ):
        super().__init__()
        #self.save_hyperparameters() saves all hyperparameters so a model can be reconstructed from checkpoint
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.weight_decay = weight_decay
        self.num_tracks = num_tracks
        self.species_lm_embedded = species_lm_embedded
        self.original_input = original_input
        self.skip_conv_dna = skip_conv_dna
        self.embedding_replaced_head = embedding_replaced_head

        # Load model config (configs/baskerville/shorkie_params.json)
        # self.model_cfg is dict of model params
        with open(params_file) as f:
            raw_params = json.load(f)
        self.model_cfg = raw_params["model"]

        # Build backbone (backbone) --> make copy of model_cfg override fatures there 
        cfg = dict(self.model_cfg)


        # --------------  change of embeddign here --------------------
        if self.species_lm_embedded:
            cfg["num_features"] = 768 # this is embedding length per position
            logger.info("we are doing 768 features")
        elif self.original_input:
            cfg["num_features"] = 170  # here we add mask and pretend species token to match the pretrained model better 165 species and 4 bases and 1 mask and 1 unmapped
            logger.info("we are doing 170 features")
        else:
            cfg["num_features"] = 4  # A, C, G, T (no species embedding for fine-tuning)
            logger.info("we are doing 4 features")
        
        self.num_features = cfg["num_features"]
        
        

        # This way we dont ovverrite the complete complete model backbone
        # If the pretrained Shorkie model was originally trained with the species embedding 
        # --> this ovverides any sort of embedding we would have made

        # Override the head for regression 
        # searches for first head we find

        """
        this is ion shorkie_params.json
        "head_human": {
            "name": "final",
            "units": 4,
            "activation": "softmax"
        --> we ovveride


        this head is a linear head then in the config 
        - trunck out but is 384 feature vector at 1bp resolution
        --> this is where embeddings would go in
        in shorkie params we use all 7 unet layers so the 16p bin stop does not exist we go to 8 -> 4 -> 2 -> 1
        they stop at 3 layers we do all 7 ...
        ill jsut plug it in there with the 1bp resolution
        """
        head_key = next((k for k in cfg if k.startswith("head")), None)
        if head_key:
            # make a copy again
            cfg[head_key] = dict(cfg[head_key])
            # num tracks is number of tracks to be predicted we want one
            cfg[head_key]["units"] = num_tracks
            #log(1 + exp(x)) is softplus its relu but tweaked becase poisson loss needs üpositive values and this can never be negative
            cfg[head_key]["activation"] = "softplus"
        """
        Makes the full neural network from the  now adapted config
        build_torch_model_from_baskerville_params reads the config dict and  then constructs nn.Module layers from what is specified in config
        self.backbone turns into initialized pytorch module with randomly initialized weights
        --> will be rained or have the pretrained weights loades after 
        pytorch nn.Module:  base class for all neural networks in pyorch.
        when you do self.backbone = build_torch_model_from_baskerville_params(cfg), PyTorch automatically registers it as a submodule of ShorkieATAC, meaning its parameters become part of self.parameters() automatically.
        """


        """
        input: (B, 16384, 4) -->  one-hot DNA 
        conv_dna ->  res_tower -> transformer_tower -> u net_conv -> FinalHead 
        conv_dna: learns dna motifs
        res_tower: residual convolutional blocks (i think are for short sequences and short interactions?)
        transformer_tower transformer blocks with relative position bias (long range interaczoions?)
        unet_conv: kind of doubles the output taht was downsized before back to full input length?
        finka head: linear projection + softplus activation
        output: (B, 16384, 1) batch, seq_len, per position signal
        

        """

        '''
        slightly lengthy trying change

        idea: conv_dna glides with kernel 11 so sort of context size 11 over the input matrix along sequence
        now what this is supposed to do is to learn dna motifs but this does not really add anything worthwhile with embedding
        because like its 1D convolutional 
        --> it learns dna motifs but technically embeddings already have context and they are just huge idk if this could really do that much ...??
        so what i want to try is to skip this and add a linear layer its not much difference its not much work 



        '''
       
        if self.skip_conv_dna:
            #res_tower expects 96  filters as input 
            #got taht from the shorkei param config
            self.conv_dna_filters = int(cfg["trunk"][0].get("filters", 96))

            #linear adapter sort of projects each position from input space to conv_dna output space
            # matrix multiply per position
            # B, L, num_features wird zu B, L, 96
            self.embedding_adapter = nn.Linear(self.num_features, self.conv_dna_filters)

            # build then the backbone WITHOUT conv_dna
            # so we remove trunk[0] (conv_dna) from config and then tell backbone its first layer (res_tower) expects 96 channels as input
            cfg_no_conv = dict(cfg)
            cfg_no_conv["trunk"] = cfg["trunk"][1:] #take out the conv_dna   
            cfg_no_conv["num_features"] = self.conv_dna_filters  # res_tower expects 96
            # build the instance of model with new config 
            self.backbone = build_torch_model_from_baskerville_params(cfg_no_conv)

        # here the embedding injection with disabled backbone  --> actually useless so leave here dead code but afraid of destroying smth  
        elif self.embedding_replaced_head:
            # pure replacing: predict directly from the embeddings, backbone not necessary
            # FinalHead does transpose -> Linear(in_ch -> units) -> activation,
            # so in_ch must equal the embedding width
            assert self.species_lm_embedded, \
                "embed_replace_head only makes sense with species_lm_embedded (768-d input)"
            self.embed_head = FinalHead(
                self.num_features,   # 768 because we do embeddings
                num_tracks,
                "softplus",
                "lecun_normal",
            )
            self.backbone = None  
        else:

            # --> returns an instance of BaskervilleTorchModel
            self.backbone = build_torch_model_from_baskerville_params(cfg)

        # Load pretrained weights if provided --> also amke sure to skip if we dont have a backbone
        if pretrained_h5 is not None: 
            if not self.embedding_replaced_head:
                self._load_pretrained(pretrained_h5)

        # Track whether backbone is frozen
        self._backbone_frozen = False



#load_tf_weights translates .h5 weights into hwo pytorch needs it
#Reads  weight arrays from h5 file
#copies them into the PyTorch models state dict
#strict=False means: if a key doesnt match it skips it instead of raising an error


    """
    def _load_pretrained(self, h5_path: str) -> None:
        #Load pretrained TF weights, ignoring head mismatches.
        #logger.info("Loading pretrained weights from %s", h5_path)

        # Build the original LM config to get correct weight mapping


        # ---> unsure if i need these these look scary

        lm_cfg = dict(self.model_cfg)
        #lm_cfg["num_features"] = 4 

        # I looked into the h5 

        # and the first input of the conv_dna is (11, 170, 96)
        #(kernel_size=11, in_channels=170, out_filters=96)
        # so this i think would then missmatch mit the constrcuted num features 4 ...?
        #conv_dna weight shape: torch.Size([96, 4, 11]) --> this is our trunk 0 ...

        # Use the weight loader with strict=False to allow head mismatch 
        try:
            unmapped = load_tf_weights(
                self.backbone, h5_path, lm_cfg, strict=False
            )
            if unmapped:
                logger.info("Unmapped keys (expected for new head): %s", unmapped)
            logger.info("Successfully loaded pretrained weights.")
        except Exception as e:
            logger.warning("Weight loading had issues: %s", e)
            logger.info("Initializing from scratch for mismatched layers.")

    """

    def _load_pretrained(self, h5_path: str) -> None:
        """Load pretrained TF weights, ignoring head mismatches."""
        logger.info("Loading pretrained weights from %s", h5_path)

        lm_cfg = dict(self.model_cfg)

        if self.skip_conv_dna:
            # backbone starts at res_tower so trunk[0] again is now res_tower not conv_dna
            #give weight loader a different config which also starts at res toweer
            #  our trunk.0 = pretrained trunk.1
            lm_cfg_for_loading = dict(lm_cfg)
            lm_cfg_for_loading["trunk"] = lm_cfg["trunk"][1:]  # drop conv_dna again
            lm_cfg_for_loading["num_features"] = self.conv_dna_filters #and adjuts the input
            cfg_to_use = lm_cfg_for_loading
        else:
            cfg_to_use = lm_cfg

        try:
            unmapped = load_tf_weights(
                self.backbone, h5_path, cfg_to_use, strict=False
            )
            if unmapped:
                logger.info("Unmapped keys (skipped): %s", unmapped)
            logger.info("Successfully loaded pretrained weights.")

            if not self.skip_conv_dna:
                conv_dna_shape = tuple(self.backbone.trunk[0].conv.weight.shape)
                logger.info("conv_dna weight shape: %s", conv_dna_shape)
                if conv_dna_shape[1] == 170:
                    logger.info("conv_dna pretrained weights LOADED")
                else:
                    logger.info("conv_dna pretrained weights SKIPPED")
            else:
                logger.info("conv_dna skipped entirely, linear adapter used instead")

        except Exception as e:
            logger.warning("Weight loading had issues: %s", e)
            logger.info("Initializing from scratch for mismatched layers.")




    # Every parameter (weight/bias) in a pytorch model has a requires_grad flag
    # If requires_grad = True --> pytorch tracks gradients for it during backpropagation and the optimizer updates the prameter 
    # If requires_grad = False -->  parameter is frozen and doesnt change during training
    # Here we are freezing all non-head parameters for the num ber of freeze_backbone_epochs epochs
    # the pretrained layers here stay the same while only the  regression head learns to make ATAC predictions 
    # after freeze_backbone_epochs everything is unfrozen for finetuning
    # otehrwise parameters could get completely randomized again immediately and we loose pretraining information

    """
    def freeze_backbone(self) -> None:
        Freeze all backbone parameters except the head.
        for name, param in self.backbone.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False
        self._backbone_frozen = True
        logger.info("Backbone frozen (head is trainable).")
    """

   
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters except the head."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False
        self._backbone_frozen = True
        logger.info("Backbone frozen (head is trainable).")
        # adapter stays trainable so it can learn to produce
        # sensible input for the frozen pretrained res_tower
        if self.skip_conv_dna:
            # dont freeze linear adabpter
            for param in self.embedding_adapter.parameters():
                param.requires_grad = True
            

    

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False
        logger.info("Backbone unfrozen (all parameters trainable).")

    def on_train_epoch_start(self) -> None:
        """Handle backbone freezing/unfreezing schedule."""
        if self.backbone is None:  # here a little safe guard cuz there is no backbpone with only the embeding head
            return
        if self.freeze_backbone_epochs > 0:
            if self.current_epoch < self.freeze_backbone_epochs:
                if not self._backbone_frozen:
                    self.freeze_backbone()
            else:
                if self._backbone_frozen:
                    self.unfreeze_backbone()

#Input: one_hot tensor of shape (B, L, 4)  batches, seq_length, onehot 
#Output: tensor of shape (B, L, num_tracks) atac seq trensor
# forward here has no layers ebcause all is ahndled by baskerville backbone

    
   
    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            one_hot: DNA one-hot tensor of shape (B, L, 4).

        Returns:
            Predicted signal of shape (B, L, num_tracks).
        """

        if self.embedding_replaced_head:
            # one_hot is embedding her ein any case with the shape (B, L, 768)
            # FinalHead expects the channels first (B, C, L) because tahts what the last trunk outputs
            # it transposes internally, so we flip is first so the final head can flip it back inside
            x = one_hot.transpose(1, 2)      # (B, L, 768) -> (B, 768, L)
            return self.embed_head(x)        #FinalHead: -> (B,L,768) -> Linear and softplus -> (B, L, num_tracks) so (B, L, 1) cuz we hardcoded earlier

        elif self.skip_conv_dna:
            # project input from num_features into 96 with linear adapter
            # then put that directly into teh adjusted backbone which starts at res_tower
            x = self.embedding_adapter(one_hot)  # (B, L, 96)
            return self.backbone(x)
        else:
            return self.backbone(one_hot)

    # poisson nll loss
    def _compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask=None) -> torch.Tensor:
        """Poisson NLL loss (matching Shorkies poisson_mn).

        Args:
            pred: Predicted signal (B, L, num_tracks), output of softplus.
            target: Ground truth signal (B, L) or (B, L, num_tracks).
        """
        if target.ndim == 2:
            target = target.unsqueeze(-1)

        # Poisson NLL: target * log(pred + eps) - pred
        # We negate because we want to minimize
        eps = 1e-6
        loss = pred - target * torch.log(pred + eps)
        if mask is None:
            return loss.mean()
        mask = mask.unsqueeze(-1) # (B, L) -> (B, L, 1)
        # times mask will make the posiitons we dont care about 0 so they dont
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)
     


    #Pytorch lightning calls this for every batch during training.
    #return value is be the loss tensor
    #lightning automatically calls .backward() on it to compute gradients then calls  optimizer to update weights
    #self.log() sends values to all loggers so csv and the weights and bais thing 

    #essentially this is the training loop per batch 
    # step is one batch?
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        one_hot = batch["one_hot"]  # (B, L, 4)
        target = batch["target"]    # (B, L)
        mask = batch.get("loss_mask")    

        pred = self(one_hot)  # (B, L, num_tracks)
        loss = self._compute_loss(pred, target, mask)

        self.log("train/loss", loss, prog_bar=True)
        return loss

     # samebut no gradient computation --> no weight updating
     # logs  loss and pearson r 
    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        one_hot = batch["one_hot"]
        target = batch["target"]

        mask = batch.get("loss_mask")        # NEW: None for raw/tobias only for the chrombpnet one
        pred = self(one_hot)
        loss = self._compute_loss(pred, target, mask)   # NEW: pass mask
         # Also compute Pearson correlation
        # squeeze but DONT flatten yet / was .squeeze(-1).flatten()
        # because mask needs the (B, L) structure
        if pred.shape[-1] == 1:
            pred_flat = pred.squeeze(-1)     
        else:
            pred_flat = pred.flatten()  
        target_flat = target                

        # mask drops edge positions else behaves exactly like the original
        if mask is not None:
            # this flattens it --> makes 0 /1 to bool
            valid = mask.bool()
            pred_flat = pred_flat[valid]
            target_flat = target_flat[valid]
        else:
            pred_flat = pred_flat.flatten()
            target_flat = target_flat.flatten()

        corr = self._pearson_corr(pred_flat, target_flat)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/pearson_r", corr, prog_bar=True)
        



    @staticmethod
    def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pearson correlation coefficient."""
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        cov = (x_centered * y_centered).mean()
        std_x = x_centered.std()
        std_y = y_centered.std()
        return cov / (std_x * std_y + 1e-8)


    #Pytorch optimizer: (eg adam)
    # for each parameter it makes a running estimate of gradient magnitude and adjusts the step size --> prevents overfitiing..?
    # lr_lambda function defines how the learning rate scales at each step

    #during warmup_steps learning rate linearly ramps up from near 0 to the target lr. 
    #this prevents  making huge updates at start of training when gradients still kind of noisy 
    #after warmupe learning rate stays constant at lr

    #interval="step" --> updates every batch not epoch frequency=1 --> every single step
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(step: int) -> float:
            if step <= self.warmup_steps:
                return (step + 1) / (self.warmup_steps + 1)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
