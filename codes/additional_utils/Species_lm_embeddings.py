import torch
from torch import nn
import lightning as L
import numpy as np
import polars as pl
from functools import partial
import yaml
import omegaconf
from abc import ABC, abstractmethod
# from transformers import AutoModelForMaskedLM

from transformers import BertConfig
from flash_attn.models.bert import BertForPreTraining


# this is an abstract class like a template ti say what deos what?
class BaseEmbedderWrapper(ABC): 

    @abstractmethod
    def embed(self, X): 
        """Embeds input tokenized DNA sequences

        Parameters
        ----------
        X: torch.Tensor
            Tokenized input sequences, shape (B, L), B=batch size, L=sequence length

        Returns
        -------
        tuple[torch.Tensor]
            (x_input, Optional[x_target])
            Embedded sequences of shape (B, L, D), D=embedding dimension
    
        """
        pass

    @abstractmethod
    def tokenize_func(self): 
        """
        Returns a partial function used for tokenizing the input sequences.
        This function will be applied during data-loading.
        """

        def _tok(seqs, species, *args, **kwargs): 
            return 

        return partial(_tok)


# this is the actual embedder 

#for calling i need a model path
# i saw in git
#  from transformers import Trainer
# from transformers import DataCollatorForLanguageModeling
# from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig  
# tokenizer = AutoTokenizer.from_pretrained("gagneurlab/SpeciesLM", revision = "downstream_species_lm")
# model = AutoModelForMaskedLM.from_pretrained("gagneurlab/SpeciesLM", revision = "downstream_species_lm")

class SpeciesLMV1_embedder(BaseEmbedderWrapper):
    #here for initializing
    def __init__(
            self, 
            model_path: str, 
            kmer_size: int,
            layer = 11, 
            method: str = "single",
            pool_func = "average",
            revision="main",
            device="cuda",
            torch_compile=False,
            whitener=None,
            **model_kwargs
        ):


        self.model_path = model_path
        self.kmer_size = kmer_size
        self.layer = layer
        if isinstance(self.layer, omegaconf.listconfig.ListConfig):
            self.layer = list(self.layer)
        assert isinstance(self.layer, int) or isinstance(self.layer, list) or (self.layer is None), f"`layer` must be int, list or None, but received {type(self.layer)}"

        self.device = device
        self.revision = revision
        self.torch_compile = torch_compile

        self.right_special_tokens = 1
        self.left_special_tokens = 2

        self.prepare_model()

        # store method and pooling
        self.method = method
        self.pool_func = pool_func
        # activations container (populated per-forward)
        self.activations = {}

        # validate and set blocks depending on method
        if self.method == "single":
            assert isinstance(self.layer, int), "For 'single' method, layer must be an integer."
            self.block = self.model.bert.encoder.layers[self.layer]
        elif self.method == "pool_average":
            assert isinstance(self.layer, list), "For 'pool_average' method, layer must be a list of integers."
            assert len(layer) > 0, "Layer list for pool_average must be non-empty."
            self.blocks = [self.model.bert.encoder.layers[l] for l in layer]
        elif self.method == "tuple":
            assert isinstance(self.layer, list), "For 'tuple' method, layer must be a list."
            assert isinstance(self.layer[0], int), "For 'tuple' method, first element must be an integer (target layer)."
            if isinstance(self.layer[1], omegaconf.listconfig.ListConfig):
                self.layer[1] = list(self.layer[1])
            assert isinstance(self.layer[1], list), "For 'tuple' method, second element must be a list of integers (input layers)."
            if len(self.layer[1]) > 1:
                assert pool_func is not None, "If multiple embedding layers are provided in 'tuple' method, pool_func must be specified."
            self.input_block = self.model.bert.encoder.layers[self.layer[0]]
            self.embedding_blocks = [self.model.bert.encoder.layers[l] for l in self.layer[1]]
        else:
            raise ValueError(f"Unknown method {method}. Supported methods are 'single', 'pool_average', and 'tuple'.")
        
        # self.block = self.model.bert.encoder.layers[self.layer]
        pass

    def prepare_model(self):
        import os
        config = BertConfig.from_pretrained(self.model_path)
        self.model = BertForPreTraining(config)
        ckpt_path = os.path.join(self.model_path, "pytorch_model.bin")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        device = torch.device(self.device)
        self.model = self.model.to(device)
        self.model.eval()
    
    def _make_hook(self, key):
        """Return a forward hook that stores detached activation under self.activations[key]."""
        def _hook(module, inp, out):
            # detach to avoid keeping computational graph
            self.activations[key] = out.detach()
        return _hook

    def _pool_tensors(self, tensors, pool_func) -> torch.Tensor:
        """Pool a list of tensors into one tensor using pool_func ('average' or 'max')."""
        if len(tensors) == 0:
            raise RuntimeError("No tensors to pool.")
        if len(tensors) == 1:
            return tensors[0] # (B, L, D)
        stacked = torch.stack(tensors, dim=0)  # shape (num_layers, ...) # (k, B, L, D)
        if pool_func is None or pool_func == "average":
            return stacked.mean(dim=0)
        else:
            raise ValueError(f"Unknown pool_func {pool_func}. Supported: 'average', 'max', or None.")

    def embed(self, x):
        """Run model forward and return activations according to selected method.

        Returns
        -------
        tuple[torch.Tensor, Optional[torch.Tensor]]
            (input_embeddings, Optional[target_embeddings])
        
        """
        # reset activations and handles
        self.activations = {}
        handles = []

        try:
            if self.method == "single":
                handle = self.block.register_forward_hook(self._make_hook("single"))
                handles.append(handle)
            elif self.method == "pool_average":
                # register a hook per block, keys 0..n-1
                for i, blk in enumerate(self.blocks):
                    handles.append(blk.register_forward_hook(self._make_hook(i)))
            elif self.method == "tuple":
                # input block under key "input", embeddings keys 0..n-1
                handles.append(self.input_block.register_forward_hook(self._make_hook("target")))
                for i, blk in enumerate(self.embedding_blocks):
                    handles.append(blk.register_forward_hook(self._make_hook(i)))
            else:
                raise RuntimeError(f"Unsupported method {self.method}")
            
            
            
            with torch.autocast(device_type=self.device):
                _ = self.model(x)
            

        finally:
            # ensure hooks are removed regardless of forward success/failure
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        # collect and return activations depending on method
        if self.method == "single":
            if "single" not in self.activations:
                raise RuntimeError(f"Hook did not fire for layer {self.layer}")
            return (self.activations["single"][:, self.left_special_tokens:-self.right_special_tokens, :], )

        elif self.method == "pool_average":
            # map activations in order of blocks (0..)
            tensors = []
            for i in range(len(self.blocks)):
                if i not in self.activations:
                    raise RuntimeError(f"Hook did not fire for one of the pool layers (index {i}).")
                tensors.append(self.activations[i])
            
            pooled = self._pool_tensors(tensors, self.pool_func)
            return (pooled[:, self.left_special_tokens:-self.right_special_tokens, :], )

        elif self.method == "tuple":
            if "target" not in self.activations:
                raise RuntimeError(f"Hook did not fire for target layer {self.layer[0]}")
            target_emb = self.activations["target"][:, self.left_special_tokens:-self.right_special_tokens, :]
            # collect embedding activations
            emb_tensors = []
            for i in range(len(self.embedding_blocks)):
                if i not in self.activations:
                    raise RuntimeError(f"Hook did not fire for embedding layer index {i} (layer id {self.layer[1][i]}).")
                emb_tensors.append(self.activations[i][:, self.left_special_tokens:-self.right_special_tokens, :])

            # if multiple embeddings and pool_func set, return pooled embedding
            if len(emb_tensors) > 1 and self.pool_func is not None:
                input_emb = self._pool_tensors(emb_tensors, self.pool_func)
                return (input_emb, target_emb)
            elif len(emb_tensors) == 1:
                input_emb = emb_tensors[0]
                return (input_emb, target_emb)
            else: 
                raise RuntimeError("Multiple embedding tensors returned but pool_func is None.")


    def tokenize_func(self): 
        from transformers import AutoTokenizer
        kmers_stride1 = lambda seq, k: [seq[i : i + k] for i in range(0, len(seq) - k + 1)]
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, revision="main")

        def _tok(seq, species, tokenizer, kmer_size):
        
            if species == "_saccharomyces_cerevisiae":
                species = "kazachstania_africana_cbs_2517_gca_000304475"

            assert species in tokenizer.vocab.keys(), f"Species {species} is not in tokenizer vocab."
            if kmer_size > 1:
                kmers = kmers_stride1(seq, k=kmer_size)
                seq = species + " " + " " .join(kmers)
                seq = torch.tensor(seq)
                tokenized = tokenizer(seq)
                species_ids = tokenizer(species)["input_ids"]
                tokenized["species_ids"] = species_ids[1]
            else: 
                seq = tokenizer(species + " ".join(list(seq)))["input_ids"]
                seq = torch.tensor(seq)
                tokenized = {"input_ids": seq, "species_ids": seq[1]}
            return tokenized

        return partial(_tok, tokenizer=tokenizer, kmer_size=self.kmer_size)
    

class SpeciesLMV2_embedder(BaseEmbedderWrapper):
    
    def __init__(self):
        pass

class GenomeLM_embedder(BaseEmbedderWrapper): 
    
    def __init__(self): 
        pass