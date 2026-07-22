import abc
import re
import time
from pathlib import Path
from typing import Any, Literal

try:
    from lightning import LightningModule
except ImportError:
    from pytorch_lightning import LightningModule
import wandb
try:
    from lightning.pytorch.utilities import grad_norm
except ImportError:
    from pytorch_lightning.utilities import grad_norm
from omegaconf import DictConfig, OmegaConf
from torch import nn
from wandb.apis.public.runs import Run

from genome_lm.train.metrics.throughput import ThroughputMetric
from genome_lm.util.checkpoint import consolidate_checkpoint, get_latest_checkpoint
from genome_lm.util.reflection import import_class


class GenomeLM(LightningModule, abc.ABC):
    def __init__(self) -> None:
        super().__init__()
        self.last_batch_time: float | None = None
        self.train_throughput_metric: ThroughputMetric | None = None
        self.test_throughput_metric: ThroughputMetric | None = None

    def configure_model(self) -> None:
        if self.train_throughput_metric is not None:
            return  # Already configured
        self.train_throughput_metric = ThroughputMetric()
        self.test_throughput_metric = ThroughputMetric()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["model_cls"] = f"{self.__class__.__module__}.{self.__class__.__name__}"

    @staticmethod
    def load_from_checkpoint_dynamic(
        checkpoint_path: str | Path, model_kwargs: dict[str, Any] | None = None
    ) -> "GenomeLM":
        checkpoint_path = consolidate_checkpoint(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="meta")
        model_cls = import_class(checkpoint["model_cls"])
        if not issubclass(model_cls, GenomeLM):
            raise TypeError(f"{model_cls} is not a subclass of GenomeLM.")
        model = model_cls.load_from_checkpoint(checkpoint_path, **(model_kwargs or {}))
        return model

    @staticmethod
    def load_from_wandb(
        run_id: str, checkpoint_regex: str | None = None, model_kwargs: dict[str, Any] | None = None
    ) -> tuple["GenomeLM", DictConfig]:
        run: Run = wandb.Api().run(run_id)
        assert run.metadata is not None
        root = Path(
            re.sub(
                r"(/data/nasif12/home_if12/.+/genome-lm/data|/p/project1/hai_1134/.+/genome-lm/data)",
                "/s/project/genome-lm",
                run.metadata["root"],
            )
        )
        checkpoint = get_latest_checkpoint(root, regex=checkpoint_regex)
        print(f"Loading checkpoint from {checkpoint}")
        model = GenomeLM.load_from_checkpoint_dynamic(checkpoint, model_kwargs=model_kwargs)
        config = OmegaConf.create(run.config["hydra"])
        assert isinstance(config, DictConfig)
        return model, config

    def compute_throughput(
        self, batch: dict[str, Any], mode: Literal["train", "test"] = "train"
    ) -> None:
        if mode == "train":
            metric = self.train_throughput_metric
            metric_name = "throughput"
        else:
            metric = self.test_throughput_metric
            metric_name = "test/throughput"
        if metric is None:
            raise RuntimeError("compute_throughput() called before configure_model()")
        with torch.no_grad():
            # Manual logging because automatic logging does not seem to reduce results correctly
            current_time = time.time()
            if self.last_batch_time is not None:
                elapsed = current_time - self.last_batch_time
                num_tokens = sum(end - start for start, end in zip(batch["start"], batch["end"]))
                metric.update(torch.as_tensor(num_tokens), torch.as_tensor(elapsed))
                if self.trainer._logger_connector.should_update_logs:
                    self.log(metric_name, metric.compute())
                    metric.reset()
            self.last_batch_time = current_time

    def on_train_end(self):
        self.last_batch_time = None

    def on_test_end(self):
        self.last_batch_time = None
