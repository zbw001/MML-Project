import hydra
from omegaconf import DictConfig

from collections import namedtuple
from typing import Any, Dict
from lightning.pytorch import Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers.wandb import WandbLogger
import torch.multiprocessing as multiprocessing

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    print(cfg)

if __name__ == "__main__":
    main()