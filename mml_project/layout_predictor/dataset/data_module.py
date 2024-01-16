import lightning.pytorch as pl
from omegaconf import DictConfig
import torch
from mml_project.layout_predictor.dataset.coco_dataset import COCORelDataset
from torch.utils.data import DataLoader
from pathlib import Path

def _collate(batch):
    return {'bpe_toks': [batch[i][0] for i in range(len(batch))], 'object_index': [batch[i][1] for i in range(len(batch))], 'caption': [batch[i][3] for i in range(len(batch))], \
        'relation': [batch[i][4] for i in range(len(batch))]}

class RelDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_cfg: DictConfig):
        super().__init__()
        self.data_dir = Path(data_cfg.data_dir)
        if data_cfg.get('batch_size', None) is not None:
            raise ValueError("batch_size should be specified in the training config, not the data config")
        
        self.batch_size = batch_size

        self.instances_data_path = self.data_dir / "instances_train2017.json"
        self.stuff_data_path = self.data_dir / "stuff_train2017.json"

        self.test_instances_data_path = self.data_dir / "instances_val2017.json"
        self.test_stuff_data_path = self.data_dir / "stuff_val2017.json"

        self.val_split = data_cfg.val_split
        self.num_workers = data_cfg.num_workers
        self.shuffle = data_cfg.shuffle
        
        assert self.instances_data_path.exists(), f"{self.instances_data_path} does not exist."
        assert self.stuff_data_path.exists(), f"{self.stuff_data_path} does not exist."
        assert self.test_instances_data_path.exists(), f"{self.test_instances_data_path} does not exist."
        assert self.test_stuff_data_path.exists(), f"{self.test_stuff_data_path} does not exist."

    def setup(self, stage: str):
        self._dataset = COCORelDataset(instances_json=str(self.instances_data_path), stuff_json=str(self.stuff_data_path))
        dataset_size = len(self._dataset)
        
        val_size = int(dataset_size * self.val_split)
        train_size = dataset_size - val_size

        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self._dataset, [train_size, val_size], generator=generator)

        self.test_dataset = COCORelDataset(instances_json=str(self.test_instances_data_path), stuff_json=str(self.test_stuff_data_path))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=self.shuffle, collate_fn=_collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, collate_fn=_collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, collate_fn=_collate)