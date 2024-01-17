import lightning.pytorch as pl
from omegaconf import DictConfig
import torch
from mml_project.layout_predictor.dataset.coco_dataset import MixedDataset, load_datasets
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

        self.train_instances_data_path = self.data_dir / "instances_train2017.json"
        self.train_stuff_data_path = self.data_dir / "stuff_train2017.json"
        # self.val_instances_data_path = self.data_dir / "instances_val2017.json"
        # self.val_stuff_data_path = self.data_dir / "stuff_val2017.json"

        self.val_split_ratio = data_cfg.val_split_ratio

        self.num_workers = data_cfg.num_workers
        self.shuffle = data_cfg.shuffle

        paths_to_check = [self.train_instances_data_path, self.train_stuff_data_path]
        for path in paths_to_check:
            if not path.exists():
                raise ValueError(f"{path} does not exist")

    def setup(self, stage: str):
        coco_dataset, gpt_dataset = load_datasets(instances_json=str(self.train_instances_data_path), stuff_json=str(self.train_stuff_data_path))
        coco_dataset_train, coco_dataset_val = torch.utils.data.random_split(coco_dataset, [len(coco_dataset) - int(len(coco_dataset) * self.val_split_ratio), int(len(coco_dataset) * self.val_split_ratio)])
        gpt_dataset_train, gpt_dataset_val = torch.utils.data.random_split(gpt_dataset, [len(gpt_dataset) - int(len(gpt_dataset) * self.val_split_ratio), int(len(gpt_dataset) * self.val_split_ratio)])

        self.train_dataset = MixedDataset(coco_dataset_train, gpt_dataset_train)
        self.val_dataset = MixedDataset(coco_dataset_val, gpt_dataset_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=self.shuffle, collate_fn=_collate)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, collate_fn=_collate)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, collate_fn=_collate)