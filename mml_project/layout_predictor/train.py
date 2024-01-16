import hydra
from omegaconf import DictConfig
from lightning.pytorch import Trainer, LightningModule
from mml_project.layout_predictor.dataset.data_module import RelDataModule
from mml_project.layout_predictor.model.text2coord import Text2Coord
from mml_project.layout_predictor.utils.schedulers import build_optimizer_and_scheduler
from mml_project.layout_predictor.utils.loss import RelHingeLoss
from lightning.pytorch.loggers import WandbLogger
import torch
from mml_project.layout_predictor.utils.gmm import GMM2D
import rich
from datetime import datetime

class Text2CoordModule(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.train_cfg = cfg.train
        self.model_cfg = cfg.model
        self.model = Text2Coord(cfg.model)
        self.relative_loss_coef = cfg.train.relative_loss_coef
        self.rel_hinge_loss = RelHingeLoss()

        self.automatic_optimization = False

        if cfg.train.pretrained_weights is not None:
            rich.print(f"Loaded pretrained checkpoint: {cfg.train.pretrained_weights}")
            state_dict = torch.load(cfg.train.pretrained_weights)['state_dict']
            
            results = self.model.load_state_dict(state_dict, strict=False)
            if len(results.missing_keys) > 0 or len(results.unexpected_keys) > 0:
                rich.print(f"[bold yellow]Warning: pretrained checkpoint {cfg.train.pretrained_weights} has missing or unexpected keys[/bold yellow]")
                rich.print(f"Missing keys: {results.missing_keys}")
                rich.print(f"Unexpected keys: {results.unexpected_keys}")
        else:
            rich.print("No pretrained checkpoint provided, training from scratch")

    def loss(self, batch):
        bpe_toks = batch['bpe_toks']
        batch_object_index = batch['object_index']
        batch_caption = batch['caption']
        batch_relations = batch['relation']

        batch_size = len(bpe_toks)
        bpe_toks_tensor = torch.cat(bpe_toks).int()

        object_tensors = []
        for object_index in batch_object_index:
            object_tensor = torch.zeros(bpe_toks[0].shape[1], dtype=torch.bool, device=self.device)
            for index in object_index:
                object_tensor[index] = True
            object_tensors.append(object_tensor)
        object_tensor = torch.stack(object_tensors)

        gmm: GMM2D = self.model(bpe_toks_tensor, object_pos=object_tensor)

        gmm_loss, relative_loss = 0.0, 0.0
        

        for i in range(batch_size):
            relations = batch_relations[i]
            if len(relations) == 0:
                continue
            if len(relations[0]) == 3:
                # Data for relative loss
                for relation in relations:
                    obj_pos1, obj_pos2, relation_name = relation
                    relative_loss += self.rel_hinge_loss(
                        gmm[i][obj_pos1],
                        gmm[i][obj_pos2],
                        relation_name,
                    )
            elif len(relations[0]) == 4:
                for obj_id, obj_pos in enumerate(relations):
                    gt_xy = torch.tensor(obj_pos[:2], dtype=torch.float32, device=self.device) 
                    # It looks strange. The data loader should be refactored.
                    gmm_loss += - gmm[i][batch_object_index[i][obj_id]].log_prob(gt_xy).mean()
            else:
                raise ValueError("Invalid data format for relations")

        total_loss = gmm_loss + self.relative_loss_coef * relative_loss
        return total_loss, gmm_loss, relative_loss

    def training_step(self, batch, batch_idx):
        total_loss, gmm_loss, relative_loss = self.loss(batch)
        
        encoder_optimizer, coord_head_optimizer = self.optimizers()
        encoder_optimizer.zero_grad()
        coord_head_optimizer.zero_grad()

        self.manual_backward(total_loss)

        encoder_optimizer.step()
        coord_head_optimizer.step()

        encoder_scheduler, coord_head_scheduler = self.lr_schedulers()
        encoder_scheduler.step()
        coord_head_scheduler.step()

        self.log('train/total_loss', total_loss)
        self.log('train/gmm_loss', gmm_loss)
        self.log('train/relative_loss', relative_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            total_loss, gmm_loss, relative_loss = self.loss(batch)
        self.log('val/total_loss', total_loss)
        self.log('val/gmm_loss', gmm_loss)
        self.log('val/relative_loss', relative_loss)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            total_loss, gmm_loss, relative_loss = self.loss(batch)
        self.log('test/total_loss', total_loss)
        self.log('test/gmm_loss', gmm_loss)
        self.log('test/relative_loss', relative_loss)

    def configure_optimizers(self):
        encoder_optimizer, encoder_scheduler = build_optimizer_and_scheduler(self.train_cfg.optimize.encoder, self.model_cfg.hidden_size, self.model.encoder.parameters())
        coord_head_optimizer, coord_head_scheduler = build_optimizer_and_scheduler(self.train_cfg.optimize.coord_head, self.model_cfg.hidden_size, self.model.coord_head.parameters())

        return [encoder_optimizer, coord_head_optimizer], [encoder_scheduler, coord_head_scheduler]

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    model_module = Text2CoordModule(cfg)
    data_module = RelDataModule(batch_size=cfg.train.batch_size, data_cfg=cfg.data)
    wandb_logger = WandbLogger(
        name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        **cfg.train.wandb
    )
    trainer = Trainer(logger=wandb_logger, **cfg.train.lightning_trainer)
    
    trainer.fit(model_module, datamodule=data_module)

if __name__ == "__main__":
    main()