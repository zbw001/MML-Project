import torch
import torch.nn as nn
from omegaconf import DictConfig
from mml_project.layout_predictor.model.coord_head import Coord2DHead
from mml_project.layout_predictor.model.roberta_modified import RobertaEncoder
from copy import deepcopy

class Text2Coord(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.encoder = torch.hub.load('pytorch/fairseq', 'roberta.base')
        state_dict = deepcopy(self.encoder.model.encoder.state_dict())
        self.encoder.model.encoder = RobertaEncoder()
        self.encoder.model.encoder.load_state_dict(state_dict, strict=False)
        self.coord_head = Coord2DHead(model_cfg)

    def forward(self, bpe_toks, object_pos):
        features, _ = self.encoder.model.encoder(bpe_toks, object_pos=object_pos)
        return self.coord_head(features)