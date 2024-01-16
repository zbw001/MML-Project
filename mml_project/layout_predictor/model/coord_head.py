import torch.nn as nn
from mml_project.layout_predictor.utils.gmm import GMM2D
from omegaconf import DictConfig

class GMMHead(nn.Module):
    GMM_PARAMS = 6
    def __init__(self, 
                 hidden_size: int, 
                 num_components: int, 
                 temperature: float = None, 
                 greedy: bool = False):
        super().__init__()
        self.num_components = num_components
        self.temperature = temperature
        self.greedy = greedy

        self.fc = nn.Linear(hidden_size, num_components * self.GMM_PARAMS)

    def forward(self, x) -> GMM2D:
        params = self.fc(x)
        return GMM2D(num_components=self.num_components, 
                     params=params, 
                     temperature=self.temperature, 
                     greedy=self.greedy)

class Coord2DHead(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.hidden_size = model_cfg.hidden_size

        self.output_Layer = nn.Linear(self.hidden_size, self.hidden_size) # for compatibility with the original checkpoint
        self.box_predictor = GMMHead(self.hidden_size,
                                     num_components=model_cfg.gmm_head.num_components,
                                     temperature=model_cfg.gmm_head.temperature,
                                     greedy=model_cfg.gmm_head.greedy)

    def forward(self, x) -> GMM2D:
        x = self.output_Layer(x)
        return self.box_predictor(x)