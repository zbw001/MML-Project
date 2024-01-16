import torch
import torch.nn as nn
from mml_project.layout_predictor.utils.gmm import GMM2D

class RelHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gmm1: GMM2D, gmm2: GMM2D, relation_name: str):
        gmm1 = gmm1.copy()
        gmm2 = gmm2.copy()
        gmm1.u_x = torch.clamp(gmm1.u_x, min=0.0, max=1.0)
        gmm1.u_y = torch.clamp(gmm1.u_y, min=0.0, max=1.0)
        gmm2.u_x = torch.clamp(gmm2.u_x, min=0.0, max=1.0)
        gmm2.u_y = torch.clamp(gmm2.u_y, min=0.0, max=1.0)

        if relation_name == "above":
            diff = torch.max(gmm1.u_y, dim=-1)[0] - torch.min(gmm2.u_y, dim=-1)[0]
        elif relation_name == "below":
            diff = torch.max(gmm2.u_y, dim=-1)[0] - torch.min(gmm1.u_y, dim=-1)[0]
        elif relation_name == "left of":
            diff = torch.max(gmm1.u_x, dim=-1)[0] - torch.min(gmm2.u_x, dim=-1)[0]
        elif relation_name == "right of":
            diff = torch.max(gmm2.u_x, dim=-1)[0] - torch.min(gmm1.u_x, dim=-1)[0]
        else:
            raise Exception(f"Unknown relation: {relation_name}")
        return torch.sum(torch.max(diff, torch.tensor([-0.2]).to("cuda")))