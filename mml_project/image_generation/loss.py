from typing import Dict, Any
from cachetools import LRUCache
import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoTokenizer, CLIPModel
from torch.nn.functional import interpolate
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import rich

class CLIPLoss(nn.Module):
    
    def __init__(self, cfg: DictConfig, dtype: torch.dtype):
        super().__init__()
        loss_cfg = cfg.loss
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(loss_cfg.clip_model, torch_dtype=dtype)
        self.clip_model = CLIPModel.from_pretrained(loss_cfg.clip_model, torch_dtype=dtype)
        self.mean = torch.tensor(OPENAI_CLIP_MEAN, dtype=dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(OPENAI_CLIP_STD, dtype=dtype).view(1, 3, 1, 1)
        self.similarity_func = torch.nn.CosineSimilarity()
        self.cached_text_embeddings = LRUCache(maxsize=16)
        self.radius = cfg.radius
        self.local_loss_coef = loss_cfg.local_loss_coef

    def encode_images(self, images: torch.Tensor):
        assert images.dtype == self.dtype
        _, channels, height, width = images.shape
        assert channels == 3, "images must have 3 channels"
        normalized_images = (images - self.mean) / self.std
        if height != 224 or width != 224:
            normalized_images = interpolate(normalized_images, size=(224, 224), mode="bilinear", align_corners=False)
        return self.clip_model.get_image_features(pixel_values=normalized_images)
    
    def encode_text(self, text: str):
        if text in self.cached_text_embeddings:
            return self.cached_text_embeddings[text]
        inputs = self.tokenizer([text], padding=True, return_tensors="pt")
        self.cached_text_embeddings[text] = self.clip_model.get_text_features(**inputs)
        return self.cached_text_embeddings[text]
    
    def forward(self, images: torch.Tensor, encoder_conds: Dict[str, Any]):
        assert images.shape[0] == 1, "Only batch size 1 is supported"

        loss_global = 0.0
        loss_local = 0.0
        for k, v in encoder_conds["object_pos"].items():
            if isinstance(v, str):
                if v == "global":
                    text_embeddings = self.encode_text(k)
                    image_embeddings = self.encode_images(images)
                    loss_global += 1 - self.similarity_func(text_embeddings, image_embeddings)
                else :
                    continue
            else:
                text_embeddings = self.encode_text(k)
                height, width = images.shape[-2:]
                corner1 = torch.clamp(v - self.radius, min=0, max=1)
                corner2 = torch.clamp(v + self.radius, min=0, max=1)
                x_range = int (corner1[0] * width), int (corner2[0] * width)
                y_range = int (corner1[1] * height), int (corner2[1] * height)
                if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
                    rich.print(f"[red]Warning: empty sub-image for {k}[/red]")
                    continue
                
                image_embeddings = self.encode_images(
                    images[:, :, 
                           (height - 1 - y_range[1]): (height - 1 - y_range[0]), 
                           x_range[0]: x_range[1]
                          ]
                )
                loss_local += 1 - self.similarity_func(text_embeddings, image_embeddings)
        return loss_global + loss_local * self.local_loss_coef