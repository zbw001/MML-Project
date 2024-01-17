import functools
import math
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
import torch
from functools import partial
import numpy as np
from typing import List
from PIL import Image
from typing import Union, Optional, Callable, Tuple

class AttnContext:
    def __init__(self):
        self.step_idx = None
    
    def set_step_idx(self, step_idx: Optional[int]):
        self.step_idx = step_idx

class CustomAttnProcessor(AttnProcessor2_0):
    def __init__(self, name: str, ctx: AttnContext): 
        super().__init__()
        self.name = name
        self.ctx = ctx

    def _get_mask(self, trg_len, center):
        w = int (round (math.sqrt(trg_len)))
        assert w * w == trg_len

        if center is None:
            return torch.zeros((trg_len,), dtype=self.ctx.dtype, device=self.ctx.device)

        mask = torch.zeros((w, w), dtype=torch.bool, device=self.ctx.device)

        x_diff = (torch.arange(w, dtype=self.ctx.dtype, device=self.ctx.device) / w - center[0]) ** 2
        y_diff = (torch.arange(w, dtype=self.ctx.dtype, device=self.ctx.device) / w - center[1]) ** 2 # TODO: 检查最终图像的位置s是否符合预期
        dist_mat = torch.sqrt(x_diff.unsqueeze(0) + y_diff.unsqueeze(1)) # TODO: check if this is correct
        
        mask[dist_mat > self.ctx.radius] = 1
        return mask.view(-1)

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        cross_attention = encoder_hidden_states is not None
        if not cross_attention:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask)
        else:
            assert hidden_states.shape[0] == 2 # unconditional and conditional
            object_keys = [
                key for key, value in encoder_hidden_states['object_pos'].items() if not isinstance(value, str)
            ]
            full_prompt = next(filter(lambda k: encoder_hidden_states['object_pos'][k] == 'global', encoder_hidden_states['object_pos'].keys()))
            extended_hidden_states = torch.cat(
                [hidden_states[: 1]] + [hidden_states[1:]] * (len(object_keys) + 1)
            )
            extended_encoder_hidden_states = torch.cat(
                [
                    encoder_hidden_states['text_embeddings']['<uncond>'],
                    encoder_hidden_states['text_embeddings'][full_prompt]
                ] +
                [encoder_hidden_states['text_embeddings'][key] for key in object_keys]
            )
            masks = torch.stack(
                [
                    self._get_mask(hidden_states.shape[-2], center=None),
                    self._get_mask(hidden_states.shape[-2], center=None),
                ] + 
                [
                    self._get_mask(hidden_states.shape[-2], center=encoder_hidden_states['object_pos'][key])
                    for key in object_keys
                ]
            ) # 0 : not masked, 1 : masked

            attn_out = super().__call__(attn, extended_hidden_states, extended_encoder_hidden_states)

            ret = torch.zeros((2, attn_out.shape[1], attn_out.shape[2]), dtype=self.ctx.dtype, device=self.ctx.device) # unconditional and conditional

            masks = masks.to(self.ctx.dtype)
            ret[0] = attn_out[0]
            ret[1] = attn_out[1] + ((attn_out[2:] - attn_out[1].unsqueeze(0)) * (self.ctx.params[self.ctx.step_idx].unsqueeze(-1) * (1 - masks[2:])).unsqueeze(-1)).sum(dim=0)
            return ret
            