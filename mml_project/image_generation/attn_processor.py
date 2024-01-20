from mml_project.image_generation.context import AttnOptimContext
import math
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
import torch
from diffusers.utils import USE_PEFT_BACKEND
    
class CustomAttnProcessor(AttnProcessor2_0):
    def __init__(self, name: str, ctx: AttnOptimContext): 
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
        y_diff = (torch.arange(w, dtype=self.ctx.dtype, device=self.ctx.device) / w - center[1]) ** 2 # TODO: check the generated image to see if this is correct
        dist_mat = torch.sqrt(x_diff.unsqueeze(0) + y_diff.unsqueeze(1)) # TODO: check if this is correct
        
        mask[dist_mat > self.ctx.radius] = 1
        return mask.view(-1)
    
    def _process(self, 
                 attn: Attention, 
                 hidden_states, 
                 encoder_hidden_states=None, 
                 attention_mask=None,
                 temb=None,
                 scale=1.0
    ):
        residual = hidden_states
        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states, attention_probs

    def __call__(self, 
                 attn: Attention, 
                 hidden_states, 
                 encoder_hidden_states=None, 
                 attention_mask=None,
                 temb=None,
                 scale=1.0
    ):
        cross_attention = encoder_hidden_states is not None
        if not cross_attention:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask)
        else:
            assert hidden_states.shape[0] == 2 # unconditional and conditional
            object_keys = encoder_hidden_states['object_keys']
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

            attn_out, attn_map = self._process(attn, extended_hidden_states, extended_encoder_hidden_states, attention_mask, temb, scale)

            masks = torch.stack(
                [
                    self._get_mask(hidden_states.shape[-2], center=encoder_hidden_states['object_pos'][key])
                    for key in object_keys
                ]
            ).to(self.ctx.dtype) # 0 : not masked, 1 : masked

            # ======= compute losas and store attention map =======
            avg_attn_map = attn_map.reshape((extended_hidden_states.shape[0], -1) + attn_map.shape[-2:]).mean(dim=1)
            sot_avg_attn_map = avg_attn_map[..., 0]

            common_area = ((1 - masks) * ((1 - sot_avg_attn_map))[2:]).sum(dim=-1)
            if self.ctx.step_idx >= 40 and self.name.startswith("down_blocks.2") or self.name.startswith("up_blocks.1"):
                self.ctx.acc_loss += - (common_area.abs() ** 0.5).sum()

            if self.ctx.debug and self.ctx.step_idx % self.ctx.save_interval == 0:
                info_key = (self.ctx.epoch, self.ctx.step_idx, self.name)
                self.ctx.info["attention_maps"][info_key] = avg_attn_map.detach()
            # ====================================================
                
            ret = torch.zeros((2, attn_out.shape[1], attn_out.shape[2]), dtype=self.ctx.dtype, device=self.ctx.device) # unconditional and conditional

            ret[0] = attn_out[0]
            ret[1] = attn_out[1] + ((attn_out[2:] - attn_out[1].unsqueeze(0)) * (self.ctx.params[self.ctx.step_idx].unsqueeze(-1) * (1 - masks)).unsqueeze(-1)).sum(dim = 0)
            # ret[1] = ((attn_out[2:]) * (encoder_hidden_states["params"].unsqueeze(-1) * (1 - masks)).unsqueeze(-1)).sum(dim = 0)
            return ret
            