import torch
from typing import Any, Dict, Union
from diffusers.pipelines import StableDiffusionPipeline
import torch.nn as nn
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from mml_project.image_generation.attn_processor import CustomAttnProcessor, AttnContext
from mml_project.image_generation.loss import CLIPLoss
from mml_project.layout_predictor.layout_predictor import LayoutPredictor
from torch.utils.checkpoint import checkpoint
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers import PNDMScheduler

class AttnOptimSampler:
    revision = "main"
    dtype = torch.float32
    width = 512
    height = 512

    def __init__(self, cfg: DictConfig, device: Union[torch.device, str] = "cuda"):
        self.device = device
        self.cfg = cfg
        
        self.unet = UNet2DConditionModel.from_pretrained(cfg.diffusion_model, subfolder="unet", torch_dtype=self.dtype).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.diffusion_model, subfolder="tokenizer", torch_dtype=self.dtype)
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.diffusion_model, subfolder="text_encoder", torch_dtype=self.dtype).to(self.device) # TODO: maybe fp32 is better for text encoder?
        self.scheduler = PNDMScheduler.from_pretrained(cfg.diffusion_model, subfolder="scheduler")

        self.vae = AutoencoderKL.from_pretrained(cfg.diffusion_model, subfolder="vae", torch_dtype=torch.float32).to(self.device) # vae is always fp32 for numerical stability

        self.clip_loss = CLIPLoss(self.cfg, dtype=torch.float32)
        self.clip_loss.to(self.device)
        
        self.layout_predictor = LayoutPredictor(
            cfg_path = self.cfg.layout_predictor.cfg_path,
            model_path = self.cfg.layout_predictor.model_path,
            device = self.device
        )

        self.cfg = cfg
        self.guidance_scale = cfg.guidance_scale
        self.num_inference_steps = cfg.num_inference_steps
        self.radius = cfg.radius

        self.ctx = AttnContext()
        self.ctx.dtype = self.dtype
        self.ctx.device = self.device
        self.ctx.radius = self.radius

        self._setup_attention_control()
        self._freeze_modules() # The parameters to be optimized are not in the modules

    def _freeze_modules(self):
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.clip_loss.parameters():
            param.requires_grad = False

    def _gen_latents(self, batch_size: int = 1, seed: int = 42):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, self.height // 8, self.width // 8),
            dtype=self.dtype,
            generator=generator,
            device=self.device
        )
        return latents
    
    def _setup_attention_control(self):
        self.attn_processors = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_att_count += 1
            self.attn_processors[name] = CustomAttnProcessor(name=name, ctx=self.ctx)

        self.unet.set_attn_processor(self.attn_processors)
    
    def _denoise_latents(self, latents: torch.Tensor, encoder_conds: Dict[str, Any]):
        batch_size = latents.shape[0]
        assert batch_size == 1

        self.scheduler.set_timesteps(self.num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma

        block_size = self.cfg.optimization.gradient_checkpointing_block_size

        def denoise_steps(latents, idxs, timesteps):
            for i, t in zip(idxs, timesteps):
                unet_input = torch.cat([latents] * 2) # conditional and unconditional
                unet_input = self.scheduler.scale_model_input(unet_input, timestep=t)

                self.ctx.set_step_idx(i)
                unet_output = self.unet(unet_input, t, encoder_hidden_states=encoder_conds).sample
                self.ctx.set_step_idx(None)

                unconditioned_noise, conditioned_noise = unet_output.chunk(2)
                noise_pred = unconditioned_noise + self.guidance_scale * (conditioned_noise - unconditioned_noise)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            return latents

        pbar = tqdm(total=len(self.scheduler.timesteps), desc="Denoising")

        for start_idx in range(0, len(self.scheduler.timesteps), block_size):
            end_idx = min(start_idx + block_size, len(self.scheduler.timesteps))
            timesteps = self.scheduler.timesteps[start_idx : end_idx]
            latents = checkpoint(denoise_steps, latents, range(start_idx, end_idx), timesteps, use_reentrant=False)
            pbar.update(len(timesteps))

        pbar.close()

        return latents

    def _decode_latents(self, latents: torch.Tensor):
        latents = latents.to(torch.float32)
        batch_size = latents.shape[0]
        assert batch_size == 1
        latents = 1 / 0.18215 * latents
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        return images
    
    def _prepare_conditions(self, prompt: str):
        object_pos = self.layout_predictor.inference_sentence(prompt)
        num_objects = len(object_pos)

        prompts = [prompt] + list(object_pos.keys())
        text_input = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        input_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        text_embeddings = dict(zip(prompts, input_embeddings.chunk(len(prompts))))
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0] 
        text_embeddings["<uncond>"] = uncond_embeddings
        
        object_pos[prompt] = "global"
        object_pos["<uncond>"] = "uncond"

        return {
            "text_embeddings": text_embeddings,
            "object_pos": object_pos,
            "num_objects": num_objects
        }
    
    def sample(self, prompt: str, seed: int = 42):
        encoder_conds = self._prepare_conditions(prompt=prompt)

        latents = self._gen_latents(batch_size=1, seed=seed)
        optimizer_cls = None
        if self.cfg.optimization.optimizer_type == "Adam":
            optimizer_cls = torch.optim.Adam
        elif self.cfg.optimization.optimizer_type == "AdamW":
            optimizer_cls = torch.optim.AdamW
        else:
            raise ValueError(f'Unsupported optimizer type: {self.cfg.optimization.optimizer_type}')

        with torch.set_grad_enabled(True), torch.cuda.amp.autocast(): # be careful not to include the text encoder in the gradient computation
            num_objects = encoder_conds["num_objects"]
            self.ctx.params = nn.Parameter(
                torch.full(
                    (self.num_inference_steps + 1, num_objects),
                    fill_value=self.cfg.optimization.weight_initialize_coef / num_objects if num_objects else 0.0,
                    dtype=torch.float32,
                    device=self.device,
                )
            )

            optimizer = optimizer_cls([self.ctx.params], **self.cfg.optimization.optimizer_kwargs)
            for _ in tqdm(range(self.cfg.optimization.num_steps), desc="Optimizing"):
                denoised_latents = self._denoise_latents(
                    latents = latents,
                    encoder_conds = encoder_conds,
                )
                images = self._decode_latents(denoised_latents)
                
                optimizer.zero_grad()
                loss = self.clip_loss(images, encoder_conds)
                loss.backward()
                optimizer.step()

                torch.cuda.empty_cache()
            self.ctx.parmas = None

        return images
    
    def to_pil_image(self, image_tensor: torch.Tensor):
        image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).round().astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        return pil_image

if __name__ == "__main__":
    sampler = AttnOptimSampler(
        cfg = OmegaConf.load("configs/default.yaml"),
        device = "cuda"
    )
    images = sampler.sample(prompt="an apple tree")
    pil_image = sampler.to_pil_image(images[0])
    file_name = f"test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    pil_image.save("outputs/" + file_name)