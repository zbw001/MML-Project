import torch
from typing import Any, Dict, Union
import torch.nn as nn
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from mml_project.image_generation.context import AttnOptimContext
from mml_project.image_generation.attn_processor import CustomAttnProcessor
from mml_project.image_generation.loss import CLIPLoss
from mml_project.layout_predictor.layout_predictor import LayoutPredictor
from torch.utils.checkpoint import checkpoint
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers import PNDMScheduler
import cv2
import json
import rich

class AttnOptimSampler:
    revision = "main"
    dtype = torch.float32
    width = 512
    height = 512

    def __init__(self, cfg: DictConfig, device: Union[torch.device, str] = "cuda", debug: bool = False):
        self.device = device
        self.cfg = cfg
        self.debug = debug
        
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

        self.reg_coef = cfg.optimization.reg_coef

        self.save_interval = cfg.save_interval

        self.ctx = AttnOptimContext(
            dtype=self.dtype,
            device=self.device,
            radius=self.radius
        )

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
        for name in self.unet.attn_processors.keys():
            self.attn_processors[name] = CustomAttnProcessor(name=name, ctx=self.ctx)

        self.unet.set_attn_processor(self.attn_processors)
    
    def _denoise_latents(self, latents: torch.Tensor, encoder_conds: Dict[str, Any]):
        batch_size = latents.shape[0]
        assert batch_size == 1

        self.scheduler.set_timesteps(self.num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma

        block_size = self.cfg.optimization.gradient_checkpointing_block_size
        encoder_conds = encoder_conds.copy()

        def denoise_steps(latents, idxs, timesteps):
            for i, t in zip(idxs, timesteps):
                unet_input = torch.cat([latents] * 2) # conditional and unconditional
                unet_input = self.scheduler.scale_model_input(unet_input, timestep=t)

                encoder_conds['params'] = self.ctx.params[i]
                unet_output = self.unet(unet_input, t, encoder_hidden_states=encoder_conds).sample

                unconditioned_noise, conditioned_noise = unet_output.chunk(2)
                noise_pred = unconditioned_noise + self.guidance_scale * (conditioned_noise - unconditioned_noise)
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            return latents

        pbar = tqdm(total=len(self.scheduler.timesteps), desc="Denoising")

        for start_idx in range(0, len(self.scheduler.timesteps), block_size):
            end_idx = min(start_idx + block_size, len(self.scheduler.timesteps))
            timesteps = self.scheduler.timesteps[start_idx : end_idx]
            latents = checkpoint(denoise_steps, latents, range(start_idx, end_idx), timesteps, use_reentrant=False)
            # latents = denoise_steps(latents, range(start_idx, end_idx), timesteps)
            if self.debug and end_idx - end_idx % self.save_interval >= start_idx:
                images = self._decode_latents(latents.detach())
                pil_image = self.to_pil_image(images[0])
                self.ctx.info["intermediate_images"].append({
                    "image": pil_image,
                    "epoch": self.ctx.epoch,
                    "step_idx": end_idx,
                })
            pbar.update(len(timesteps))

        pbar.close()

        return latents

    def _decode_latents(self, latents: torch.Tensor):
        batch_size = latents.shape[0]
        assert batch_size == 1
        latents = 1 / 0.18215 * latents
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        return images
    
    def _prepare_conditions(self, prompt: str):
        object_pos = self.layout_predictor.inference_sentence(prompt)
        num_objects = len(object_pos)

        self.ctx.info["prompt"] = prompt
        self.ctx.info["object_pos"] = object_pos

        def format_prompt(s: str):
            return "a photo of " + s.lower()

        prompts = [prompt] + list(object_pos.keys())
        text_input = self.tokenizer([format_prompt(p) for p in prompts], padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
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
        self.ctx.info.clear()
        encoder_conds = self._prepare_conditions(prompt=prompt)

        
        optimizer_cls = None
        if self.cfg.optimization.optimizer_type == "Adam":
            optimizer_cls = torch.optim.Adam
        elif self.cfg.optimization.optimizer_type == "AdamW":
            optimizer_cls = torch.optim.AdamW
        elif self.cfg.optimization.optimizer_type == "SGD":
            optimizer_cls = torch.optim.SGD
        else:
            raise ValueError(f'Unsupported optimizer type: {self.cfg.optimization.optimizer_type}')

        num_objects = encoder_conds["num_objects"]
        self.ctx.params = nn.Parameter(
            torch.full(
                (self.num_inference_steps + 1, num_objects),
                fill_value=self.cfg.optimization.weight_initialize_coef / num_objects if num_objects else 0.0,
                dtype=torch.float32,
                device=self.device,
            )
        )

        
        self.ctx.info["intermediate_images"] = []
        
        org_latents = self._gen_latents(batch_size=1, seed=seed)
        latents = nn.Parameter(org_latents)

        optimizer = optimizer_cls([self.ctx.params, latents], **self.cfg.optimization.optimizer_kwargs)
        with torch.set_grad_enabled(True): # be careful not to include the text encoder in the gradient computation
            for epoch in tqdm(range(1, self.cfg.optimization.num_steps + 1), desc="Optimizing"):
                self.ctx.epoch = epoch
                optimizer.zero_grad()
                denoised_latents = self._denoise_latents(
                    latents = latents,
                    encoder_conds = encoder_conds,
                )
                images = self._decode_latents(denoised_latents)
                loss_clip = self.clip_loss(images, encoder_conds)
                loss_reg = latents.pow(2).mean()
                loss = loss_clip + self.reg_coef * loss_reg

                if loss.requires_grad:
                    loss.backward()
                    self.ctx.params.grad = self.ctx.params.grad.clamp(-self.cfg.optimization.gradient_clipping, self.cfg.optimization.gradient_clipping)
                    optimizer.step()
                    self.ctx.params.data = self.ctx.params.data.clamp(self.cfg.optimization.clip_range[0], self.cfg.optimization.clip_range[1])
                else :
                    rich.print("[red]Warning: loss does not require grad[/red]")
                    break
                rich.print(f"[green]Loss: {loss.item():.4f} = {loss_clip.item():.4f} + {self.reg_coef} * {loss_reg.item():.4f}[/green]")
        self.ctx.parmas = None

        if self.debug:
            self.ctx.info["final_image"] = self.to_pil_image(images[0].detach())

        return images
    
    def to_pil_image(self, image_tensor: torch.Tensor):
        image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).round().astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        return pil_image

    def save_info(self, path: Union[str, Path]):
        assert self.debug, "Cannot save info when debug is False"
        if isinstance(path, str):
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        info = self.ctx.info
        info_json = info.copy()
        info_json.pop("intermediate_images")
        info_json.pop("final_image")

        with open(str(path / "info.json"), "w") as f:
            json.dump(info_json, f, indent=4)

        for image_info in info["intermediate_images"]:
            epoch, step_idx = image_info["epoch"], image_info["step_idx"]
            epoch_path = path / f"epoch_{epoch}"
            epoch_path.mkdir(parents=True, exist_ok=True)
            image_info["image"].save(str(epoch_path / f"{step_idx:04d}.png"))
        
        if "final_image" in info:
            info["final_image"].save(str(path / "final_image.png"))

        final_image_with_caption = info["final_image"]
        final_image_with_caption = np.asarray(final_image_with_caption)
        final_image_with_caption = final_image_with_caption[:, :, ::-1].copy()
        for key, value in info["object_pos"].items():
            if isinstance(value, str):
                continue
            x, y = value
            x, y = int(x * self.width), int(y * self.height)
            cv2.putText(
                final_image_with_caption, key, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
            )
            assert self.width == self.height
            cv2.circle(
                final_image_with_caption, (x, y), int(self.radius * self.width), (255, 0, 0), 2
            )

        cv2.imwrite(str(path / "final_image_with_caption.png"), final_image_with_caption)


if __name__ == "__main__":
    sampler = AttnOptimSampler(
        cfg = OmegaConf.load("configs/default.yaml"),
        device = "cuda",
        debug = True
    )
    # images = sampler.sample(prompt="a bird flying above an elephant")
    image = sampler.sample(prompt="green house")
    save_path = f"outputs/test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    sampler.save_info(save_path)