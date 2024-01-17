import torch
from typing import Dict, Union
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.schedulers import PNDMScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
import torch
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

class AttnOptimSampler:
    revision = "fp16"
    dtype = torch.float16
    width = 512
    height = 512

    def __init__(self, cfg: DictConfig, device: Union[torch.device, str] = "cuda"):
        self.device = device
        
        pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision=self.revision, torch_dtype=self.dtype).to(self.device)

        self.unet, self.vae, self.tokenizer, self.text_encoder, self.scheduler = pipeline.unet, pipeline.vae, pipeline.tokenizer, pipeline.text_encoder, pipeline.scheduler

        self.cfg = cfg
        self.guidance_scale = cfg.guidance_scale
        self.num_inference_steps = cfg.num_inference_steps

    def _gen_latents(self, batch_size: int = 1, seed: int = 42):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, self.height // 8, self.width // 8),
            dtype=self.dtype,
            generator=generator,
            device=self.device
        )
        return latents
    
    def _denoise_latents(self, latents: torch.Tensor, text_embeddings: Dict[str, torch.Tensor], object_pos: Dict[str, torch.Tensor]):
        batch_size = latents.shape[0]
        assert batch_size == 1
        self.scheduler.set_timesteps(self.num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma

        keys = list(text_embeddings.keys())
        text_embeddings_tensor = torch.cat([text_embeddings[key] for key in keys])

        for t in tqdm(self.scheduler.timesteps):
            unet_input = torch.cat([latents] * len(keys))
            unet_input = self.scheduler.scale_model_input(unet_input, timestep=t)

            with torch.no_grad():
                unet_output = self.unet(unet_input, t, encoder_hidden_states=text_embeddings_tensor).sample

            noise_preds = dict(zip(keys, unet_output.chunk(len(keys))))
            noise_pred = noise_preds["uncond"] + self.guidance_scale * (noise_preds["input"] - noise_preds["uncond"])
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def _decode_latents(self, latents: torch.Tensor):
        batch_size = latents.shape[0]
        assert batch_size == 1
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            images = self.vae.decode(latents).sample

        images = (images / 2 + 0.5).clamp(0, 1)
        return images

    def sample(self, prompt: str, seed: int = 42):
        batch_size = 1

        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        input_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0] 

        text_embeddings = {
            "uncond": uncond_embeddings,
            "input": input_embeddings
        }

        latents = self._gen_latents(batch_size=batch_size, seed=seed)
        denoised_latents = self._denoise_latents(
            latents=latents,
            text_embeddings=text_embeddings,
            object_pos=None
        )

        images = self._decode_latents(denoised_latents)

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
    # test no grad
    images = sampler.sample(prompt="a photograph of an astronaut riding a horse")
    pil_image = sampler.to_pil_image(images[0])
    file_name = f"test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    pil_image.save("outputs/" + file_name)