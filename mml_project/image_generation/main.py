from diffusers.pipelines import StableDiffusionPipeline
from diffusers.schedulers import PNDMScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
import torch
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

# todo: 不要写一堆 cuda
# 注意，现在 no_grad!

revision = "fp16"
dtype = torch.float32
device = torch.device("cuda")

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision=revision, torch_dtype=dtype).to(device)

unet, vae, tokenizer, text_encoder, scheduler = pipeline.unet, pipeline.vae, pipeline.tokenizer, pipeline.text_encoder, pipeline.scheduler

height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion
num_inference_steps = 50            # Number of denoising steps
guidance_scale = 7.5                # Scale for classifier-free guidance

generator = torch.manual_seed(0)    # Seed generator to create the inital latent noise
prompt = ["a photograph of an astronaut riding a horse"]
batch_size = 1

text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0] 

text_embeddings = torch.cat([uncond_embeddings, text_embeddings]) # concat into a single batch

latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    dtype=dtype,
    generator=generator
).to(device)

scheduler.set_timesteps(num_inference_steps)
latents = latents * scheduler.init_noise_sigma

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
image = (image * 255).round().astype(np.uint8)
pil_images = [Image.fromarray(img) for img in image]
