import argparse
import os
from omegaconf import DictConfig, OmegaConf

from mml_project.image_generation.sampler import AttnOptimSampler

# load dataset
gpt_path="datasets/gpt.txt"
mscoco_path="datasets/mscoco.txt"
vsr_path="datasets/vsr.txt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="image_generation/outputs/default/"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="image_generation/configs/default.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--gpt",
        action='store_true',
        help="run gpt datasets",
    )
    parser.add_argument(
        "--mscoco",
        action='store_true',
        help="run mscoco datasets",
    )
    parser.add_argument(
        "--vsr",
        action='store_true',
        help="run vsr datasets",
    )
    parser.add_argument(
        "--demo",
        action='store_true',
        help="run demo and debug",
    )
    opt = parser.parse_args()

    output_path=opt.outdir
    debug_flag=False
    
    prompts=[]
    if opt.demo:
        prompts.append(opt.prompt)
        output_path="image_generation/outputs/demo/"
        debug_flag=True
    elif opt.gpt:
        output_path="image_generation/outputs/gpt/"
        with open(gpt_path, "r") as f:
            contents = f.read()
            rows = contents.split('\n')
            rows = rows[:2000]
            for i in range(500):
                prompts.append(rows[4*i + 2][10:])
    elif opt.mscoco:
        output_path="image_generation/outputs/mscoco/"
        with open(mscoco_path, "r") as f:
            contents = f.read()
            rows = contents.split('\n')
            rows = rows[:500]
            prompts = []
            for i in range(500):
                prompts.append(rows[i])
    elif opt.vsr:
        output_path="image_generation/outputs/vsr/"
        with open(vsr_path, "r") as f:
            contents = f.read()
            rows = contents.split('\n')
            rows = rows[:500]
            prompts = []
            for i in range(500):
                prompts.append(rows[i])
    os.makedirs(output_path, exist_ok=True)

    seed = opt.seed
    sampler = AttnOptimSampler(
        cfg = OmegaConf.load(opt.config),
        device = "cuda"
    )
    for index in range(len(prompts)):
        images = sampler.sample(prompt=prompts[index], seed=seed, debug_flag=debug_flag)
        pil_image = sampler.to_pil_image(images[0])
        pil_image.save(output_path + "seed%d_index_%d.png"%(seed, index))
if __name__ == "__main__":
    main()