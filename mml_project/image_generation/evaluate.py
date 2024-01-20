import argparse
from mml_project.image_generation.sampler import AttnOptimSampler
from omegaconf import OmegaConf
import json
from datetime import datetime
import shutil
from pathlib import Path
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the config file.')
    parser.add_argument('--dataset', type=str, default='../datasets/gpt4-2.json', help='Dataset to use.')
    args = parser.parse_args()

    save_path = Path(f"outputs/eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    save_path.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    dataset_path = Path(args.dataset)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    sampler = AttnOptimSampler(
        cfg = OmegaConf.load(str(config_path)),
        device = "cuda",
        debug = False
    )
    
    shutil.copy(str(config_path), str(save_path))
    data = json.load(open(dataset_path))

    for record in data:
        sampler.sample(record['sentence'], noun_phrases=record['noun_phrases'])
        output_dir = save_path / re.sub(r'[\\/*?:"<>|]', "_", record['sentence'])
        sampler.save_info(output_dir)