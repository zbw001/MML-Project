from mml_project.layout_predictor.loader import build_loader
from mml_project.layout_predictor.model import build_model
from mml_project.layout_predictor.trainer import build_trainer
from mml_project.layout_predictor.paths import CONFIG_PATH
from pathlib import Path
import argparse
import yaml
from omegaconf import OmegaConf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default=str(CONFIG_PATH / "coco_seq2seq_v9_ablation_4.yaml"))
    parser.add_argument('--default_cfg_path', type=str, default=str(CONFIG_PATH / "default.yaml"))
    parser.add_argument('--checkpoint', type=str, default=None)

    opt = parser.parse_args()

    cfg_path = Path(opt.cfg_path)
    default_cfg_path = Path(opt.default_cfg_path)
    if not cfg_path.exists():
        raise ValueError(f"Config file {cfg_path} does not exist")
    if not default_cfg_path.exists():
        raise ValueError(f"Default config file {default_cfg_path} does not exist")
    
    cfg = OmegaConf.load(opt.default_cfg_path)
    cfg.merge_with(OmegaConf.load(opt.cfg_path))

    with open(opt.default_cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    with open(opt.cfg_path, 'r') as f:
        cfg.update(yaml.safe_load(f))

    assert Path(cfg['OUTPUT']['OUTPUT_DIR']).exists(), \
        f"Output directory {cfg['OUTPUT']['OUTPUT_DIR']} does not exist" # 只用于 logger，也许可以删除
    assert Path(cfg['TEST']['OUTPUT_DIR']).exists(), \
        f"Test output directory {cfg['TEST']['OUTPUT_DIR']} does not exist" 

    data_loader = build_loader(cfg)
    model = build_model(cfg)
    
    trainer = build_trainer(cfg=cfg, model=model, dataloader=data_loader, opt=opt)
    trainer.train()