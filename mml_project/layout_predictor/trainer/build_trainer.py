import logging
from mml_project.layout_predictor.trainer.PretrainTrainer import PretrainTrainer


def build_trainer(cfg, model, dataloader, opt):
    logger = logging.getLogger('dataloader')
    
    T = PretrainTrainer(model = model, dataloader = dataloader, opt = opt, cfg = cfg)
    
    logger.info('Setup trainer {}.'.format(T.__class__.__name__))
    return T