
import logging
from mml_project.layout_predictor.model.Model import Rel2Bbox

logger = logging.getLogger('model')

def build_model(cfg):
    hidden_size = cfg['MODEL']['ENCODER']['HIDDEN_SIZE']
    dropout = cfg['MODEL']['ENCODER']['DROPOUT']

    model = Rel2Bbox(hidden_size=hidden_size, dropout=dropout, cfg=cfg)

    logger.info('Setup model {}.'.format(model.__class__.__name__))
    logger.info('Model structure:')
    logger.info(model)
    return model
