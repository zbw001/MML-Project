import os
import logging
from .base_data_loader import BaseDataLoader
from torch.utils.data.dataloader import default_collate

from .base_data_loader import BaseDataLoader
from ..dataset.COCODataset import COCORelDataset

class DataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers=0, collate_fn=None):
        if collate_fn is None:
            collate_fn = default_collate
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn)
        
def build_loader(cfg):
    logger = logging.getLogger('dataloader')
    
    data_dir = cfg['DATASETS']['DATA_DIR_PATH']
    batch_size = cfg['SOLVER']['BATCH_SIZE']
    batch_size = cfg['SOLVER']['BATCH_SIZE']
    shuffle = cfg['DATALOADER']['SHUFFLE']
    validation_split =cfg['DATALOADER']['VAL_SPLIT']
    num_workers = cfg['DATALOADER']['NUM_WORKER']
    
    if cfg['DATASETS']['NAME'] == 'coco':
        ins_data_path = os.path.join(data_dir, 'instances_train2017.json')
        sta_data_path = os.path.join(data_dir,'stuff_train2017.json')
        dataset = COCORelDataset(ins_data_path, sta_data_path)
    else:
        raise Exception("Sorry, we only support coco datasets.")
    
    mode = 'Pretrain' if cfg['MODEL']['PRETRAIN'] else 'Seq2Seq'
    logger.info('Setup [{}] dataset in [{}] mode.'.format(cfg['DATASETS']['NAME'], mode))
    logger.info('[{}] dataset in [{}] mode => Test dataset {}.'.format(cfg['DATASETS']['NAME'], mode))
    mycollator=None
    def mycollator(batch):
        return {'bpe_toks': [batch[i][0] for i in range(len(batch))], 'object_index': [batch[i][1] for i in range(len(batch))], 'caption': [batch[i][3] for i in range(len(batch))], \
            'relation': [batch[i][4] for i in range(len(batch))]}
    return DataLoader(dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=mycollator)
