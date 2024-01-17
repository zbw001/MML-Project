import numpy as np
import pickle
import json, os, random
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from fairseq.models.roberta import alignment_utils
import random
from mml_project.layout_predictor.paths import DATA_PATH
random.seed(42)

class GPTDataset(Dataset):
    def __init__(self, images, roberta, sentence_size):
        self.images = images
        self.roberta = roberta
        self.sentence_size = sentence_size

    def __getitem__(self, index):
        caption, spacy_tokens, object_index, relations = self.images[index][:4]

        bpe_toks = self.roberta.encode(caption)
        alignment = alignment_utils.align_bpe_to_words(self.roberta, bpe_toks, spacy_tokens)

        real_object_index = [] # aligned with bpe_toks
        for index in object_index:
            real_object_index.append(alignment[index][0])
        real_relations = []
        for relation in relations:
            assert len(relation) == 3
            real_relation = [
                alignment[relation[0]][0],
                alignment[relation[1]][0],
                relation[2]
            ]
            real_relations.append(real_relation)

        if self.sentence_size >= bpe_toks.shape[0]:
            padding = torch.ones(self.sentence_size - bpe_toks.shape[0]).int()
            bpe_toks = torch.cat((bpe_toks, padding), dim = 0)
        else:
            bpe_toks = bpe_toks[:128]
        return [bpe_toks.unsqueeze(0), real_object_index, None, caption, real_relations]
    
    def __len__(self):
        return len(self.images)
    
class COCODataset(Dataset):
    def __init__(self, images, roberta, sentence_size):
        self.images = images
        self.roberta = roberta
        self.sentence_size = sentence_size

    def __getitem__(self, index):
        caption, spacy_tokens = self.images[index][: 2]

        bpe_toks = self.roberta.encode(caption)
        while spacy_tokens[-1].strip() == '':
            spacy_tokens = spacy_tokens[:-1] # temp fix for some captions ending with a space or newline
        alignment = alignment_utils.align_bpe_to_words(self.roberta, bpe_toks, spacy_tokens)

        real_object_index = []
        returnbboxs = []
        cands = self.images[index][2:]
        for each in cands:
            if len(each) == 4:
                try:
                    real_object_index.append(alignment[each[0]][0])
                    returnbboxs.append(each[3])
                except:
                    continue
        if self.sentence_size >= bpe_toks.shape[0]:
            padding = torch.ones(self.sentence_size - bpe_toks.shape[0]).int()
            bpe_toks = torch.cat((bpe_toks, padding), dim = 0)
        else:
            bpe_toks = bpe_toks[:128]
        return [bpe_toks.unsqueeze(0), real_object_index, True, caption, returnbboxs]
    
    def __len__(self):
        return len(self.images)
    
def load_datasets(instances_json, stuff_json=None,
               stuff_only=True, min_object_size=0.02,
               sentence_size=128,
               min_objects_per_image=3, max_objects_per_image=8,
               include_other=False, instance_whitelist=None, stuff_whitelist=None):
    '''
    This function is mostly copied from the original repo. 
    I don't think it's understandable for anyone except the original author.
    '''
    if stuff_only and stuff_json is None:
        print('WARNING: Got stuff_only=True but stuff_json=None.')
        print('Falling back to stuff_only=False.')

    with open(instances_json, 'r') as f:
        instances_data = json.load(f)

    stuff_data = None
    if stuff_json is not None and stuff_json != '':
        with open(stuff_json, 'r') as f:
            stuff_data = json.load(f)

    image_ids = []
    image_id_to_filename = {}
    image_id_to_size = {}
    for image_data in instances_data['images']:
        image_id = image_data['id']
        filename = image_data['file_name']
        width = image_data['width']
        height = image_data['height']
        image_ids.append(image_id)
        image_id_to_filename[image_id] = filename
        image_id_to_size[image_id] = (width, height)

    vocab = {
        'object_name_to_idx': {},
        'pred_name_to_idx': {},
        'object_pred_name_to_idx': {},
        'object_pred_idx_to_name': {},
    }
    # setting predictes
    sentence_token = ['[PAD]', '[CLS]', '[SEP]', '[MASK]']
    
    vocab['pred_idx_to_name'] = [
        '__in_image__',
        'left of',
        'right of',
        'above',
        'below',
        'inside',
        'surrounding',
    ]
    vocab['pred_name_to_idx'] = {}
    for idx, name in enumerate(vocab['pred_idx_to_name']):
        vocab['pred_name_to_idx'][name] = idx
        
    object_idx_to_name = {}
    all_instance_categories = []
    for idx, token in enumerate(sentence_token):
        vocab['object_name_to_idx'][token] = idx
        
    # COCO category labels start at 1, so use 0 for __image__
    vocab['object_name_to_idx']['__image__'] = len(sentence_token)
    
    for category_data in instances_data['categories']:
        category_id = category_data['id'] + len(sentence_token)
        category_name = category_data['name']
        all_instance_categories.append(category_name)
        object_idx_to_name[category_id] = category_name
        vocab['object_name_to_idx'][category_name] = category_id
        
    all_stuff_categories = []
    if stuff_data:
        for category_data in stuff_data['categories']:
            category_id = category_data['id'] + len(sentence_token)
            category_name = category_data['name']
            all_stuff_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            vocab['object_name_to_idx'][category_name] = category_id
    
    # Build object_idx_to_name
    name_to_idx = vocab['object_name_to_idx']
    assert len(name_to_idx) == len(set(name_to_idx.values()))
    max_object_idx = max(name_to_idx.values())
    idx_to_name = ['NONE'] * (1 + max_object_idx)
    for name, idx in vocab['object_name_to_idx'].items():
        idx_to_name[idx] = name
    vocab['object_idx_to_name'] = idx_to_name
    
    vocab['object_pred_name_to_idx']
    all_vocabs = []
    for idx, name in enumerate(vocab['object_name_to_idx'].keys()):
        all_vocabs.append(name)
    for idx, name in enumerate(vocab['pred_name_to_idx'].keys()):
        all_vocabs.append(name)
    for i in range(len(all_vocabs)):
        vocab['object_pred_name_to_idx'][all_vocabs[i]] = i
    for i in range(len(all_vocabs)):
        vocab['object_pred_idx_to_name'][i] = all_vocabs[i]
    
    if instance_whitelist is None:
        instance_whitelist = all_instance_categories
    if stuff_whitelist is None:
        stuff_whitelist = all_stuff_categories
    category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

    # Add object data from instances
    image_id_to_objects = defaultdict(list)
    for object_data in instances_data['annotations']:
        image_id = object_data['image_id']
        _, _, w, h = object_data['bbox']
        W, H = image_id_to_size[image_id]
        box_area = (w * h) / (W * H)
        box_ok = box_area > min_object_size
        object_name = \
            object_idx_to_name[object_data['category_id']+len(sentence_token)]
        category_ok = object_name in category_whitelist
        other_ok = object_name != 'other' or include_other
        if box_ok and category_ok and other_ok:
            image_id_to_objects[image_id].append(object_data)

    # Add object data from stuff
    if stuff_data:
        image_ids_with_stuff = set()
        for object_data in stuff_data['annotations']:
            image_id = object_data['image_id']
            image_ids_with_stuff.add(image_id)
            _, _, w, h = object_data['bbox']
            W, H = image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > min_object_size
            object_name = \
                object_idx_to_name[object_data['category_id']+len(sentence_token)]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok:
                image_id_to_objects[image_id].append(object_data)

        if stuff_only:
            new_image_ids = []
            for image_id in image_ids:
                if image_id in image_ids_with_stuff:
                    new_image_ids.append(image_id)
            image_ids = new_image_ids

            all_image_ids = set(image_id_to_filename.keys())
            image_ids_to_remove = all_image_ids - image_ids_with_stuff
            for image_id in image_ids_to_remove:
                image_id_to_filename.pop(image_id, None)
                image_id_to_size.pop(image_id, None)
                image_id_to_objects.pop(image_id, None)

    # Prune images that have too few or too many objects
    new_image_ids = []
    total_objs = 0
    for image_id in image_ids:
        num_objs = len(image_id_to_objects[image_id])
        total_objs += num_objs
        if min_objects_per_image <= num_objs <= max_objects_per_image:
            new_image_ids.append(image_id)
    image_ids = new_image_ids
    
    with open(str(DATA_PATH / "coco" /"parsed_caption_label_dict.pkl"), "rb") as f:
        anno_text = pickle.load(f)

    text = anno_text
    all_possible_ids = anno_text.keys()
    real_ids = []
    for possible_id in image_ids:
        if possible_id in all_possible_ids:
            real_ids.append(possible_id)
    image_ids = real_ids
    image_ids_with_bbox = []
    for _, each in enumerate(image_ids):
        image_id = each
        image = text[image_id][0]
        cands = image[2:]
        found = False
        try:
            bboxs = image_id_to_objects[image_id]
            W, H = image_id_to_size[image_id]
            for _, each_cand in enumerate(cands):
                candidate_name = each_cand[1]
                for bbox in bboxs:
                    if vocab['object_idx_to_name'][bbox['category_id'] + 4] == candidate_name:
                        x,y,w,h = bbox['bbox']
                        each_cand.append([(x+w/2.)/W, (y+h/2.)/H, w/W, h/H])
                        found = True
                        break
            if found:
                image_ids_with_bbox.append(image_id)
        except:
            continue
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')

    import pickle as pkl
    with open(str(DATA_PATH / "gpt-3.pkl"), "rb") as f:
        gpt3_data = pkl.load(f)        
    
    coco_dataset = COCODataset(
        images=[text[i][0] for i in image_ids_with_bbox],
        roberta=roberta,
        sentence_size=sentence_size,
    )

    gpt_dataset = GPTDataset(
        images=gpt3_data,
        roberta=roberta,
        sentence_size=sentence_size,
    )

    return coco_dataset, gpt_dataset

class MixedDataset(Dataset):
    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.rng = random.Random(42)
    
    def __len__(self):
        return min(len(self.dataset1), len(self.dataset2)) * 2 # match the original implementation
    
    def __getitem__(self, index):
        if self.rng.random() < 0.5:
            i = self.rng.randint(0, len(self.dataset1) - 1)
            return self.dataset1[i]
        else:
            i = self.rng.randint(0, len(self.dataset2) - 1)
            return self.dataset2[i]