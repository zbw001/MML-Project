import numpy as np
from pycocotools.coco import COCO
import numpy as np
import pickle
import json, os, random, math
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from fairseq.models.roberta import alignment_utils
import random
from mml_project.layout_predictor.paths import DATA_PATH
random.seed(42)

class COCORelDataset(Dataset):
    def __init__(self, instances_json, stuff_json=None,
               stuff_only=True, normalize_images=True, max_samples=None,
               include_relationships=True, min_object_size=0.02,
               sentence_size=128, is_mask=True, is_std=False,
               min_objects_per_image=3, max_objects_per_image=8,
               include_other=False, instance_whitelist=None, stuff_whitelist=None, obj_id_v2=False):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Modify:
        Input: Text
        Output: bbox

        Inputs:
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - normalize_image: If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - include_relationships: If True then include spatial relationships; if
          False then only include the trivial __in_image__ relationship.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
          
        - 0 for PAD, 1 for BOS, 2 for EOS, 3 for MASK
        - [PAD], [CLS], [SEP], [MASK]
        """
        
        super(Dataset, self).__init__()

        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')
            
        self.is_std = is_std
        self.is_mask = is_mask
        self.max_samples = max_samples
        self.sentence_size = sentence_size
        self.include_relationships = include_relationships
        self.obj_id_v2 = obj_id_v2

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)

        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r') as f:
                stuff_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        self.vocab = {
          'object_name_to_idx': {},
          'pred_name_to_idx': {},
          'object_pred_name_to_idx': {},
          'object_pred_idx_to_name': {},
        }
        # setting predictes
        self.snetence_token = ['[PAD]', '[CLS]', '[SEP]', '[MASK]']
        
        self.vocab['pred_idx_to_name'] = [
          '__in_image__',
          'left of',
          'right of',
          'above',
          'below',
          'inside',
          'surrounding',
        ]
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx
            
        object_idx_to_name = {}
        all_instance_categories = []
        for idx, token in enumerate(self.snetence_token):
            self.vocab['object_name_to_idx'][token] = idx
            
        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = len(self.snetence_token)
        
        for category_data in instances_data['categories']:
            category_id = category_data['id'] + len(self.snetence_token)
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
            
        all_stuff_categories = []
        if stuff_data:
            for category_data in stuff_data['categories']:
                category_id = category_data['id'] + len(self.snetence_token)
                category_name = category_data['name']
                all_stuff_categories.append(category_name)
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id
        
        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name
        
        self.vocab['object_pred_name_to_idx']
        all_vocabs = []
        for idx, name in enumerate(self.vocab['object_name_to_idx'].keys()):
            all_vocabs.append(name)
        for idx, name in enumerate(self.vocab['pred_name_to_idx'].keys()):
            all_vocabs.append(name)
        for i in range(len(all_vocabs)):
            self.vocab['object_pred_name_to_idx'][all_vocabs[i]] = i
        for i in range(len(all_vocabs)):
            self.vocab['object_pred_idx_to_name'][i] = all_vocabs[i]
        
        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
        category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

        # Add object data from instances
        self.image_id_to_objects = defaultdict(list)
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > min_object_size
            object_name = \
                object_idx_to_name[object_data['category_id']+len(self.snetence_token)]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok:
                self.image_id_to_objects[image_id].append(object_data)

        # Add object data from stuff
        if stuff_data:
            image_ids_with_stuff = set()
            for object_data in stuff_data['annotations']:
                image_id = object_data['image_id']
                image_ids_with_stuff.add(image_id)
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                box_ok = box_area > min_object_size
                object_name = \
                    object_idx_to_name[object_data['category_id']+len(self.snetence_token)]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok:
                    self.image_id_to_objects[image_id].append(object_data)

            if stuff_only:
                new_image_ids = []
                for image_id in self.image_ids:
                    if image_id in image_ids_with_stuff:
                        new_image_ids.append(image_id)
                self.image_ids = new_image_ids

                all_image_ids = set(self.image_id_to_filename.keys())
                image_ids_to_remove = all_image_ids - image_ids_with_stuff
                for image_id in image_ids_to_remove:
                    self.image_id_to_filename.pop(image_id, None)
                    self.image_id_to_size.pop(image_id, None)
                    self.image_id_to_objects.pop(image_id, None)

        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids
        
        # boxes = [xc, yc, w, h] normalized
        all_boxes = []
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            W, H = self.image_id_to_size[image_id]
            x0, y0, w, h = object_data['bbox']
            xc, yc, w, h = (x0+w/2.)/W, (y0+h/2.)/H, w/W, h/H
            all_boxes.append([xc, yc, w, h])
        if stuff_data:
            for object_data in stuff_data['annotations']:
                image_id = object_data['image_id']
                W, H = self.image_id_to_size[image_id]
                x0, y0, w, h = object_data['bbox']
                xc, yc, w, h = (x0+w/2.)/W, (y0+h/2.)/H, w/W, h/H
                all_boxes.append([xc, yc, w, h])

        all_boxes = np.array(all_boxes)
        self.x_mean, self.x_std = all_boxes[:,0].mean(), all_boxes[:,0].std()
        self.y_mean, self.y_std = all_boxes[:,1].mean(), all_boxes[:,1].std()
        self.w_mean, self.w_std = all_boxes[:,2].mean(), all_boxes[:,2].std()
        self.h_mean, self.h_std = all_boxes[:,3].mean(), all_boxes[:,3].std()
        sta_dict = {}
        sta_dict['x_mean'], sta_dict['x_std'] = self.x_mean, self.x_std
        sta_dict['y_mean'], sta_dict['y_std'] = self.y_mean, self.y_std
        sta_dict['w_mean'], sta_dict['w_std'] = self.w_mean, self.w_std
        sta_dict['h_mean'], sta_dict['h_std'] = self.h_mean, self.h_std
        
        sta_dict_path = os.path.dirname(instances_json)
        with open(os.path.join(sta_dict_path,'sta_dict.json'), 'w') as fp:
            json.dump(sta_dict, fp)
        with open(str(DATA_PATH / "coco" /"parsed_caption_label_dict.pkl"), "rb") as f:
            anno_text = pickle.load(f)
        self.text = anno_text
        # remove some ids from self.image_ids; those ids do not have parsed words
        all_possible_ids = anno_text.keys()
        real_ids = []
        for possible_id in self.image_ids:
            if possible_id in all_possible_ids:
                real_ids.append(possible_id)
        self.image_ids = real_ids
        # assign bbox for self.text
        self.image_ids_with_bbox = []
        for data_index, each in enumerate(self.image_ids):
            image_id = each
            image = self.text[image_id][0]
            cands = image[2:]
            found=False
            try:
                bboxs = self.image_id_to_objects[image_id]
                W, H = self.image_id_to_size[image_id]
                for cand_index, each_cand in enumerate(cands):
                    candidate_name = each_cand[1]
                    for bbox in bboxs:
                        if self.vocab['object_idx_to_name'][bbox['category_id'] + 4] == candidate_name:
                            x,y,w,h = bbox['bbox']
                            each_cand.append([(x+w/2.)/W, (y+h/2.)/H, w/W, h/H])
                            found=True
                            break
                if found:
                    self.image_ids_with_bbox.append(image_id)
            except:
                continue
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        self.tokenizer = alignment_utils.spacy_tokenizer()

        import pickle as pkl
        with open(str(DATA_PATH / "gpt-3.pkl"), "rb") as f:
            self.gpt3 = pkl.load(f)        
            
    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def __len__(self):
        return len(self.gpt3) * 2

    def sta_normalized(self, box):
        """
        (x-mean)/std
        """
        box[0] = (box[0]-self.x_mean)/self.x_std
        box[1] = (box[1]-self.y_mean)/self.y_std
        box[2] = (box[2]-self.w_mean)/self.w_std
        box[3] = (box[3]-self.h_mean)/self.h_std
        return box
    
    def __getitem__(self, index):
        if index < len(self.gpt3):
            image = self.gpt3[index]
            caption = image[0]
            bpe_toks = self.roberta.encode(caption)
            sample_caption_tokens = image[1]
            alignment = alignment_utils.align_bpe_to_words(self.roberta, bpe_toks, sample_caption_tokens)
            relation = image[3]
            object_index = image[2]
            real_object_index = [] # "real" means the index have been fixed to roberta index
            for each_index in object_index:
                real_object_index.append(alignment[each_index][0])
            real_relation = []
            for each_relation in relation:
                assert len(each_relation) == 3
                real_each_relation = []
                real_each_relation.append(alignment[each_relation[0]][0])
                real_each_relation.append(alignment[each_relation[1]][0])
                real_each_relation.append(each_relation[2])
                real_relation.append(real_each_relation)
            

            if self.sentence_size >= bpe_toks.shape[0]:
                padding = torch.ones(self.sentence_size - bpe_toks.shape[0]).int()
                bpe_toks = torch.cat((bpe_toks, padding), dim = 0)
            else:
                bpe_toks = bpe_toks[:128]
            return [bpe_toks.unsqueeze(0), real_object_index, None, caption, real_relation]
        else:
            # random sample a self.text
            all_keys = list(self.image_ids_with_bbox)
            # all_keys = list(self.text.keys())
            sample_key = random.sample(range(len(all_keys)), k=1)[0]
            curr_index = all_keys[sample_key]
            image = self.text[curr_index][0] # always the first description
            caption = image[0]
            bpe_toks = self.roberta.encode(caption)
            sample_caption_tokens = image[1]
            alignment = alignment_utils.align_bpe_to_words(self.roberta, bpe_toks, sample_caption_tokens)
            real_object_index = []
            returnbboxs = []
            cands = image[2:]
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

if __name__ == '__main__':
    ins_train_path = '../data/coco/instances_train2017.json'
    sta_train_path = '../data/coco/stuff_train2017.json'
    COCO = COCORelDataset(ins_train_path, sta_train_path)
    print(COCO.vocab)
    print(len(COCO))
