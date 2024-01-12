import inflect
import torch
import pickle
from omegaconf import OmegaConf
from pathlib import Path
from nltk.corpus import wordnet
from typing import Union
from mml_project.layout_predictor.paths import DATA_PATH, CHECKPOINT_PATH, CONFIG_PATH
from mml_project.layout_predictor.model import build_model
from nltk.corpus import stopwords
from fairseq.models.roberta import alignment_utils

import nltk
import spacy

def setup_nltk_and_spacy():
    nltk.download('wordnet')
    nltk.download('stopwords')
    try:
        spacy.load("en_core_web_sm")
    except:
        spacy.cli.download("en_core_web_sm")

setup_nltk_and_spacy()

class LayoutPredictor:
    def __init__(self, cfg_path: Union[str, Path], model_path: Union[str, Path]):
        if not Path(cfg_path).is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.engine = inflect.engine()
        self.word_set = {}
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')

        self._load_word_set()
        model_cfg = OmegaConf.load(cfg_path)
        self.model = build_model(model_cfg)
        checkpoint = torch.load(model_path)

        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.cuda()


        self.nlp = spacy.load("en_core_web_sm")
        self.stoplist = set(stopwords.words("english")).union(
            self.nlp.Defaults.stop_words
        )

    def inference_sentence(self, sentence: str) -> None:
        with torch.no_grad():
            def check_relation(sentence, object_index):
                bpe_toks = self.roberta.encode(sentence)
                padding = torch.ones(128 - bpe_toks.shape[0]).int()
                bpe_toks = torch.cat((bpe_toks, padding), dim = 0)
                src_mask = (bpe_toks != 1).to(bpe_toks)
                bpe_toks = bpe_toks.unsqueeze(0).to("cuda")
                src_mask = src_mask.unsqueeze(0).to("cuda").unsqueeze(0)
                trg_tmp = bpe_toks[:,:-1].to("cuda")
                trg_mask = (trg_tmp != 1).unsqueeze(1)
                trg_mask[:,0] = 1
                doc = self.nlp(sentence)
                alignment = alignment_utils.align_bpe_to_words(self.roberta, self.roberta.encode(sentence), doc)
                object_tensor = torch.zeros(128).to(torch.bool)
                for each_object_index in object_index:
                    object_tensor[alignment[each_object_index]] = True
                object_tensor = object_tensor.unsqueeze(0)
                output1, _, _, _ = self.model(bpe_toks, src_mask, None, trg_mask=trg_mask, object_pos_tensor=object_tensor)
                return output1, alignment

            sentence = sentence.replace("\n", "")
            sentence = sentence.rstrip()
            sentence = sentence.lstrip()
            doc = self.nlp(sentence)
            pos = []
            for chunk in doc.noun_chunks:
                full_noun = chunk.text
                if full_noun.lower() in self.stoplist:
                    continue
                key_noun = chunk.root.text
                word_index = chunk.root.i
                key_noun = key_noun.lower()
                if self._check_in_mscoco(full_noun):
                    pos.append(word_index)
            try:
                output, alignment = check_relation(sentence, pos)
            except:
                return None
            index = 0
            print("Sentence: %s"%(sentence))
            results = {}
            for chunk in doc.noun_chunks:
                if self._check_in_mscoco(chunk.text):
                    result_index = alignment[pos[index]][0]
                    x_cord = output[:,result_index][0][0]
                    y_cord = output[:,result_index][0][1]
                    print("%s position: (%.3f, %.3f)"%(chunk.text, x_cord, y_cord))
                    results[chunk.text] = [x_cord.item(), y_cord.item()]
                    index += 1
            return results

    def _check_in_mscoco(self, noun_pharse: str) -> bool:
        for each_cate in self.word_set:
            if each_cate in noun_pharse:
                return True
        return False

    def _load_word_set(self) -> None:
        with open(str(DATA_PATH / 'coco' / 'category_dict.pkl'), 'rb') as f:
            all_categories = pickle.load(f)
        for each in all_categories:
            self.word_set[each] = [each.lower()]
        for each in self.word_set:
            synonyms = []
            for syn in wordnet.synsets(each, pos='n'):
                for l in syn.lemmas():
                    synonyms.append(l.name().lower())
            synonyms.append(self.engine.plural(each))
            synonyms = set(synonyms)
            for each_syn in synonyms:
                self.word_set[each].append(each_syn)

# debug
if __name__ == "__main__":
    layout_predictor = LayoutPredictor(
        cfg_path=CONFIG_PATH / "coco_seq2seq_v9_ablation_4.yaml",
        model_path=CHECKPOINT_PATH / "checkpoint_90_0.0.pth"
    )
    print(layout_predictor.inference_sentence("The silver bed was situated to the right of the white couch."))