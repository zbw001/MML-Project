import inflect
import torch
import pickle
from omegaconf import OmegaConf
from pathlib import Path
from nltk.corpus import wordnet
from typing import Union
from mml_project.layout_predictor.paths import DATA_PATH, CHECKPOINT_PATH, CONFIG_PATH
from nltk.corpus import stopwords
from fairseq.models.roberta import alignment_utils
from mml_project.layout_predictor.model.text2coord import Text2Coord
from nltk.wsd import lesk

import nltk
import spacy

from mml_project.layout_predictor.utils.gmm import GMM2D

def setup_nltk_and_spacy():
    nltk.download('wordnet')
    nltk.download('stopwords')
    try:
        spacy.load("en_core_web_md")
    except:
        spacy.cli.download("en_core_web_md")

setup_nltk_and_spacy()

class LayoutPredictor:
    def __init__(self, cfg_path: Union[str, Path], model_path: Union[str, Path], device: Union[torch.device, str] = "cuda"):
        if not Path(cfg_path).is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.device = device
        self.engine = inflect.engine()
        self._load_word_set()

        model_cfg = OmegaConf.load(cfg_path)
        self.model = Text2Coord(model_cfg)
        checkpoint = torch.load(model_path)
        state_dict_to_load = {}
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('model.'):
                state_dict_to_load[k[6:]] = v
            elif k.startswith('encoder.model.encoder.sentence_encoder'):
                state_dict_to_load[k] = v
            elif k.startswith('bbox_head.Decoder'):
                if k.startswith('bbox_head.Decoder.output_Layer'):
                    state_dict_to_load[k.replace('bbox_head.Decoder', 'coord_head')] = v
                elif k.startswith('bbox_head.Decoder.box_predictor.xy_bivariate'):
                    state_dict_to_load[k.replace('bbox_head.Decoder', 'coord_head').replace('xy_bivariate', 'fc')] = v

        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict_to_load, strict=False)
        for key in missing_keys:
            print(f"Missing key: {key}")
        for key in unexpected_keys:
            print(f"Unexpected key: {key}")
        self.model.to(self.device)
        self.model.eval()

        self.nlp = spacy.load("en_core_web_md")
        self.stoplist = set(stopwords.words("english")).union(
            self.nlp.Defaults.stop_words
        )

    def _check_in_mscoco(self, noun_pharse: str) -> bool:
        for each_cate in self.word_set:
            if each_cate in noun_pharse:
                return True
        return False

    def _load_word_set(self) -> None:
        self.word_set = {}
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

    @torch.no_grad()
    def inference_sentence(self, sentence: str, noun_phrases=None):
        def check_relation(sentence, object_index):
            bpe_toks = self.model.encoder.encode(sentence)
            padding = torch.ones(128 - bpe_toks.shape[0]).int()
            bpe_toks = torch.cat((bpe_toks, padding), dim = 0)
            doc = self.nlp(sentence)
            alignment = alignment_utils.align_bpe_to_words(self.model.encoder, self.model.encoder.encode(sentence), doc)
            object_tensor = torch.zeros(128).to(torch.bool)
            for each_object_index in object_index:
                object_tensor[alignment[each_object_index]] = True
            bpe_toks = bpe_toks.unsqueeze(0)
            object_tensor = object_tensor.unsqueeze(0)
            gmm: GMM2D = self.model(bpe_toks.to(self.device), object_pos=object_tensor.to(self.device))[0]

            return gmm, alignment

        sentence = sentence.replace("\n", "")
        sentence = sentence.rstrip()
        sentence = sentence.lstrip()
        doc = self.nlp(sentence)
        chunks, pos = [], []
        for chunk in doc.noun_chunks:
            word_index = chunk.root.i
            if noun_phrases is not None:
                for phrase in noun_phrases:
                    if phrase.lower() in chunk.text.lower():
                        chunks.append(chunk)
                        pos.append(word_index)
                        break
                continue
            if self._check_chunk(chunk, sentence):
                chunks.append(chunk)
                pos.append(word_index)
        try:
            gmm, alignment = check_relation(sentence, pos)
        except:
            return None
        index = 0
        print("Sentence: %s"%(sentence))
        results = {}
        for chunk, p in zip(chunks, pos):
            result_index = alignment[p][0]
            x, y = gmm[result_index].sample()
            print("%s position: (%.3f, %.3f)" % (chunk.text, x, y))
            results[chunk.text] = [x.item(), y.item()]
            index += 1
        return results

    def _check_word(self, word, sentence):
        synset = lesk(sentence, word, 'n')
        hyper = lambda s: s.hypernyms()
        hypernym_chain = list(synset.closure(hyper))
        if wordnet.synset('object.n.01') in hypernym_chain:
            return True
        return False

    def _check_chunk(self, chunk, sentence):
        full_noun = chunk.text
        if full_noun.lower() in self.stoplist:
            return False
        
        key_noun = chunk.root.text
        key_noun = key_noun.lower()
        return self._check_in_mscoco(chunk.text) or self._check_word(key_noun, sentence)
    
# debug
if __name__ == "__main__":
    layout_predictor = LayoutPredictor(
        cfg_path=CONFIG_PATH / "model" / "replicate.yaml",
        model_path=CHECKPOINT_PATH / "checkpoint_90_0.0.pth"
    )
    print(layout_predictor.inference_sentence("a bed room with a bed and a large window"))
    print(layout_predictor.inference_sentence("The silver bed was situated to the right of the white couch."))
    print(layout_predictor.inference_sentence("a bed room with a bed and a large window"))
    print(layout_predictor.inference_sentence("an apple on the left of a table"))
    print(layout_predictor.inference_sentence("an apple on the top of a cup"))
    print(layout_predictor.inference_sentence("an apple placed on the left of a cup"))
    print(layout_predictor.inference_sentence("an apple placed on the right of a cup"))
    print(layout_predictor.inference_sentence("a bear flying high above a table"))
    print(layout_predictor.inference_sentence("a bear hiding under a chair"))