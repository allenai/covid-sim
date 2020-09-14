import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, RobertaModel, RobertaForMaskedLM, \
    RobertaTokenizer, RobertaConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig
import numpy as np
from typing import List, Tuple, Dict
import tqdm


class BertEncoder(object):

    def __init__(self, device='cpu', model="bert"):

        if model == "bert":
            config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)

        elif model == "scibert":
            config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=config)

        self.model.eval()
        self.model.to(device)
        self.device = device

    def tokenize(self, original_sentence: List[str]) -> Tuple[List[str], Dict[int, int]]:

        """
        Parameters
        ----------
        Returns
        -------
        bert_tokens: The sentence, tokenized by BERT tokenizer.
        orig_to_tok_map: An output dictionary consisting of a mapping (alignment) between indices in the original tokenized sentence, and indices in the sentence tokenized by the BERT tokenizer. See https://github.com/google-research/bert
        """

        bert_tokens = ["[CLS]"]
        orig_to_tok_map = {}
        tok_to_orig_map = {}
        has_subwords = False
        is_subword = []

        for i, w in enumerate(original_sentence):
            tokenized_w = self.tokenizer.tokenize(w)
            has_subwords = len(tokenized_w) > 1
            is_subword.append(has_subwords)
            bert_tokens.extend(tokenized_w)

            orig_to_tok_map[i] = len(bert_tokens) - 1

        tok_to_orig_map = {}

        keys = list(sorted(orig_to_tok_map.keys()))
        for k, k2 in zip(keys, keys[1:]):
            for k3 in range(orig_to_tok_map[k], orig_to_tok_map[k2]):
                tok_to_orig_map[k3] = k

        bert_tokens.append("[SEP]")
        return (bert_tokens, orig_to_tok_map, tok_to_orig_map)

    def encode(self, sentence: str, layers: List[int]):

        tokenized_text, orig2tok, tok_to_orig_map = self.tokenize(sentence.split(" "))
        # pos_ind_bert = orig2tok[pos_ind]
        # if np.random.random() < mask_prob:
        #    tokenized_text[pos_ind_bert] = self.mask
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor)

            all_layers = outputs[-1]
            layers_concat = torch.cat([all_layers[l] for l in layers], dim=-1)

            return layers_concat[0].detach().cpu().numpy(), tokenized_text, tok_to_orig_map, orig2tok