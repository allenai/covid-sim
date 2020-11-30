import pandas as pd
import tqdm
import pickle
import random
import itertools

import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, AutoTokenizer, AutoModel
import numpy as np
from typing import List
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Dict, Tuple
from scipy.spatial.distance import cosine as cosine_distance
from collections import defaultdict, Counter
import nltk
from nltk import ngrams as get_ngrams
from termcolor import colored
import streamlit as st

class BertModel(torch.nn.Module):

    def __init__(self, device: str, mode: str = "eval", load_existing = True):
        
        super().__init__()
        
        config = AutoConfig.from_pretrained('Shauli/RE-metric-model-siamese-spike', output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained('Shauli/RE-metric-model-siamese-spike')
        self.model = AutoModel.from_pretrained('Shauli/RE-metric-model-siamese-spike', config=config)            
        
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.linear_arg1_1 = torch.nn.Linear(768, 64) #torch.load("finetuned_model/metric_model/linear.pt") #torch.nn.Linear(768, 64)
        self.linear_arg1_1.load_state_dict(torch.load("linear1.pt"))
        self.linear_arg2_1 = torch.nn.Linear(768, 64)
        self.linear_arg1_2 = torch.nn.Linear(768, 64)
        self.linear_arg1_2.load_state_dict(torch.load("linear2.pt"))

        self.linear_arg2_2 = torch.nn.Linear(768, 64)
        self.linear_is_same_relation = torch.nn.Linear(768, 1)
        #self.linear_is_same_relation.load_state_dict(torch.load("finetuned_model/metric_model3/linear_rel_clf.pt"))
        
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        
        self.model.eval()

        
    def tokenize(self, original_sentence: List[str], add_cls = True, add_sep = True) -> Tuple[List[str], Dict[int, int]]:

        """
        Parameters
        ----------
        Returns
        -------
        bert_tokens: The sentence, tokenized by BERT tokenizer.
        orig_to_tok_map: An output dictionary consisting of a mapping (alignment) between indices in the original tokenized sentence, and indices in the sentence tokenized by the BERT tokenizer. See https://github.com/google-research/bert
        """

        if add_cls:
            bert_tokens = ["[CLS]"]
        else:
            bert_tokens = []
            
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

        if add_sep:
            bert_tokens.append("[SEP]")
        tok_to_orig_map = get_tok_to_orig_map(orig_to_tok_map, len(original_sentence), len(bert_tokens))        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device_to_use)

        return (bert_tokens, orig_to_tok_map, tok_to_orig_map, tokens_tensor)
    
    
    def forward(self, x):
        
        outputs = self.model(x)
        states = outputs[0][0] #[seq_len, 768]
        return states

             
    def forward_pass(self, x, is_query: bool):
        
        outputs = self.model(x)
        states = outputs[0][0] #[seq_len, 768]
        if is_query:
            states = self.linear_arg1_1(states)
        else:
            states = self.linear_arg1_2(states)
        return states
        
    def forward_with_loss_calculation_inference(self, x_2, sent1_arg1_vec, sent1_arg2_vec, n_max = 8, normalize = False):
        
        
        idx_arg1_all, idx_arg2_all, all_ngrams = None, None, None
        
        states_2 = self.forward_pass(x_2, is_query = False)
        is_neg_pred = torch.zeros(1)# self.linear_is_same_relation(states_2[0]) 
        
        all_ngrams = get_all_ngrams_spans(len(x_2[0]), [], start_ind = 0,
                                                      n_max = n_max)
        
        ngrams = [states_2[ngram[0]:ngram[1]].mean(dim=0) for ngram in all_ngrams]
        ngrams = torch.stack(ngrams).to(self.device_to_use)

        dist_arg1_all = torch.sqrt(((ngrams-sent1_arg1_vec)**2).sum(dim = 1))
        dist_arg2_all = torch.sqrt(((ngrams-sent1_arg2_vec)**2).sum(dim = 1))
        idx_arg1_all = torch.argsort(dist_arg1_all).detach().cpu().numpy()
        idx_arg2_all = torch.argsort(dist_arg2_all).detach().cpu().numpy()   

        return idx_arg1_all, idx_arg2_all, all_ngrams, is_neg_pred

    
    
def get_entity_range(index_orig, orig_to_tok_map):
    
    m = min(orig_to_tok_map.keys())
    if orig_to_tok_map[index_orig] == 1: return (1,2)
    if index_orig == 0: return (1, orig_to_tok_map[index_orig] + 1)
    
    before = index_orig - 1
    tok_range = (orig_to_tok_map[before] + 1, orig_to_tok_map[index_orig] + 1)
    return tok_range

def get_entity_range_multiword_expression(start_and_end, orig_to_tok_map):
    
    start, end = start_and_end
    start_range = get_entity_range(start, orig_to_tok_map)
    end_range = get_entity_range(end, orig_to_tok_map)
    return [start_range[0], end_range[1]]

def get_tok_to_orig_map(orig_to_tok_map, num_words, num_tokens):
    
    ranges = [get_entity_range(i, orig_to_tok_map) for i in range(num_words)]
    tok_to_orig_map = {}
    for i in range(num_words):
        min,max = ranges[i]
        for tok in range(min,max):
            tok_to_orig_map[tok] = i
    
    for tok in range(num_tokens):
        if tok not in tok_to_orig_map:
            tok_to_orig_map[tok] = num_words -1
    
    return tok_to_orig_map
        
def get_all_ngrams_spans(seq_len, forbidden_ranges: List[tuple], start_ind = 0, n_max = 15):
    
    def is_intersecting(ngram, forbidden_ranges):
        
        return [(r[1] > ngram[0] >= r[0]) or(r[1] > ngram[1] >= r[0]) for r in forbidden_ranges]
    
    all_ngrams = []
    for n in range(2,n_max+1):
        ngrams = list(get_ngrams(range(start_ind, seq_len), n))
        all_ngrams.extend(ngrams)
    
    all_ngrams = [(ngram[0], ngram[-1]) for ngram in all_ngrams]
    all_ngrams = [ngram for ngram in all_ngrams if not any(is_intersecting(ngram, forbidden_ranges))]
    return all_ngrams

def add_arguments(sent:str, arg1_start, arg1_end, arg2_start, arg2_end):
    
        s_lst = sent.split(" ")
        if arg1_start > arg2_start:
            arg1_start, arg2_start = arg2_start, arg1_start
            arg1_end, arg2_end = arg2_end, arg1_end
            arg1_str, arg2_str = "{{ARG2:", "<<ARG1:"
        else:
            arg1_str, arg2_str = "<<ARG1:", "{{ARG2:"
        
        s_with_args = s_lst[:arg1_start] + [arg1_str] + s_lst[arg1_start:arg1_end+1] + [">>"] + s_lst[arg1_end+1:arg2_start] + [arg2_str] + s_lst[arg2_start:arg2_end+1] + ["}}"] +s_lst[arg2_end+1:]  
        #s_with_args = s_lst[:arg1_start] + [arg1_str+s_lst[arg1_ind]] + s_lst[arg1_ind+1:arg2_ind] + [arg2_str+s_lst[arg2_ind]] + s_lst[arg2_ind+1:]
        s_with_args = " ".join(s_with_args).replace("ARG1: ", "ARG1:").replace("ARG2: ", "ARG2:")
        s_with_args = s_with_args.replace(" >>", ">>").replace(" }}", "}}")
        return s_with_args
    
def prepare_example(sent1, arg1_sent1, arg2_sent1):

            sent1 = add_arguments(sent1, arg1_sent1[0], arg1_sent1[1], arg2_sent1[0], arg2_sent1[1])
            idx = [[[arg1_sent1[0], arg1_sent1[1]], [arg2_sent1[0], arg2_sent1[1]]], [[0, 1], [0, 1]]] 
            return sent1, np.array(idx)
        
def evaluate_model(spike_df, sents, model, k, max_ngrams = 5, num_examples = 200):
    
    
    arg1_mean, arg2_mean = get_query_rep(spike_df, model, k = k)
    
    preds = []
    count = 0
    
    for i in range(len(sents)):
               
        with torch.no_grad():
            bert_tokens, orig_to_tok_map, tok_to_orig_map, tokens_tensor = model.tokenize(sents[i].split(" "), add_sep = True, add_cls = False)
            x = tokens_tensor
            
            idx_arg1_all, idx_arg2_all, all_ngrams, is_neg_pred = model.forward_with_loss_calculation_inference(x, arg1_mean, arg2_mean, orig_to_tok_map, mode = "eval", n_max=max_ngrams)
        preds.append({"sent": sents_concat, "tokens": bert_tokens, "tok2orig": tok_to_orig_map, "orig2tok": orig_to_tok_map,
                     "preds_arg1_tokens": idx_arg1_all, "preds_arg2_tokens": idx_arg2_all,
                      "all_ngrams": all_ngrams})
        
    return preds


def get_query_rep(spike_df, model, k = 5):

    query_sents = spike_df["sentence_text"].tolist()[:k]
    query_arg1_starts = spike_df["arg1_first_index"][:k]
    query_arg1_ends = spike_df["arg1_last_index"][:k]
    query_arg2_starts = spike_df["arg2_first_index"][:k]
    query_arg2_ends = spike_df["arg2_last_index"][:k]
    
    arg1_vecs, arg2_vecs = [], []
    
    for i in range(min(len(spike_df), k)):
    
        sent1 = query_sents[i] # use first query in all examples.
        arg1_sent1 = [query_arg1_starts[i], query_arg1_ends[i]]
        arg2_sent1 = [query_arg2_starts[i], query_arg2_ends[i]]
        sent1, idx = prepare_example(sent1, arg1_sent1, arg2_sent1)
        bert_tokens, orig_to_tok_map, tok_to_orig_map, tokens_tensor = model.tokenize(sent1.split(" "), add_sep = False, add_cls = True)
        
        with torch.no_grad():
            x = torch.unsqueeze(tokens_tensor,0)
            states = model.forward_pass(tokens_tensor, is_query = True)
            sent1_range_arg1 = get_entity_range_multiword_expression(idx[0][0], orig_to_tok_map)
            sent1_range_arg2 = get_entity_range_multiword_expression(idx[0][1], orig_to_tok_map)       
            sent1_arg1_vec, sent1_arg2_vec = states[sent1_range_arg1[0]:sent1_range_arg1[1]].mean(dim=0), states[sent1_range_arg2[0]:sent1_range_arg2[1]].mean(dim=0)
            arg1_vecs.append(sent1_arg1_vec)
            arg2_vecs.append(sent1_arg2_vec)
        
    
    arg1_mean = torch.stack(arg1_vecs, dim = 0).mean(dim = 0)
    arg2_mean = torch.stack(arg2_vecs, dim = 0).mean(dim = 0)
    
    return arg1_mean, arg2_mean

        
        
        
        
        
def main(model, results_sents, spike_df, num_results, max_ngrams):

    captures = []
    captures_tuples = []
    
    def pretty_print(sent, idx_arg1, idx_arg2):
    
        sent_lst = sent.split(" ")
        sent = " ".join(sent_lst[:idx_arg1[0]]) + " " + colored(" ".join(sent_lst[idx_arg1[0]:idx_arg1[1]]), "red") + " " + " ".join(sent_lst[idx_arg1[1]:])
        sent_lst = sent.split(" ")
        sent = " ".join(sent_lst[:idx_arg2[0]]) + " " + colored(" ".join(sent_lst[idx_arg2[0]:idx_arg2[1]]), "blue") + " " + " ".join(sent_lst[idx_arg2[1]:])
        return sent

    def perform_annotation(sent, arg_borders):

        def is_between(k, borders):
            return len([(s, e) for (s, e) in borders if s <= k < e]) != 0

        sent_lst = sent.split(" ")
        sent_new = []
        arg_colors = ["#8ef", "#fea", "#faa", "#fea", "#8ef", "#afa", "#d8ff35", "#8c443b", "#452963"]

        for i, w in enumerate(sent_lst):

            for arg in range(len(arg_borders)):
                if is_between(i, [arg_borders[arg]]):
                    sent_new.append((w, "ARG{}".format(arg+1), arg_colors[arg]))
                    break
            else:

                sent_new.append(" " + w + " ")

        return sent_new

    results_sents = results_sents[:num_results]
    results = evaluate_model(spike_df, results_sents, model, k=5, max_ngrams = max_ngrams, num_examples = len(results_sents))
    annotated = []
    
    for p in results:
        pred_arg1, pred_arg2 = p["preds_arg1_tokens"], p["preds_arg2_tokens"]
        ngram_pred_arg1_idx, ngram_pred_arg2_idx = p["all_ngrams"][pred_arg1[0]], p["all_ngrams"][pred_arg2[0]]     
        arg1_start = p["tok2orig"][ngram_pred_arg1_idx[0]]
        arg1_end = p["tok2orig"][ngram_pred_arg1_idx[1]]
        arg2_start = p["tok2orig"][ngram_pred_arg2_idx[0]]
        arg2_end = p["tok2orig"][ngram_pred_arg2_idx[1]]        
        sent = p["sent"]
        sent_lst = sent.split(" ")
        
        arg1_str = " ".join(sent_lst[arg1_start:arg1_end])
        arg2_str = " ".join(sent_lst[arg2_start:arg2_end])
        captures.append((arg1_str, arg2_str))
        captures_tuples.append("{}; {}".format(arg1_str, arg2_str))
        annotated_sent = perform_annotation(sent, [[arg1_start, arg1_end], [arg2_start, arg2_end]])
        #annotated_sent = annotated_sent[p["l"]:]
        annotated.append(annotated_sent)
    
    # aggregate arguments
    args1, args2 = list(zip(*captures))
    arg1_counter, arg2_counter, tuples_counter = Counter(args1), Counter(args2), Counter(captures_tuples)
    
    return annotated, arg1_counter.most_common(500), arg2_counter.most_common(500), tuples_counter.most_common(500)
