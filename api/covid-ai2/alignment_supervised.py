
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
        
        self.device = device
        
        if load_existing:
            config = AutoConfig.from_pretrained('Shauli/RE-metric-model-spike', output_hidden_states=True)
            self.model = AutoModel.from_pretrained('Shauli/RE-metric-model-spike', config=config)    
            self.tokenizer = AutoTokenizer.from_pretrained('Shauli/RE-metric-model-spike')
        
        else:
            config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)
            self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=config)    
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        
        if load_existing:
            self.linear_arg1_1 = torch.nn.Linear(768, 64)
            self.linear_arg1_1.load_state_dict(torch.load("linear.pt", map_location = torch.device('cpu')))
        else:
            self.linear_arg1_1 = torch.nn.Linear(768, 64)
        self.linear_arg2_1 = torch.nn.Linear(768, 64)
        self.linear_arg1_2 = torch.nn.Linear(768, 64)
        self.linear_arg2_2 = torch.nn.Linear(768, 64)
        
        if mode == "eval":
            
            self.model.eval()
        else:
            self.model.train()
        
        for p in self.model.parameters():
            p.requires_grad = False

        
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

        bert_tokens.append("[SEP]")
        tok_to_orig_map = get_tok_to_orig_map(orig_to_tok_map, len(original_sentence), len(bert_tokens))        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)

        return (bert_tokens, orig_to_tok_map, tok_to_orig_map, tokens_tensor)
    
    
    def forward(self, x):
        
        outputs = self.model(x)
        states = outputs[0][0] #[seq_len, 768]
        return states
    


    def forward_with_loss_calculation(self, bert_tokens, x, range_sent1, range_sent2, orig_to_tok_map, l, l_tokens,
                                      metric = "l2", n_max = 5, alpha = 0.075, mode = "train", normalize=False, nb=0):
        idx_arg1_all, idx_arg2_all, all_ngrams = None, None, None
        
        outputs = self.model(x)
        states = outputs[0][0] #[seq_len, 768]
        if metric == "cosine" or normalize:
            states = states / (torch.norm(states, dim = 1, keepdim = True)+1e-8)
        
        states = self.linear_arg1_1(states)
        arg1_sent1, arg2_sent1 = range_sent1
        arg1_sent2, arg2_sent2 = range_sent2
        
        sent1_arg1_vec, sent1_arg2_vec = states[arg1_sent1[0]:arg1_sent1[1]].mean(dim=0), states[arg2_sent1[0]:arg2_sent1[1]].mean(dim=0)
        sent2_arg1_vec, sent2_arg2_vec = states[arg1_sent2[0]:arg1_sent2[1]].mean(dim=0), states[arg2_sent2[0]:arg2_sent2[1]].mean(dim=0)        
        
        all_false_ngrams_ranges = get_all_ngrams_spans(len(states), [arg1_sent1, arg1_sent2, arg2_sent1, arg2_sent2], start_ind = l_tokens,
                                                      n_max = n_max)      
        negatives = [states[ngram[0]:ngram[1]].mean(dim=0) for ngram in all_false_ngrams_ranges]
        negatives_arg1 = negatives + [sent1_arg2_vec, sent2_arg2_vec]
        negatives_arg2 = negatives + [sent1_arg1_vec, sent2_arg1_vec]
        negatives_arg1 = torch.stack(negatives_arg1).to(self.device)
        negatives_arg2 = torch.stack(negatives_arg2).to(self.device)
        

        if mode == "eval":
            all_ngrams = get_all_ngrams_spans(len(states), [], start_ind = l_tokens,
                                                      n_max = n_max)
            ngrams = [states[ngram[0]:ngram[1]].mean(dim=0) for ngram in all_ngrams]
            ngrams = torch.stack(ngrams).to(self.device)
        
        
        if metric == "l2":
            dists_arg1 = torch.sqrt(((negatives_arg1-sent1_arg1_vec)**2).sum(dim = 1))
            dists_arg2 = torch.sqrt(((negatives_arg2-sent1_arg2_vec)**2).sum(dim = 1))
            dist_arg1_gold = (sent1_arg1_vec - sent2_arg1_vec).norm()
            dist_arg2_gold = (sent1_arg2_vec - sent2_arg2_vec).norm()
            if mode == "eval":
                dist_arg1_all = torch.sqrt(((ngrams-sent1_arg1_vec)**2).sum(dim = 1))
                dist_arg2_all = torch.sqrt(((ngrams-sent1_arg2_vec)**2).sum(dim = 1))
                idx_arg1_all = torch.argsort(dist_arg1_all).detach().cpu().numpy()
                idx_arg2_all = torch.argsort(dist_arg2_all).detach().cpu().numpy()   
                
        elif metric == "cosine":
            dists_arg1 = 1 - negatives_arg1@sent1_arg1_vec.T
            dists_arg2 = 1 - negatives_arg2@sent1_arg2_vec.T
            dist_arg1_gold = 1 - sent1_arg1_vec@sent2_arg1_vec.T
            dist_arg2_gold = 1 - sent1_arg2_vec@sent2_arg2_vec.T
        
        idx_arg1 = torch.argsort(dists_arg1).detach().cpu().numpy()
        idx_arg2 = torch.argsort(dists_arg2).detach().cpu().numpy()
        l = max(int(len(negatives)*0.3),1)
        k = random.choice(range(min(len(negatives), 2))) if np.random.random() < 0.5 else random.choice(range(l))

        dist_arg1_argmax = dists_arg1[idx_arg1[k]]
        dist_arg2_argmax = dists_arg2[idx_arg2[k]]
        
        loss_arg1 = torch.max(torch.zeros(1).to(self.device), dist_arg1_gold - dist_arg1_argmax + alpha)
        loss_arg2 = torch.max(torch.zeros(1).to(self.device), dist_arg2_gold - dist_arg2_argmax + alpha)
        
        # softmax triplet
        
        z = torch.max(dist_arg1_argmax, dist_arg1_gold)
        temp = 1
        pos_arg1 = torch.exp((dist_arg1_gold - z)/temp)
        neg_arg1 = torch.exp((dist_arg1_argmax - z)/temp)
        loss_arg1 = (pos_arg1 / (pos_arg1 + neg_arg1))**2

        z = torch.max(dist_arg2_argmax, dist_arg2_gold)
        pos_arg2 = torch.exp((dist_arg2_gold - z)/temp)
        neg_arg2 = torch.exp((dist_arg2_argmax - z)/temp)
        loss_arg2 = (pos_arg2 / (pos_arg2 + neg_arg2))**2

        
        loss = states[0,0:1]**2 #torch.zeros(1).to(self.device)
        loss2_isnan = np.isnan(loss_arg2.detach().cpu().numpy().item())
        loss1_isnan = np.isnan(loss_arg1.detach().cpu().numpy().item())
        if not loss2_isnan:
            loss += loss_arg2
        if not loss1_isnan:
            loss += loss_arg1
        
        if loss1_isnan or loss2_isnan:
            print("ERROR: nan loss", loss1_isnan, loss2_isnan, nb)
            return
        
        return loss, idx_arg1, idx_arg2, idx_arg1_all, idx_arg2_all, all_false_ngrams_ranges, all_ngrams
        #return loss, np.argsort(dists_arg1+mask_gold_arg1)

    
    
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
            arg1_str, arg2_str = "<<ARG2:", "<<ARG1:"
        else:
            arg1_str, arg2_str = "<<ARG1:", "<<ARG2:"
        
        s_with_args = s_lst[:arg1_start] + [arg1_str] + s_lst[arg1_start:arg1_end+1] + [">>"] + s_lst[arg1_end+1:arg2_start] + [arg2_str] + s_lst[arg2_start:arg2_end+1] + [">>"] +s_lst[arg2_end+1:]  
        #s_with_args = s_lst[:arg1_start] + [arg1_str+s_lst[arg1_ind]] + s_lst[arg1_ind+1:arg2_ind] + [arg2_str+s_lst[arg2_ind]] + s_lst[arg2_ind+1:]
        s_with_args = " ".join(s_with_args).replace("ARG1: ", "ARG1:").replace("ARG2: ", "ARG2:")
        s_with_args = s_with_args.replace(" >>", ">>")
        return s_with_args
    
def prepare_example(sent1, sent2, arg1_sent1, arg2_sent1):

            sent1 = add_arguments(sent1, arg1_sent1[0], arg1_sent1[1], arg2_sent1[0], arg2_sent1[1])
            l = len(sent1.split(" ")) + 1 
            #arg1_sent1[0] += l
            #arg1_sent1[1] += l
            #arg2_sent1[0] += l
            #arg2_sent1[1] += l

            sents_concat = sent1 + " ***** " + sent2 #sents_concat.split(" ")[l] is the first token in the 2nd sent
            #create idx tensor. # 1stdim: sents, 2st dim: arg, 3st dim: start and end
            idx = [[[arg1_sent1[0], arg1_sent1[1]], [arg2_sent1[0], arg2_sent1[1]]], [[0, 1], [0, 1]] ] 
            sent2_with_args = sent2
            return sents_concat, torch.tensor(idx).int(), l, sent2_with_args
        
        
def evaluate_model(sents1, sents2, arg1_sent1, arg2_sent1, model, max_ngrams = 5, num_examples = 200):
    
    preds = []
    count = 0
    
    for i in range(len(sents1)):
               
        sents_concat, idx, l, sent2_with_args = prepare_example(sents1[i], sents2[i], arg1_sent1[i], arg2_sent1[i])
        
        
        idx = idx.detach().cpu().numpy()
        bert_tokens, orig_to_tok_map, tok_to_orig_map, tokens_tensor = model.tokenize(sents_concat.split(" "))
        l_tokens = len(bert_tokens[:orig_to_tok_map[l-1]])
        
        
        sent1_range_arg1 = get_entity_range_multiword_expression(idx[0][0], orig_to_tok_map)
        sent1_range_arg2 = get_entity_range_multiword_expression(idx[0][1], orig_to_tok_map)
        sent2_range_arg1 = get_entity_range_multiword_expression(idx[1][0], orig_to_tok_map)
        sent2_range_arg2 = get_entity_range_multiword_expression(idx[1][1], orig_to_tok_map)

        range_sent1 = (sent1_range_arg1,sent1_range_arg2)
        range_sent2 = (sent2_range_arg1,sent2_range_arg2)
        
        with torch.no_grad():
            loss, idx_arg1, idx_arg2, idx_arg1_all, idx_arg2_all, all_false_ngrams_ranges, all_ngrams = model.forward_with_loss_calculation(bert_tokens, tokens_tensor, range_sent1, range_sent2, orig_to_tok_map, l, l_tokens, mode = "eval", n_max=max_ngrams)
        preds.append({"sent": sents_concat, "tokens": bert_tokens, "tok2orig": tok_to_orig_map, "orig2tok": orig_to_tok_map,
                     "preds_arg1_tokens": idx_arg1_all, "preds_arg2_tokens": idx_arg2_all, "false_ngrams": all_false_ngrams_ranges,
                      "all_ngrams": all_ngrams, "gold_arg1_range_tokens": sent2_range_arg1, "gold_arg2_range_tokens": sent2_range_arg2, "l": l})
        
    return preds


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
    
    query_sents = spike_df["sentence_text"].tolist()
    query_arg1_starts = spike_df["arg1_first_index"]
    query_arg1_ends = spike_df["arg1_last_index"]
    query_arg2_starts = spike_df["arg2_first_index"]
    query_arg2_ends = spike_df["arg2_last_index"]
    
    query_used = query_sents[0] # use first query in all examples.
    query_used_arg1 = [query_arg1_starts[0], query_arg1_ends[0]]
    query_used_arg2 = [query_arg2_starts[0], query_arg2_ends[0]]
    
    sents2 = results_sents
    sents1 = [query_used] * len(sents2)
    query_used_arg1 = [query_used_arg1] * len(sents2)
    query_used_arg2 = [query_used_arg2] * len(sents2)
    
    results = evaluate_model(sents1, sents2, query_used_arg1, query_used_arg2, model, max_ngrams = max_ngrams, num_examples = len(sents1))
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
        captures_tuples.append("ARG1: {}; ARG2: {}".format(arg1_str, arg2_str))
        annotated_sent = perform_annotation(sent, [[arg1_start, arg1_end], [arg2_start, arg2_end]])
        annotated_sent = annotated_sent[p["l"]:]
        annotated.append(annotated_sent)
    
    # aggregate arguments
    args1, args2 = list(zip(*captures))
    arg1_counter, arg2_counter, tuples_counter = Counter(args1), Counter(args2), Counter(captures_tuples)
    
    return annotated, arg1_counter.most_common(500), arg2_counter.most_common(500), tuples_counter.most_common(500)
