
import pandas as pd
import tqdm
import pickle
import random
import itertools

import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, AutoTokenizer, AutoModel, PreTrainedTokenizerFast
import numpy as np
from typing import List
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Dict, Tuple
from scipy.spatial.distance import cosine as cosine_distance
from collections import defaultdict
from nltk import ngrams as get_ngrams
from termcolor import colored
from torch.optim.lr_scheduler import ReduceLROnPlateau 



class BertModel(pl.LightningModule):

    def __init__(self, train_dataset: Dataset, dev_dataset: Dataset, batch_size, device: str, mode: str = "eval", alpha=0.1, lr = 1e-4, momentum=0.5, l2_loss = False, same_rel_weight = 1, pretrained = True, train_only_linear=True):
        
        super().__init__()
        
        self.device_to_use = device
        
        if not pretrained:
            config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=config) 
        else:
            print("loading pretrained model")
            config = AutoConfig.from_pretrained('Shauli/RE-metric-model-spike', output_hidden_states=True)
            self.model = AutoModel.from_pretrained('Shauli/RE-metric-model-spike', config=config)    
            self.tokenizer = AutoTokenizer.from_pretrained('Shauli/RE-metric-model-spike')          
        
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.linear_arg1_1 = torch.nn.Linear(768, 128) #torch.load("finetuned_model/metric_model/linear.pt") #torch.nn.Linear(768, 64)
        
        if pretrained:
            self.linear_arg1_1.load_state_dict(torch.load("linear.pt", map_location = torch.device('cpu')))
        self.same_rel_mlp = torch.nn.Sequential(*[torch.nn.Linear(768, 1)])#, torch.nn.ReLU(), torch.nn.Linear(128, 1)])     
        #if pretrained: 
        #    self.same_rel_mlp.load_state_dict(torch.load("finetuned_model/metric_model/same_rel_mlp.pt")) 
                 
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.lr = lr
        self.train_only_linear = train_only_linear
        self.l2_loss = l2_loss
        self.momentum = momentum
        self.same_rel_weight = same_rel_weight
        
        if mode == "eval":
            
            self.model.eval()
        else:
            self.model.train()
        
        for p in self.model.parameters():
            p.requires_grad = True
            #if len(p.shape) == 1:
            #    p.requires_grad = True
            
        for p in self.model.embeddings.parameters():
            p.requires_grad = True
        #for p in self.model.encoder.layer[-1].parameters():
        #    p.requires_grad = True
        #for p in self.model.encoder.layer[-2].parameters():
        #    p.requires_grad = True       
        #for p in self.model.encoder.layer[-3].parameters():
        #    p.requires_grad = True    
            
        self.train_gen = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, drop_last=False, shuffle=True,
                                                    num_workers = 4)
        self.dev_gen = torch.utils.data.DataLoader(self.dev_dataset, batch_size=batch_size, drop_last=False, shuffle=False,
                                                  num_workers = 4)
        self.acc = None
        self.total = 0
        self.total_same_rel = 0
        self.count_same_rel = 0
        self.count = 0
        
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
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device_to_use)

        return (bert_tokens, orig_to_tok_map, tok_to_orig_map, tokens_tensor)
    
    
    def forward(self, x):
        
        outputs = self.model(x)
        states = outputs[0][0] #[seq_len, 768]
        return states
    

    def forward_with_loss_calculation(self, bert_tokens, x, range_sent1, range_sent2, orig_to_tok_map, l, l_tokens,
                                      metric = "l2", n_max = 9, mode = "train", normalize=False, nb=0):
        idx_arg1_all, idx_arg2_all, all_ngrams = None, None, None
        
        if self.train_only_linear:
          with torch.no_grad():
            outputs = self.model(x)
        else:
            outputs = self.model(x)
        states = outputs[0][0] #[seq_len, 768]
        if not self.l2_loss or normalize:
            states = states / (torch.norm(states, dim = 1, keepdim = True)+1e-8)
        
        is_neg_pred = self.same_rel_mlp(states[0])
        
        states = self.linear_arg1_1(states)
        arg1_sent1, arg2_sent1 = range_sent1
        arg1_sent2, arg2_sent2 = range_sent2
        
        sent1_arg1_vec, sent1_arg2_vec = states[arg1_sent1[0]:arg1_sent1[1]].mean(dim=0), states[arg2_sent1[0]:arg2_sent1[1]].mean(dim=0)
        sent2_arg1_vec, sent2_arg2_vec = states[arg1_sent2[0]:arg1_sent2[1]].mean(dim=0), states[arg2_sent2[0]:arg2_sent2[1]].mean(dim=0)        
        
        all_false_ngrams_ranges = get_all_ngrams_spans(len(states), [arg1_sent1, arg1_sent2, arg2_sent1, arg2_sent2], start_ind = 0,
                                                      n_max = n_max)      
        negatives = [states[ngram[0]:ngram[1]].mean(dim=0) for ngram in all_false_ngrams_ranges]
        negatives_arg1 = negatives + [sent1_arg2_vec, sent2_arg2_vec]
        negatives_arg2 = negatives + [sent1_arg1_vec, sent2_arg1_vec]
        negatives_arg1 = torch.stack(negatives_arg1).to(self.device_to_use)
        negatives_arg2 = torch.stack(negatives_arg2).to(self.device_to_use)
        

        if mode == "eval":
            all_ngrams = get_all_ngrams_spans(len(states), [], start_ind = l_tokens,
                                                      n_max = n_max)
            ngrams = [states[ngram[0]:ngram[1]].mean(dim=0) for ngram in all_ngrams]
            ngrams = torch.stack(ngrams).to(self.device_to_use)
        
        
        if self.l2_loss:
            dists_arg1 = torch.sqrt(((negatives_arg1-sent1_arg1_vec)**2).sum(dim = 1))
            dists_arg2 = torch.sqrt(((negatives_arg2-sent1_arg2_vec)**2).sum(dim = 1))
            dist_arg1_gold = (sent1_arg1_vec - sent2_arg1_vec).norm()
            dist_arg2_gold = (sent1_arg2_vec - sent2_arg2_vec).norm()
            if mode == "eval":
                dist_arg1_all = torch.sqrt(((ngrams-sent1_arg1_vec)**2).sum(dim = 1))
                dist_arg2_all = torch.sqrt(((ngrams-sent1_arg2_vec)**2).sum(dim = 1))
                idx_arg1_all = torch.argsort(dist_arg1_all).detach().cpu().numpy()
                idx_arg2_all = torch.argsort(dist_arg2_all).detach().cpu().numpy()   
                
        else:
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


        if self.l2_loss:
            loss_arg1 = torch.max(torch.zeros(1).to(self.device_to_use), dist_arg1_gold - dist_arg1_argmax + self.alpha)
            loss_arg2 = torch.max(torch.zeros(1).to(self.device_to_use), dist_arg2_gold - dist_arg2_argmax + self.alpha)     
        # softmax triplet
        else:
            z = torch.max(dist_arg1_argmax, dist_arg1_gold)
            temp = 1
            pos_arg1 = torch.exp((dist_arg1_gold - z)/temp)
            neg_arg1 = torch.exp((dist_arg1_argmax - z)/temp)
            loss_arg1 = (pos_arg1 / (pos_arg1 + neg_arg1))#**2

            z = torch.max(dist_arg2_argmax, dist_arg2_gold)
            pos_arg2 = torch.exp((dist_arg2_gold - z)/temp)
            neg_arg2 = torch.exp((dist_arg2_argmax - z)/temp)
            loss_arg2 = (pos_arg2 / (pos_arg2 + neg_arg2))#**2

        
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

        self.total += 1                                                      
        #if loss.detach().cpu().numpy().item() < 1e-5:
        if (dist_arg1_gold - dist_arg1_argmax).detach().cpu().numpy().item() < 0 and (dist_arg2_gold - dist_arg2_argmax).detach().cpu().numpy().item() < 0:
            self.count += 1
        
        return loss, idx_arg1, idx_arg2, idx_arg1_all, idx_arg2_all, all_false_ngrams_ranges, all_ngrams, is_neg_pred
        #return loss, np.argsort(dists_arg1+mask_gold_arg1)
        
    def training_step(self, batch, batch_nb):
        
        sents_concat, idx, l, sent2_with_args, is_negative, id1, id2 = batch
        idx = idx.detach().cpu().numpy()[0]
        bert_tokens, orig_to_tok_map, tok_to_orig_map, tokens_tensor = self.tokenize(sents_concat[0].split(" "))             
        self.total_same_rel += 1
        
        if not is_negative:
            l_tokens = len(bert_tokens[:orig_to_tok_map[l.detach().cpu().numpy().item()-1]]) 
            sent1_range_arg1 = get_entity_range_multiword_expression(idx[0][0], orig_to_tok_map)
            sent1_range_arg2 = get_entity_range_multiword_expression(idx[0][1], orig_to_tok_map)
            sent2_range_arg1 = get_entity_range_multiword_expression(idx[1][0], orig_to_tok_map)
            sent2_range_arg2 = get_entity_range_multiword_expression(idx[1][1], orig_to_tok_map)
            range_sent1 = [sent1_range_arg1, sent1_range_arg2]
            range_sent2 = [sent2_range_arg1, sent2_range_arg2]
            
            
            loss, _, _, _, _, _, _, is_neg_pred = self.forward_with_loss_calculation(bert_tokens, tokens_tensor, range_sent1, range_sent2, orig_to_tok_map, l, l_tokens, nb = batch_nb)
        else:
            loss = torch.zeros(1).to(self.device)       
            outputs = self.model(tokens_tensor)
            states = outputs[0][0]
            is_neg_pred = self.same_rel_mlp(states[0])
        
        if (is_negative and is_neg_pred.detach().cpu().numpy().item() > 0) or ((not is_negative) and (is_neg_pred.detach().cpu().numpy().item() < 0)):
            self.count_same_rel += 1
            
        y = torch.ones(1).to(self.device) if is_negative else torch.zeros(1).to(self.device)   
        loss += self.same_rel_weight * self.bce_loss(is_neg_pred, y)
        #loss = self.same_rel_weight * self.bce_loss(is_neg_pred, y)
        
#         if np.isnan(loss.detach().cpu().numpy().item()) or loss.detach().cpu().numpy().item() > 1e4:
#             print("ERRROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#             print(sents_concat, range_sent1, range_sent2, sent1_idx, sent2_idx)
#             return {"loss": loss*0}

        if self.total%500 == 0 and self.total > 1:
            self.log('train_loss_1k', self.count/self.total)
            self.log("train_loss_1k_same_rel", self.count_same_rel/self.total_same_rel)
            print("argument identification accuracy", self.count/self.total)
            print("same-relation identification accuracy", self.count_same_rel/self.total_same_rel)
            self.count = 0
            self.count_same_rel = 0
            self.total = 0
            self.total_same_rel = 0
            
        return {'loss': loss}
    
    """
    def validation_step(self, batch, batch_nb):

        sents_concat, idx, l, sent2_with_args, is_negative, id1, id2 = batch
        print(is_negative)
        if is_negative:
            return {'val_loss': torch.zeros(1).to(self.device)}
        
        idx = idx.detach().cpu().numpy()[0]
        bert_tokens, orig_to_tok_map, tok_to_orig_map, tokens_tensor = self.tokenize(sents_concat[0].split(" "))
        l_tokens = len(bert_tokens[:orig_to_tok_map[l.detach().cpu().numpy().item()-1]]) 
        sent1_range_arg1 = get_entity_range_multiword_expression(idx[0][0], orig_to_tok_map)
        sent1_range_arg2 = get_entity_range_multiword_expression(idx[0][1], orig_to_tok_map)
        sent2_range_arg1 = get_entity_range_multiword_expression(idx[1][0], orig_to_tok_map)
        sent2_range_arg2 = get_entity_range_multiword_expression(idx[1][1], orig_to_tok_map)
        
        range_sent1 = [sent1_range_arg1,sent1_range_arg2]
        range_sent2 = [sent2_range_arg1,sent2_range_arg2]
        loss, _, _, _, _, _, _, is_neg_pred = self.forward_with_loss_calculation(bert_tokens, tokens_tensor, range_sent1, range_sent2, orig_to_tok_map, l, l_tokens)


        return {'val_loss': loss}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print("Loss is {}".format(avg_loss))
        return {'avg_val_loss': avg_loss}
    """
    
    def configure_optimizers(self):
        #return torch.optim.RMSprop(self.parameters())
        #return torch.optim.ASGD(self.parameters())
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        return {"optimizer": optimizer, 'scheduler': ReduceLROnPlateau(optimizer, patience = 1, factor = 0.5, verbose = True), 'monitor': 'train_loss_1k'}
        #return torch.optim.Adam(self.parameters())

    

def evaluate_model(dev_dataset, model, max_ngrams = 2, num_examples = 200):
    
    preds = []
    count = 0
    
    for batch in tqdm.tqdm(dev_dataset):
        if count > num_examples: break
        count += 1

        sents_concat, idx, l, sent2_with_args, is_negative, id1, id2 = batch
        if is_negative: continue
        idx = idx.detach().cpu().numpy()
        bert_tokens, orig_to_tok_map, tok_to_orig_map, tokens_tensor = model.tokenize(sents_concat.split(" "))
        l_tokens = len(bert_tokens[:orig_to_tok_map[l-1]]) 
        sent1_range_arg1 = get_entity_range_multiword_expression(idx[0][0], orig_to_tok_map)
        sent1_range_arg2 = get_entity_range_multiword_expression(idx[0][1], orig_to_tok_map)
        sent2_range_arg1 = get_entity_range_multiword_expression(idx[1][0], orig_to_tok_map)
        sent2_range_arg2 = get_entity_range_multiword_expression(idx[1][1], orig_to_tok_map)

        range_sent1 = (sent1_range_arg1,sent1_range_arg2)
        range_sent2 = (sent2_range_arg1,sent2_range_arg2)
        
        loss, idx_arg1, idx_arg2, idx_arg1_all, idx_arg2_all, all_false_ngrams_ranges, all_ngrams, is_neg_pred = model.forward_with_loss_calculation(bert_tokens, tokens_tensor, range_sent1, range_sent2, orig_to_tok_map, l, l_tokens, mode = "eval", n_max=max_ngrams)
        is_neg_pred = torch.sigmoid(is_neg_pred).detach().cpu().numpy().item()
        same_relation_score = 1 - is_neg_pred
        preds.append({"sent": sents_concat, "tokens": bert_tokens, "tok2orig": tok_to_orig_map, "orig2tok": orig_to_tok_map,
                     "preds_arg1_tokens": idx_arg1_all, "preds_arg2_tokens": idx_arg2_all, "false_ngrams": all_false_ngrams_ranges,
                      "all_ngrams": all_ngrams, "gold_arg1_range_tokens": sent2_range_arg1, "gold_arg2_range_tokens": sent2_range_arg2, "same_rel_pred": same_relation_score,
                     "is_negative": is_negative, "id1": id1, "id2": id2})
        
    return preds


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
    
 
