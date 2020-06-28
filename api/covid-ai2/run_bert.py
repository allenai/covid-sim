import torch
from transformers import BertTokenizer
import csv
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, AutoTokenizer, AutoModel
import time
import datetime
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
import sys
import argparse
import os
from collections import defaultdict
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from typing import List, Dict
from torch.utils.data import Dataset
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from typing import List
import tqdm
import csv
import json
import faiss  
import argparse

    
    
class BertEncoder(object):
    
    def __init__(self, device = 'cpu'):
        
        #self.tokenizer = BertTokenizer.from_pretrained('scibert_scivocab_uncased/vocab.txt')
        #self.model = BertModel.from_pretrained('scibert_scivocab_uncased/')
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        
    def tokenize_and_pad(self, texts: List[str]):
        
        indexed_texts = [self.tokenizer.encode(text, add_special_tokens=True, max_length = 512) for text in texts] #
        max_len = max(len(text) for text in indexed_texts)
        indexed_texts = [text + [self.pad_token] * (max_len - len(text)) for text in indexed_texts]
        idx_tensor = torch.LongTensor(indexed_texts).to(self.device)
        att_tensor = idx_tensor != self.pad_token
        
        return idx_tensor, att_tensor
    
    def encode(self, sentences: List[str], sentence_ids: List[str], batch_size: int, strategy: str = "cls", fname=""):
        assert len(sentences) == len(sentence_ids)
        
        with open(fname, "w", encoding = "utf-8") as f:
            
            for batch_idx in tqdm.tqdm(range(0, len(sentences), batch_size), total = len(sentences)//batch_size):
            
                batch_sents = sentences[batch_idx: batch_idx + batch_size]
                batch_ids = sentence_ids[batch_idx: batch_idx + batch_size]
                assert len(batch_sents) == len(batch_ids)
                
                idx, att_mask = self.tokenize_and_pad(batch_sents)
            
                with torch.no_grad():
                    outputs = self.model(idx, attention_mask = att_mask)
                    last_hidden = outputs[0]
                
                    if strategy == "cls":
                        h = last_hidden[:, 0, ...]
                    elif strategy == "mean-cls":
                        h = torch.cat([last_hidden[:, 0, ...], torch.mean(last_hidden, axis = 1)], axis = 1)
                    elif strategy == "mean-cls-max":
                       h_max = torch.max(last_hidden, axis = 1).values
                       h = torch.cat([last_hidden[:, 0, ...], torch.mean(last_hidden, axis = 1), h_max], axis = 1)
                    elif strategy == "mean":
                        h = torch.mean(last_hidden, axis = 1)
                    elif strategy == "median":
                        h = torch.median(last_hidden, axis = 1).values
                    elif strategy == "max":
                        h = torch.max(last_hidden, axis = 1).values
                    elif strategy == "min":
                        h = torch.min(last_hidden, axis = 1).values
            
                batch_np = h.detach().cpu().numpy()
                assert len(batch_np) == len(batch_sents)
                
                sents_states_ids = zip(batch_sents, batch_np, batch_ids)
                for sent, vec, sent_id in sents_states_ids:
                    
                    vec_str = " ".join(["%.4f" % x for x in vec])
                    sent_dict = {"text": sent, "vec": vec_str, "id": sent_id}
                    f.write(json.dumps(sent_dict) + "\n")
 
if __name__ == "__main__":
 


        parser = argparse.ArgumentParser(description='collect bert states over sentences',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--input-filename', dest='input_filename', type=str,
                        default="results.tsv")
            
        parser.add_argument('--pooling', dest='pooling', type=str,
                        default="cls")
        parser.add_argument('--output_fname', dest='output_fname', type=str,
                        default="output-cls.jsonl")
        parser.add_argument('--device', dest='device', type=str,
                        default="cpu")  
        args = parser.parse_args()
        
        df = pd.read_csv(args.input_filename, sep = "\t")
        ids, sents = df["sentence_id"].tolist(), df["sentence_text"].tolist()
        
        encoder = BertEncoder(args.device)
        encoder.encode(sents, ids, batch_size = 32, strategy = args.pooling, fname=args.output_fname)
