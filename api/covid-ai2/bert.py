import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertModel, AutoTokenizer, AutoModel
import numpy as np
from typing import List
import tqdm

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
    
    def encode(self, sentences: List[str], sentence_ids: List[str], batch_size: int, strategy: str = "cls", fname="", write = False):
        assert len(sentences) == len(sentence_ids)
        vecs = []
        
        with open(fname, "w", encoding = "utf-8") as f:
            
            for batch_idx in tqdm.tqdm_notebook(range(0, len(sentences), batch_size), total = len(sentences)//batch_size):
            
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
                    if write:
                        f.write(json.dumps(sent_dict) + "\n")
                    else:
                        vecs.append(vec)
        return np.array(vecs)
