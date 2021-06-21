import torch
from typing import List, Dict
import random
import numpy as np

class Dataset(torch.utils.data.Dataset):
    """Simple torch dataset class"""

    def __init__(self, data: List[Dict], device = "cpu", negative_prob = 0.0):

        self.data = data
        self.device = device
        self.negative_prob = negative_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with torch.no_grad():
            
            d = self.data[index]
            other = random.choice(range(len(self)))

            if random.random() < (1-self.negative_prob) or (self.data[other]["query_id"] == d["query_id"]):
                sent1, sent2 = d["first"], d["second"]
                id1, id2 = 1,1
                is_negative = False
            else:
                sent1 = d["first"]
                sent2 = self.data[other]["second"] 
                id1, id2 = 1,1
                is_negative = True
                
            sent1_arg1, sent1_arg2 = list(d["first_arg1"]), list(d["first_arg2"])
            sent2_arg1, sent2_arg2 = list(d["second_arg1"]), list(d["second_arg2"])            
            l = len(sent1.split(" ")) + 1 
            sent2_arg1[0] += l
            sent2_arg1[1] += l
            sent2_arg2[0] += l
            sent2_arg2[1] += l

            sent2 = sent2.replace("ARG1:", "").replace("ARG2:", "").replace("<<","").replace(">>","")
            sents_concat = sent1 + " ***** " + sent2 #sents_concat.split(" ")[l] is the first token in the 2nd sent
            #create idx tensor. # 1stdim: sents, 2st dim: arg, 3st dim: start and end
            idx = [[[sent1_arg1[0], sent1_arg1[1]], [sent1_arg2[0], sent1_arg2[1]]], [[sent2_arg1[0], sent2_arg1[1]], [sent2_arg2[0], sent2_arg2[1]]] ] 
            sent2_with_args = sent2
            return sents_concat, torch.tensor(idx).int(), l, sent2_with_args, is_negative, id1, id2
