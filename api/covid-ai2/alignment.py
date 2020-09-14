
#import bert
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
#import matplotlib.pyplot as plt
#import spike_queries
#from termcolor import colored
#import random
#from collections import Counter, defaultdict
#from viterbi_trellis import ViterbiTrellis



def get_spike_results_arguments_representations(model, spike_results, layers):
    sents = spike_results["sentence_text"].tolist()
    arg1_idx_start = spike_results["arg1_first_index"].to_numpy().astype(int)
    arg2_idx_start = spike_results["arg2_first_index"].to_numpy().astype(int)
    arg1_idx_end = spike_results["arg1_last_index"].to_numpy().astype(int)
    arg2_idx_end = spike_results["arg2_last_index"].to_numpy().astype(int)

    arg1_rep = []
    arg2_rep = []

    for s, arg1_start, arg2_start, arg1_end, arg2_end in zip(sents, arg1_idx_start, arg2_idx_start, arg1_idx_end,
                                                             arg2_idx_end):
        # idx_to_mask = [arg1_start, arg2_start, arg1_end, arg2_end]
        H, _, _, orig2tok = model.encode(s, layers=layers)

        h1, h2 = H[orig2tok[arg1_start]:orig2tok[arg1_end] + 1], H[orig2tok[arg2_start]:orig2tok[arg2_end] + 1]

        h1 = np.mean(h1, axis=0)
        h2 = np.mean(h2, axis=0)

        arg1_rep.append(h1)
        arg2_rep.append(h2)

    arg1_mean = np.mean(arg1_rep, axis=0)
    arg2_mean = np.mean(arg2_rep, axis=0)

    return arg1_mean, arg2_mean