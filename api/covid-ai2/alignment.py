
#import bert
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#import matplotlib.pyplot as plt
#import spike_queries
#from termcolor import colored
#import random
#from collections import Counter, defaultdict
#from viterbi_trellis import ViterbiTrellis
import streamlit as st
from annot import annotation




class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'



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


def get_similarity_to_arguments(padded_representations, arg1_rep, arg2_rep):
    num_sents, seq_len, bert_dim = padded_representations.shape
    padded_representations = padded_representations.reshape((num_sents*seq_len, bert_dim))
    #print(padded_representations.shape)
    sims = cosine_similarity([arg1_rep, arg2_rep], padded_representations)
    sims = sims.reshape((2, num_sents, seq_len))
    return sims

def pad(representations):
    for i in range(len(representations)):  # zero cls, ., sep
        representations[i][0][:] = np.random.rand()
        representations[i][-1][:] = np.random.rand()
        representations[i][-2][:] = np.random.rand()

    pad_width = max([len(s) for s in representations])
    padded_representations = np.array(
        [np.concatenate([r, -np.ones((pad_width - len(r), 768))]) for r in representations])
    return padded_representations


def get_probable_alignments(sims_args, mappings_to_orig):
    argument2sent2alignments = dict()

    """
    :param sims_args: similarity to arguments per query, shape: [2,num_sents,padded_sent_len]
    :return: 
    """

    for arg in range(2):
        sent2alignments = dict()
        for sent_ind in range(sims_args.shape[1]):
            sorted_sims_idx = np.argsort(-sims_args[arg][sent_ind])
            sorted_sims_vals = sims_args[arg][sent_ind][sorted_sims_idx]
            sorted_sims_idx_mapped = [mappings_to_orig[sent_ind][j] if j in mappings_to_orig[sent_ind] else -1 for j in
                                      sorted_sims_idx]
            sent2alignments[sent_ind] = list(zip(sorted_sims_idx_mapped, sorted_sims_vals))

        argument2sent2alignments[arg] = sent2alignments

    return argument2sent2alignments


def print_nicely(sent, arg1_borders, arg2_borders):
    def is_start(k, borders):
        return len([(s, e) for (s, e) in borders if s == k]) != 0

    def is_end(k, borders):
        return len([(s, e) for (s, e) in borders if e == k]) != 0

    sent_lst = sent.split(" ")
    sent_new = []
    for i, w in enumerate(sent_lst):

        if is_start(i, arg1_borders) or is_start(i, arg2_borders):
            type_arg = color.BLUE + "ARG1" if is_start(i, arg1_borders) else color.BLUE + "ARG2"
            sent_new.append(color.BOLD + "[" + type_arg)

        sent_new.append(w)

        if is_end(i, arg1_borders) or is_end(i, arg2_borders):
            # type_arg = color.BLUE + "ARG1" if is_end(i,arg1_borders) else "ARG2" + color.END
            sent_new.append("]" + color.END)

    return " ".join(sent_new)

def perform_annotation(sent, arg1_borders, arg2_borders):

    def is_between(k, borders):
        return len([(s, e) for (s, e) in borders if s <= k < e]) != 0

    sent_lst = sent.split(" ")
    sent_new = []
    arg1_color = "#8ef"
    arg2_color = "#fea"
    for i, w in enumerate(sent_lst):

        if is_between(i, arg1_borders) or is_between(i, arg2_borders):
            is_arg1 = is_between(i, arg1_borders)
            sent_new.append((w, "ARG1" if is_arg1 else "ARG2", arg1_color if is_arg1 else arg2_color))
        else:

            sent_new.append(" " + w)

    return sent_new

def main(model, results_sents, spike_results, layers, num_results):
    arg2preds = {}

    arg1_rep, arg2_rep = get_spike_results_arguments_representations(model, spike_results.head(num_results), layers)

    representations = []
    mappings_to_orig = []
    mappings_to_tok = []
    tokenized_txts = []
    orig_sents = []

    for i, s in enumerate(results_sents):
        H, tokenized_text, tok_to_orig_map, orig2tok = model.encode(s, layers=layers)
        orig_sents.append(s)
        representations.append(H)
        mappings_to_orig.append(tok_to_orig_map)
        mappings_to_tok.append(orig2tok)
        tokenized_txts.append(tokenized_text)

        if i > num_results: break

    #return (arg1_rep, arg2_rep), (representations, mappings_to_orig, mappings_to_tok, tokenized_txts, orig_sents)
    padded_representations = pad(representations)
    num_sents, seq_len, bert_dim = padded_representations.shape
    num_tokens = num_sents * seq_len
    sims_args = get_similarity_to_arguments(padded_representations, arg1_rep, arg2_rep)
    arguments2sent2alignments = get_probable_alignments(sims_args, mappings_to_orig)
    for arg in range(2):
        dicts = [{"sent": orig_sents[i], "pred_idx": list(zip(*arguments2sent2alignments[arg][i]))[0],
                  "preds_sims": list(zip(*arguments2sent2alignments[arg][i]))[1]} for i in range(num_sents)]

        arg2preds[arg] = dicts

    colored_sents = []
    annotated_sents = []

    for i in range(num_sents):
        arg1_dict, arg2_dict = arg2preds[0][i], arg2preds[1][i]
        sent = arg1_dict["sent"]
        arg1_idx, arg2_idx = arg1_dict["pred_idx"][0], arg2_dict["pred_idx"][0]
        colored_sent = print_nicely(sent, [(arg1_idx, arg1_idx+1)], [(arg2_idx, arg2_idx+1)])
        annotated_sents.append(perform_annotation(sent, [(arg1_idx, arg1_idx+1)], [(arg2_idx, arg2_idx+1)]))
        colored_sents.append(colored_sent)

    return colored_sents, annotated_sents