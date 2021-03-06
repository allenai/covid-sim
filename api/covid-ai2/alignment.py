
#import bert
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#import matplotlib.pyplot as plt
#import spike_queries
#from termcolor import colored
#import random
from collections import Counter, defaultdict
#from viterbi_trellis import ViterbiTrellis
import streamlit as st
from annot import annotation
import re



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



def get_spike_results_arguments_representations(model, spike_results, layers, num_args):
    sents = spike_results["sentence_text"].tolist()
    #arg1_idx_start = spike_results["arg1_first_index"].to_numpy().astype(int)
    #arg2_idx_start = spike_results["arg2_first_index"].to_numpy().astype(int)
    #arg1_idx_end = spike_results["arg1_last_index"].to_numpy().astype(int)
    #arg2_idx_end = spike_results["arg2_last_index"].to_numpy().astype(int)

    arguments_borders = []
    for i in range(num_args):
        start = spike_results["arg{}_first_index".format(i + 1)].to_numpy().astype(int)
        end = spike_results["arg{}_last_index".format(i + 1)].to_numpy().astype(int)
        arguments_borders.append((start, end))

    args_rep = defaultdict(list)

    for i,s in enumerate(sents):

        if not type(s) == str: continue

        H, _, _, orig2tok = model.encode(s, layers=layers)

        for arg_ind in range(num_args):
            start, end = arguments_borders[arg_ind][0][i], arguments_borders[arg_ind][1][i]
            arg_vecs = H[orig2tok[start]:orig2tok[end] + 1]
            arg_mean = np.mean(arg_vecs, axis = 0)
            args_rep[arg_ind].append(arg_mean)

    return [np.mean(args_rep[arg], axis = 0) for arg in range(num_args)]


def get_similarity_to_arguments(padded_representations, args_reps):
    num_sents, seq_len, bert_dim = padded_representations.shape
    padded_representations = padded_representations.reshape((num_sents*seq_len, bert_dim))
    #print(padded_representations.shape)
    sims = cosine_similarity(args_reps, padded_representations)
    sims = sims.reshape((len(args_reps), num_sents, seq_len))
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
    :param sims_args: similarity to arguments per query, shape: [num_args,num_sents,padded_sent_len]
    :return: 
    """

    for arg in range(sims_args.shape[0]):
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

def main(model, results_sents, spike_results, spike_query, layers, num_results):
    arg2preds = {}

    # count args
    regex = re.compile("arg.:")
    num_args = len(re.findall(regex, spike_query))

    # represent args

    args_reps = get_spike_results_arguments_representations(model, spike_results.head(num_results), layers, num_args)

    representations = []
    mappings_to_orig = []
    mappings_to_tok = []
    tokenized_txts = []
    orig_sents = []

    for i, s in enumerate(results_sents):
        if not type(s) == str: continue

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
    sims_args = get_similarity_to_arguments(padded_representations, args_reps)
    arguments2sent2alignments = get_probable_alignments(sims_args, mappings_to_orig)
    for arg in range(num_args):
        dicts = [{"sent": orig_sents[i], "pred_idx": list(zip(*arguments2sent2alignments[arg][i]))[0],
                  "preds_sims": list(zip(*arguments2sent2alignments[arg][i]))[1]} for i in range(num_sents)]

        arg2preds[arg] = dicts

    colored_sents = []
    annotated_sents = []

    for i in range(num_sents):
        #arg1_dict, arg2_dict = arg2preds[0][i], arg2preds[1][i]
        arg_dicts = [arg2preds[j][i] for j in range(num_args)]
        sent = arg_dicts[0]["sent"]
        #arg1_idx, arg2_idx = arg1_dict["pred_idx"][0], arg2_dict["pred_idx"][0]
        arg_idx = [arg_dict["pred_idx"][0] for arg_dict in arg_dicts]
        #colored_sent = print_nicely(sent, [(arg1_idx, arg1_idx+1)], [(arg2_idx, arg2_idx+1)])
        #annotated_sents.append(perform_annotation(sent, [(arg1_idx, arg1_idx+1)], [(arg2_idx, arg2_idx+1)]))

        borders = [(k,k+1) for k in arg_idx]
        colored_sent = None#print_nicely(sent, borders)
        annotated_sents.append(perform_annotation(sent, borders))

        colored_sents.append(colored_sent)

    return colored_sents, annotated_sents