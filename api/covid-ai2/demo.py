import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import faiss
import bert
from bert import BertEncoder
import pickle
import spike_queries
import sklearn
import random
import time
import alignment
import bert_all_seq
#import alignment_supervised2 as alignment_supervised
import alignment_supervised
from annot import annotation, annotated_text
import time
import SessionState
NUM_RESULTS_TO_ALIGN_DEFAULT = 200
DEFAULT_MAX_NGRAM = 5
BOOLEAN_QUERY_DEFAULT = "virus lemma=originate"
TOKEN_QUERY_DEFAULT = "novel coronavirus"
SYNTACTIC_QUERY_DEFAULT = "<>arg1:[e=CHEMICAL|SIMPLE_CHEMICAL]paracetamol $[lemma=word|act]works $by <>arg2:activation of something" #"<>arg1:[e]paracetamol is the recommended $treatment for <>arg2:[e]asthma."
SPIKE_RESULTS_DEFAULT = 75
must_include = ""

st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def load_sents_and_ids():
    with st.spinner('Loading sentences and IDs...'):
        #df = pd.read_csv("data/results.tsv", sep = "\t")
        #sents =  df["sentence_text"].tolist()
        with open("data/sents.txt", "r", encoding = "utf-8") as f:
            sents = f.readlines()
            sents = [s.strip() for s in sents]
        
        ids = [hash(s) for s in sents]

        id2ind = {ids[i]:i for i,s in enumerate(sents)}
        ind2id = {i:ids[i] for i,s in enumerate(sents)}

        return sents, ids, id2ind, ind2id

@st.cache(allow_output_mutation=True)
def load_index(similarity, pooling):
    with st.spinner('Loading FAISS index...'):
        fname = "data/output-" + pooling + ".index"
        index = faiss.read_index(fname)
        return index

@st.cache(allow_output_mutation=True)
def load_bert():
    with st.spinner('Loading BERT...'):
        model = bert.BertEncoder("cpu")
        return model

@st.cache(allow_output_mutation=True)
def load_bert_all_seq():
    with st.spinner('Loading BERT...'):
        model = bert_all_seq.BertEncoder("cpu")
        return model
    
@st.cache(allow_output_mutation=True)
def load_bert_alignment_supervised():
    with st.spinner('Loading BERT...'):
        model = alignment_supervised.BertModel("cpu")
        return model    
    
@st.cache(allow_output_mutation=True)        
def load_pca(pooling):

    fname = "data/output-" + pooling + ".pca.pickle"
    with open(fname, "rb") as f:
    
        return pickle.load(f)

@st.cache(allow_output_mutation=True)        
def encode(input_sentence, pca, bert, pooling):
    return pca.transform(bert.encode([input_sentence], [1], batch_size = 1, strategy = pooling, fname = "dummy.txt", write = False))

def zero_input():
    
    input_sentence = placeholder.text_input('Enter a sentence for similarity search', value="", key = random.randint(0,int(1e16)))

    
def write_results_menu(results, session_state, keys="random"):
    
    cols = st.beta_columns((8,1,1))
    cols[0].markdown("<b>Sentence</b>", unsafe_allow_html = True)
    cols[1].markdown("<b>Enhance?</b>", unsafe_allow_html = True)
    cols[2].markdown("<b>Decrease?</b>", unsafe_allow_html = True)
    for i in range(min(len(results), 50)):
                
                if len(results[i]) < 3: continue
                    
                cols[0].write(results[i])
                enhance = cols[1].checkbox('✓', key = "en"+str(i) if keys=="normal" else random.randint(0,int(1e16)),value=False)
                decrease = cols[2].checkbox('✗', key = "de"+str(i) if keys == "normal" else random.randint(0,int(1e16)),value=False)
                hash_val = hash(results[i])
                if enhance:
                    #st.write("added sentence {}".format(results[i]))
                    session_state.enhance.add(hash_val)
                else:
                    #st.write("removed sentence {}".format(results[i]))
                    if hash_val in session_state.enhance: session_state.enhance.remove(hash_val)
                if decrease:
                    session_state.decrease.add(hash(results[i]))
                else:
                     if hash_val in session_state.decrease: session_state.decrease.remove(hash_val)
                            
    
st.title('COVID-19 Similarity Search')
RESULT_FILTREATION = False
#a = st.empty()
mode = st.sidebar.radio("Mode", ("Start with Sentence", "Start with Query"))
similarity = "dot product" #st.sidebar.selectbox('Similarity', ('dot product', "l2"))
pooling = st.sidebar.selectbox('Pooling', ('cls', 'mean-cls'))
to_decrease, to_enhance = [], []
session_state = SessionState.get(start=False, enhance=set(), decrease=set(), interactive = False, started = False, vec=None)

#if mode == "Sentencve":
#    filter_by_spike = True if st.sidebar.selectbox('Filter by SPIKE query?', ('False', 'True'))=="True" else False

sents, ids, id2ind, ind2id = load_sents_and_ids()
#sents =  df["sentence_text"].tolist()
#ids = [hash(s) for s in sents]
print("len sents", len(sents))
#print("Creating dicts...")
#id2ind = {ids[i]:i for i,s in enumerate(sents)}
#ind2id = {i:ids[i] for i,s in enumerate(sents)}
#print("Done.")

index = load_index(similarity, pooling)
bert = load_bert()
bert_all_seq = load_bert_all_seq()
bert_alignment_supervised = load_bert_alignment_supervised()
pca = load_pca(pooling)
st.write("Uses {}-dimensional vectors".format(pca.components_.shape[0]))
st.write("Number of indexed sentences: {}".format(len(sents)))
print("Try accessing the demo under localhost:8080 (or the default port).")

#""" 
#if mode == "Sentence" and filter_by_spike:
#
#    #filter_query = st.text_input('Enter a SPIKE query to filter by', 'This [nsubj drug] treats [obj:l coronavirus].')
#
#    query_type = st.radio("Query type", ("syntactic", "boolean", "token"))
#    if query_type == "syntactic":
#        filter_query = st.text_input('Query', 'The [subj:l coronavirus] [copula:w is] prevalent among [w:e bats].')
#    elif query_type == "boolean":
#       filter_query = st.text_input('Query', 'virus lemma=persist on')
#    elif query_type == "token":
#       filter_query = st.text_input('Query', 'novel coronavirus')
#
#    filter_size = int(st.text_input('How many SPIKE search results?',  3000))
#    results_df = spike_queries.perform_query(filter_query, dataset_name = "covid19", num_results = filter_size, query_type = query_type)
#    results_sents = np.array(results_df["sentence_text"].tolist())
#    results_ids = [hash(s) for s in results_sents]
#"""

if mode == "Start with Sentence":

    placeholder = st.empty()
    input_sentence = placeholder.text_input('Enter a sentence for similarity search', '')
    #input_sentence = st.text_input('Enter a sentence for similarity search', 'The virus can spread rapidly via different transimission vectors.')
    #st.write("try", session_state.enhance, session_state.decrease)

    filter_by =  st.selectbox('Filter results based on:', ('None', 'Boolean query', 'Token query', 'Syntactic query'))
    query_type = "syntactic" if "syntactic" in filter_by.lower() else "boolean" if "boolean" in filter_by.lower() else "token" if "token" in filter_by.lower() else None
    filter_by_spike = query_type is not None

    if query_type == "syntactic":
        filter_query = st.text_input('SPIKE query', SYNTACTIC_QUERY_DEFAULT)
    elif query_type == "boolean":
       filter_query = st.text_input('SPIKE query', BOOLEAN_QUERY_DEFAULT)
    elif query_type == "token":
       filter_query = st.text_input('SPIKE query', TOKEN_QUERY_DEFAULT)
    
    if query_type is not None:

        filter_size = st.slider('Max number of results', 1, 10000, 3000)
        results_df = spike_queries.perform_query(filter_query, dataset_name = "covid19", num_results = filter_size, query_type = query_type)
        results_sents = np.array(results_df["sentence_text"].tolist())
        results_ids = [hash(s) for s in results_sents]

        st.write("Found {} matches".format(len(results_ids)))
    else:

        number_of_sentence_results  = st.slider('Number of results', 1, 1000, 50) #int(st.text_input('Number of results',  100))

elif mode == "Start with Query":

    query_type = st.radio("Query type", ("Boolean", "Token", "Syntactic"))
    query_type = query_type.lower()
    if query_type == "syntactic":
        input_query = st.text_input('Query to augment', SYNTACTIC_QUERY_DEFAULT)
    elif query_type == "boolean":
       input_query = st.text_input('Query to augment', BOOLEAN_QUERY_DEFAULT)
    elif query_type == "token":
       input_query = st.text_input('Query to augment', TOKEN_QUERY_DEFAULT)

    max_results = st.slider('Max number of SPIKE results', 1, 1000, SPIKE_RESULTS_DEFAULT)  #int(st.text_input("Max number of results", 25))
    max_number_of_augmented_results = st.slider('Number of Augmented results', 1, 250000, 1000)
    if query_type == "syntactic":
        perform_alignment = st.checkbox("Perform argument alignment", value=False, key=None)
    else:
        perform_alignment = False
    
    if perform_alignment:
        
        number_of_sentences_to_align = st.select_slider('Number of sentences to align.', options=[1, 10, 25, 50, 100, 200, 250, 500], value = NUM_RESULTS_TO_ALIGN_DEFAULT)
        alignment_method = st.radio("Alignment model", ('Metric model', 'Naive'))
        if alignment_method != "Naive": 
            max_ngrams = st.select_slider('Maximum span size to align', options=[1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15], value = DEFAULT_MAX_NGRAM)
    
    must_include = st.text_input('Get only results containing the following words', '')
    filter_by = st.selectbox('Filter results based on:', ('None', 'Boolean query', 'Token query', 'Syntactic query'))
    query_type_filtration = "syntactic" if "syntactic" in filter_by.lower() else "boolean" if "boolean" in filter_by.lower() else "token" if "token" in filter_by.lower() else None
    filter_by_spike = query_type_filtration is not None
    if filter_by_spike:
        message = "Get only results NOT captured by this query"
        if query_type_filtration == "syntactic":
            filter_query = st.text_input(message, SYNTACTIC_QUERY_DEFAULT)
        elif query_type_filtration == "boolean":
            filter_query = st.text_input(message, BOOLEAN_QUERY_DEFAULT)
        elif query_type_filtration == "token":
            filter_query = st.text_input(message, TOKEN_QUERY_DEFAULT)

        filtration_batch_size = st.slider('Filtration batch size', 1, 250, 50)
        RESULT_FILTREATION = True
show_results = True
#is_interactive_button = st.radio("Interactive?", ('✓', '✗'), index=0 if session_state.interactive else 1)
#if is_interactive_button == "✓":
#    session_state.interactive = True
#else:
#    session_state.interactive = False
start = st.button('Run')
st.write("Current query: {}".format(input_sentence))
if start:
    session_state.started = True

if (start or session_state.start) and session_state.started:
 if mode == "Start with Sentence": session_state.start = True
    
 if mode == "Start with Sentence":
    
    if len(session_state.enhance) == 0 and input_sentence != "": #not is_interactive_button=="✓":
       zero_input()
       st.write("USING A USER-PROVIDED SENTENCE")
       encoding = encode(input_sentence, pca, bert, pooling) #pca.transform(bert.encode([input_sentence], [1], batch_size = 1, strategy = pooling, fname = "dummy.txt", write = False))
       session_state.vec = encoding
    
    if start and len(session_state.enhance) != 0:
       
       zero_input()
       session_state.interactive = True
       #st.write("USING THE {} VECTORS THE USER MARKED".format(len(session_state.enhance) + len(session_state.decrease)))
       encoding_pos = np.array([index.reconstruct(id2ind[i]) for i in session_state.enhance if i in id2ind]) #np.array([index.reconstruct(i) for i in session_state.enhance])
       encoding = np.mean(encoding_pos, axis = 0)
       encoding_neg = np.zeros_like(encoding_pos)
       if len(session_state.decrease) != 0:
            encoding_neg += np.mean(np.array([index.reconstruct(id2ind[i]) for i in session_state.decrease if i in id2ind]), axis = 0)
       encoding = encoding - encoding_neg
       session_state.enhance = set()
       session_state.decrease = set()
       session_state.vec = encoding
       #write_results_menu(results, session_state)
    
    if ((not start) and len(session_state.enhance) != 0) or (input_sentence==""):
        
        encoding = session_state.vec
    
    if not filter_by_spike:
        #st.write(encoding.shape, pca.components_.shape, index.d)
        #st.write(help(index))
        
        D,I = index.search(np.ascontiguousarray(encoding).astype("float32"), number_of_sentence_results)
    
    else:
        
        encoding_of_spike_results = np.array([index.reconstruct(id2ind[i]) for i in results_ids if i in id2ind])
        if encoding_of_spike_results.shape[0] > 0:
            show_results = False
            with st.spinner('Retrieving simialr sentences...'):
                sims = sklearn.metrics.pairwise.cosine_similarity(encoding, encoding_of_spike_results)
                idx_sorted = sims.argsort()[0]
                spike_sents_sorted = results_sents[idx_sorted][::-1]
                I = np.array([[id2ind[hash(s)] for s in spike_sents_sorted if hash(s) in id2ind]])   

        else:
            show_results = False
            st.write("SPIKE search results are not indexed.")
    
    # TODO: CHECK WHY ROWS  RE ADDED TO I AFTER MARKING
    #st.write("TRY", I.squeeze())
    I = I.squeeze()
    if len(I.shape) != 1:
        I = I[0]
    
    results = [sents[i] for i in I if must_include in sents[i]]
    if RESULT_FILTREATION:
        results = result_sents 
        
        
    cols = st.beta_columns((10,1,1))
    cols[0].markdown("<b>Sentence</b>", unsafe_allow_html = True)
    #cols[1].markdown("<b>Enhance?</b>", unsafe_allow_html = True)
    #cols[2].markdown("<b>Decrease?</b>", unsafe_allow_html = True)
    for i in range(min(len(results), 50)):
                                    
                cols[0].write(results[i])
                enhance = cols[0].checkbox('✓', key = "en"+str(i) ,value=False)
                decrease = cols[0].checkbox('✗', key = "de"+str(i),value=False)
                
                #cols[0].write("")
                #cols[1].write("")
                #cols[2].write("")
                
                hash_val = hash(results[i])
                if enhance:
                    #st.write("added sentence {}".format(results[i]))
                    session_state.enhance.add(hash_val)
                else:
                    #st.write("removed sentence {}".format(results[i]))
                    if hash_val in session_state.enhance: session_state.enhance.remove(hash_val)
                if decrease:
                    session_state.decrease.add(hash(results[i]))
                else:
                     if hash_val in session_state.decrease: session_state.decrease.remove(hash_val)
        

            
 elif mode == "IDs":
    input_ids = st.text_input('Input ids', '39, 41, 49, 50, 112, 116, 119, 229, 286, 747')
    input_ids = [int(x) for x in input_ids.replace(" ", "").split(",")]
    st.write("First sentences corrsponding to those IDs:")
    l = range(min(10, len(input_ids) ) )
    query_sents = [sents[id2ind[input_ids[i]]] for i in l]
    st.table(query_sents) 
    encoding = np.array([index.reconstruct(id2ind[i]) for i in input_ids])
    encoding = np.mean(encoding, axis = 0)
    D,I = index.search(np.ascontiguousarray([encoding]).astype("float32"), 150)


 elif mode == "Start with Query":

    with st.spinner('Performing SPIKE query...'):
        results_df = spike_queries.perform_query(input_query, dataset_name = "covid19", num_results = max_results, query_type = query_type)
        results_sents = results_df["sentence_text"].tolist()

        results_ids = [hash(s) for s in results_sents] #results_df["sentence_id"].tolist()

        st.write("Found {} matches".format(len(results_ids)))

        if len(results_sents) > 0:
            st.write("First sentences retrieved:")
            st.table(results_sents[:10])

            encoding = np.array([index.reconstruct(id2ind[i]) for i in results_ids if i in id2ind])
            if encoding.shape[0] > 0:
                
                with st.spinner('Retrieving similar sentences...'):
                    encoding = np.mean(encoding, axis = 0)
                    D,I = index.search(np.ascontiguousarray([encoding]).astype("float32"), max_number_of_augmented_results)
                    result_sents = [sents[i].replace("/","-") for i in I.squeeze()]
                    if must_include != "":
                        result_sents = [sents[i].replace("/","-") for i in I.squeeze() if must_include in sents[i]]

                    if filter_by_spike:
                        with st.spinner('Filtering...'):

                            start = time.time()
                            # filter by lucene queries
                            results_sents_filtered = []

                            def remove_all_words(s):
                                words_to_remove = [" is ", " are ", " the ", " a ", " an ", " to ", " as ", " from ",
                                                   " and ", " or ", " of ", " in ", " be ", " this ", " that ", " , ", " these ", " those ",
                                                   " with ", " within ", " can ", " / "]
                                
                                s = s.replace("The ", "").replace("In ", "").replace("Although ", "").replace("It ", "").replace(" (", "").replace(" )", "").replace("A ", "").replace("An ", "").replace(" [", "").replace(" ]", "")
                                s = s.replace(' " ', ' ').replace(" ' ", " ")
                                s = s.replace(" 's "," ").replace("(","").replace(")", "").replace("[","").replace("]","")
                                for w in words_to_remove:
                                    s = s.replace(w, " ")
                                #s = s.replace("/", "-")

                                while "  " in s:
                                    s = s.replace("  ", " ")
                                    
                                words = s.split(" ")
                                s = " ".join([w for w in words if "-" not in w and "/" not in w and "'" not in w and ")" not in w and "(" not in w and "]" not in w
                                             and "[" not in w and "," not in w and not w=="has" and not w=="have" and not w=="been" and not w=="on"])
                               
                                return s

                            filtration_sents = []
                            start_time = time.time()
                            for b in range(0, len(result_sents), filtration_batch_size):
                                start, end = b, b+filtration_batch_size     
                                all_words = " OR ".join(["("+ " AND ".join(remove_all_words(s).split(" ")[:8])+")" for s in result_sents[start:end]][:])
                                #all_words = all_words.replace("AND AND", "AND")
    
                                results_df_filtration = spike_queries.perform_query(filter_query, dataset_name="covid19",
                                                                      num_results=100000,
                                                                      query_type=query_type_filtration,
                                                                      lucene_query=all_words)
                                filtration_sents.extend(results_df_filtration["sentence_text"].tolist())
            
                            #filtration_sents = results_df_filtration["sentence_text"].tolist()
                            st.write("Num filtration results: {}".format(len(filtration_sents)))
                            #st.write("Filtration sentences:")
                            #st.write(st.table(filtration_sents))
                            st.write("==============================")
                            #st.write(all_words)
                            #st.write(len(results_df_filtration))
                            #st.write("------------")
                            #st.write(st.table(filtration_sents[:5]))
                            #st.write("=====================")

                            result_sents = [s for s in result_sents if s not in set(filtration_sents)] # take only sents not captured by the query
                            st.write("Filtration took {} seconds".format(time.time() - start_time))

                            # start = time.time()
                            # # filter by lucene queries
                            # results_sents_filtered = []
                            # for s in result_sents:
                            #     words = " AND ".join(s.split(" ")[:12])
                            #     results_df = spike_queries.perform_query(filter_query, dataset_name="covid19",
                            #                                          num_results=100000,
                            #                                          query_type=query_type_filtration,
                            #                                          lucene_query=words)
                            #     if len(results_df) == 0: # if not captured by the query
                            #         results_sents_filtered.append(s)
                            # result_sents = results_sents_filtered
                            # st.write("filteration took {} seconds".format(time.time() - start))
                            # st.write(len(result_sents))

                    if query_type == "syntactic"  and perform_alignment:
                        with st.spinner('Performing argument alignment...'):
                            #colored_sents, annotated_sents= alignment.main(bert_all_seq, result_sents, results_df, input_query, [-1], NUM_RESULTS_TO_ALIGN)
                            if alignment_method == "Naive":
                                colored_sents, annotated_sents = alignment.main(bert_all_seq, result_sents, results_df, input_query, [-1], number_of_sentences_to_align)
                            else:
                                annotated_sents, arg1_items, arg2_items, tuples_items = alignment_supervised.main(bert_alignment_supervised, result_sents, results_df, number_of_sentences_to_align, max_ngrams+1)
                                arg1_counts_df = pd.DataFrame(arg1_items, columns =['entity', 'count'])
                                arg2_counts_df = pd.DataFrame(arg2_items, columns =['entity', 'count'])
                                tuples_counts_df = pd.DataFrame(tuples_items, columns =['entity', 'count'])
                            
                                st.sidebar.write('ARG1 Aggregation:')
                                st.sidebar.write(arg1_counts_df.head(30))
                                st.sidebar.write('ARG2 Aggregation:')
                                st.sidebar.write(arg2_counts_df.head(30))
                                st.sidebar.write('Tuples Aggregation:')
                                st.sidebar.write(tuples_counts_df.head(30))
                            
                            for s in annotated_sents:
                                annotated_text(*s)

            else:
                show_results = False
                st.write("SPIKE search results are not indexed.")           
            #encoding = pca.transform(bert.encode(results_sents, [1]*len(results_sents), batch_size = 8, strategy = pooling, fname = "dummy.txt", write = False))#.squeeze()
            #encoding = np.mean(encoding, axis = 0)
            #D,I = index.search(np.ascontiguousarray([encoding]), 100)
        else:
            show_results = False
            st.write("No resutls found.")

 if show_results:
    pass
    results = [sents[i] for i in I.squeeze() if must_include in sents[i]]
    if RESULT_FILTREATION:
        results = result_sents
    st.write("Performed query of type '{}'. Similarity search results:".format(mode))
    st.write(st.table(results))
