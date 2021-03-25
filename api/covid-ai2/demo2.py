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
NUM_RESULTS_TO_ALIGN_DEFAULT = 1500
DEFAULT_MAX_NGRAM = 5
BOOLEAN_QUERY_DEFAULT = "virus lemma=originate"
TOKEN_QUERY_DEFAULT = "novel coronavirus"
SYNTACTIC_QUERY_DEFAULT = "a1:[w]COVID-19 $causes a2:something" #"arg1:[e]paracetamol is the recommended $treatment for arg2:asthma."
SPIKE_RESULTS_DEFAULT = 75
must_include = ""
import base64
import plotly.graph_objects as go
from collections import Counter

st.set_page_config(layout="wide")
st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 375px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    """<style>
        .dataframe {text-align: left !important}
    </style>
    """, unsafe_allow_html=True) 

def plotly_table(results, title):
    style=True
    filter_table = results# _filter_results(results, number_of_rows, number_of_columns)

    header_values = list(filter_table.columns)
    cell_values = []
    for index in range(0, len(filter_table.columns)):
        cell_values.append(filter_table.iloc[:, index : index + 1])

    if not style:
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=header_values), cells=dict(values=cell_values)
                )
            ]
        )
    else:
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=header_values, fill_color="paleturquoise", align="left"
                    ),
                    cells=dict(values=cell_values, fill_color="lavender", align="left"),
                )
            ]
        )

    with st.beta_expander(title):
        st.plotly_chart(fig)
      
      
def print_spike_results(results, arg1_lst, arg2_lst, title):

    st.markdown("<h3>{}</h3>".format(title), unsafe_allow_html = True)
    html = """"""
    for s,arg1,arg2 in zip(results,arg1_lst,arg2_lst):
        arg1_first_idx,arg1_last_index = arg1
        arg2_first_idx,arg2_last_index = arg2  
        arg1_str = s[arg1_first_idx:arg1_last_index]
        arg2_str = s[arg2_first_idx:arg2_last_index]    
        
        arg1 = "<font color=‘orange’>{}</font>".format(arg1_str)
        arg2 = "<font color=‘cyan’>{}</font>".format(arg2_str)        
        if arg1_first_idx > arg2_first_idx:
            
            arg1,arg2 = arg2.replace("cyan","orange"), arg1.replace("cyan","orange")
            arg1_first_idx,arg1_last_index, arg2_first_idx,arg2_last_index = arg2_first_idx,arg2_last_index,  arg1_first_idx,arg1_last_index
            
        #s = s[:arg1_first_idx] + " " + arg1 + s[arg1_last_index:arg2_first_idx] + " " + arg2 + s[arg2_last_index:]
                
        html+= "<li>{}</li>".format(s)
    html+="</ul>"
    st.markdown(html, unsafe_allow_html = True)
    
    
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

def project_out(positive, negative):
    
    positive,negative = np.array(positive), np.array(negative)
    pos_basis = scipy.linalg.orth(positive.T)
    P = pos_basis.dot(pos_basis.T)
    st.write(P.shape, negative.shape, positive.shape)
    negative_different = negative - negative@P
    return positive - negative_different

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False, sep = "\t")
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href


st.title('COVID-19 Similarity Search')
RESULT_FILTREATION = False
#a = st.empty()
mode = "Start with Query" #st.sidebar.radio("Mode", ("Start with Sentence", "Start with Query"))
similarity = "dot product" #st.sidebar.selectbox('Similarity', ('dot product', "l2"))
pooling = "cls"# st.sidebar.selectbox('Pooling', ('cls', 'mean-cls'))
to_decrease, to_enhance = [], []
session_state = SessionState.get(start=False, enhance=set(), decrease=set(), interactive = False, started = False, vec=None, current_query="")


sents, ids, id2ind, ind2id = load_sents_and_ids()

print("len sents", len(sents))

index = load_index(similarity, pooling)
bert = load_bert()
bert_all_seq = load_bert_all_seq()
bert_alignment_supervised = load_bert_alignment_supervised()
pca = load_pca(pooling)

my_expander = st.beta_expander("How to query?")
my_expander.markdown("""Start by writing a query that aims to capture a relation between two entities. 
<ul>
  <li>Use <b><font color=‘blue’>$</font></b> or <b><font color=‘blue’>:[w]</font></b> to mark words that <b>must appear</b>. </li>
  <li>Mark the <b>arguments</b> with <b><font color=‘orange’>a2:</font></b> and <b><font color=‘orange’>a2:</font></b> </li>
  <li>Mark with <b><font color=‘brown’>:</font></b> additional captures that fix the required synatctic structure. </li>
</ul>  
For instance, in the query '<b><font color=‘orange’>a1:[w]</font></b>COVID-19 <b><font color=‘blue’>$</font></b>causes <b><font color=‘orange’>a2:</font></b>pain', we search for sentences where the syntactic relation between the first and second argument is the same as the relation between `COVID-19` and `pain` in this sentence (subject-object relation). We further request an exact match for the word `causes` and the argument `COVID-19`. <br> For more details on the query language, check out 
<a href="https://spike.covid-19.apps.allenai.org/datasets/covid19/search/help">this</a> tutorial.""", unsafe_allow_html=True)
#st.write("Uses {}-dimensional vectors".format(pca.components_.shape[0]))
#st.write("Number of indexed sentences: {}".format(len(sents)))
print("Try accessing the demo under localhost:8080 (or the default port).")


if mode == "Start with Query":

    query_type = "Syntactic" #st.radio("Query type", ("Boolean", "Token", "Syntactic"))
    query_type = query_type.lower()
    if query_type == "syntactic":
        input_query = st.text_input('Query to augment', SYNTACTIC_QUERY_DEFAULT)
        input_query = input_query.replace("a1:", "arg1:").replace("a2:", "arg2:")
    max_results = 100 #st.slider('Max number of SPIKE results', 1, 1000, SPIKE_RESULTS_DEFAULT)  
    max_number_of_augmented_results = 1000 #st.slider('Number of Augmented results', 1, 250000, 1000)
    if query_type == "syntactic":
        perform_alignment = True #st.checkbox("Perform argument alignment", value=True, key=None)
    
    if perform_alignment:
        
        number_of_sentences_to_align = 1000 #st.select_slider('Number of sentences to align.', options=[1, 10, 25, 50, 100, 200, 250, 500], value = NUM_RESULTS_TO_ALIGN_DEFAULT)
        alignment_method = "Metric model" #st.radio("Alignment model", ('Metric model', 'Naive'))
        if alignment_method != "Naive": 
            max_ngrams = 5 #st.select_slider('Maximum span size to align', options=[1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15], value = DEFAULT_MAX_NGRAM)
    
    must_include = st.text_input('Get only results containing the following words', '')
#     #filter_by = st.selectbox('Filter results based on:', ('None', 'Boolean query', 'Token query', 'Syntactic query'))
#     query_type_filtration = "syntactic" if "syntactic" in filter_by.lower() else "boolean" if "boolean" in filter_by.lower() else "token" if "token" in filter_by.lower() else None
#     filter_by_spike = query_type_filtration is not None
#     if filter_by_spike:
#         message = "Get only results NOT captured by this query"
#         if query_type_filtration == "syntactic":
#             filter_query = st.text_input(message, SYNTACTIC_QUERY_DEFAULT)
#         elif query_type_filtration == "boolean":
#             filter_query = st.text_input(message, BOOLEAN_QUERY_DEFAULT)
#         elif query_type_filtration == "token":
#             filter_query = st.text_input(message, TOKEN_QUERY_DEFAULT)

#         filtration_batch_size = st.slider('Filtration batch size', 1, 250, 50)
#         RESULT_FILTREATION = True
show_results = True

start = st.button('Run')
#st.write("Current query: {}".format(session_state.current_query))
if start:
    session_state.started = True

if (start or session_state.start) and session_state.started:

 if mode == "Start with Query":

    with st.spinner('Performing SPIKE query...'):
        #st.write("Performing query '{}'".format(input_query))
        results_df = spike_queries.perform_query(input_query, dataset_name = "covid19", num_results = max_results, query_type = query_type)
        results_sents = results_df["sentence_text"].tolist()

        results_ids = [hash(s) for s in results_sents] #results_df["sentence_id"].tolist()

        #st.write("Found {} matches".format(len(results_ids)))

        if len(results_sents) > 0:
            #st.write("First sentences retrieved:")
            #st.table(results_sents[:10])
            print_spike_results(results_sents[:10], list(zip(results_df["arg1_first_index"], results_df["arg1_last_index"])), list(zip(results_df["arg2_first_index"], results_df["arg2_last_index"])), title = "First Sentences Retrived:")
            st.markdown("<h3>Neural Similarity Search Results:</h3>", unsafe_allow_html = True)
            encoding = np.array([index.reconstruct(id2ind[i]) for i in results_ids if i in id2ind])
            if encoding.shape[0] > 0:
                
                with st.spinner('Retrieving similar sentences...'):
                    encoding = np.mean(encoding, axis = 0)
                    D,I = index.search(np.ascontiguousarray([encoding]).astype("float32"), max_number_of_augmented_results)
                    result_sents = [sents[i].replace("/","-") for i in I.squeeze()]
                    results_set = set()
                    result_sents_clean = []
                    for s in result_sents:
                        
                        if s not in results_set:
                            results_set.add(s)
                            result_sents_clean.append(s)
                            
                    result_sents = result_sents_clean
                    
                    if must_include != "":
                        result_sents = [sents[i].replace("/","-") for i in I.squeeze() if must_include in sents[i]]

                    if query_type == "syntactic"  and perform_alignment:
                        with st.spinner('Performing argument alignment...'):
                            #colored_sents, annotated_sents= alignment.main(bert_all_seq, result_sents, results_df, input_query, [-1], NUM_RESULTS_TO_ALIGN)
                            if alignment_method == "Naive":
                                colored_sents, annotated_sents = alignment.main(bert_all_seq, result_sents, results_df, input_query, [-1], number_of_sentences_to_align)
                            else:
                                annotated_sents, arg1_items, arg2_items, tuples_items, captures_tuples = alignment_supervised.main(bert_alignment_supervised, result_sents, results_df, number_of_sentences_to_align, max_ngrams+1)
                                tuples_items = [(t[0], t[1], count) for t, count in tuples_items]
                                arg1_counts_df = pd.DataFrame(arg1_items, columns =['ARG1', 'count'])
                                arg2_counts_df = pd.DataFrame(arg2_items, columns =['ARG2', 'count'])
                                tuples_counts_df = pd.DataFrame(tuples_items, columns =['ARG1', 'ARG2', 'count'])
                                captures_df = pd.DataFrame.from_records(captures_tuples, columns =['ARG1', 'ARG2'])
                                captures_df["sentence"] = result_sents[:len(captures_tuples)]
                                
                                plotly_table(arg1_counts_df.head(50), "Argument 1 Aggregation") 
                                plotly_table(arg2_counts_df.head(50), "Argument 2 Aggregation") 
                                plotly_table(tuples_counts_df.head(50), "Tuples Aggregation") 
                                
                                #st.sidebar.write('ARG1 Aggregation:')
                                #st.sidebar.write(arg1_counts_df.head(30))
                                #st.sidebar.write('ARG2 Aggregation:')
                                #st.sidebar.write(arg2_counts_df.head(30))
                                #st.sidebar.write('Tuples Aggregation:')
                                #st.sidebar.write(tuples_counts_df.head(30))
                                
                                st.markdown(get_table_download_link(captures_df), unsafe_allow_html=True) # download augmented results
     
                            for s in annotated_sents:
                                annotated_text(*s)

            else:
                show_results = False
                st.write("SPIKE search results are not indexed.")           

        else:
            show_results = False
            st.write("No resutls found.")

            
#  if show_results:
#     pass
#     results = [sents[i] for i in I.squeeze() if must_include in sents[i]]
#     if RESULT_FILTREATION:
#         results = result_sents
#     st.write("Performed query of type '{}'. Similarity search results:".format(mode))
#     st.write(st.table(results))
