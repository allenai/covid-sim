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
import time
from sklearn.cluster import KMeans as Kmeans


@st.cache(allow_output_mutation=True)
def load_sents_and_ids():
    with st.spinner('Loading sentences and IDs...'):
        df = pd.read_csv("covid-all-sents-2.4.csv", sep = "\t")
        sents =  df["sentence_text"].tolist()
        ids = [hash(s) for s in sents]

        id2ind = {ids[i]:i for i,s in enumerate(sents)}
        ind2id = {i:ids[i] for i,s in enumerate(sents)}

        return df, sents, ids, id2ind, ind2id

@st.cache(allow_output_mutation=True)
def load_index(similarity, pooling):
    with st.spinner('Loading FAISS index...'):
        fname = "output.new." + pooling + ".index"
        index = faiss.read_index(fname)
        return index

@st.cache(allow_output_mutation=True)
def load_bert():
    with st.spinner('Loading BERT...'):
        model = bert.BertEncoder("cpu")
        return model

@st.cache(allow_output_mutation=True)        
def load_pca(pooling):

    fname = "output.new." + pooling + ".pca.pickle"
    with open(fname, "rb") as f:
    
        return pickle.load(f)

@st.cache(allow_output_mutation=True)
def perform_clustering(vecs, num_clusts):
    kmeans = Kmeans(n_clusters = num_clusts, random_state = 0)
    kmeans.fit(vecs)
    return kmeans.labels_, kmeans.cluster_centers_

def build_html(first, results):

  s = "<details><summary>" + first + "</summary>"
  s += "<ul>"
  for result in results[1:]:
      s += "<li>" + result + "</li>"

  s += "</ul>"
  s += "</details>"
  return s

st.title('COVID-19 Clustering')

mode = "Sentence" #st.sidebar.radio("Mode", ("Sentence", "SPIKE-covid19"))
similarity = "dot product" #st.sidebar.selectbox('Similarity', ('dot product', "l2"))
pooling = "cls" #st.sidebar.selectbox('Pooling', ('cls', 'mean-cls'))

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_colwidth = 150


df, sents, ids, id2ind, ind2id = load_sents_and_ids()
print("len sents", len(sents))

index = load_index(similarity, pooling)
bert = load_bert()
pca = load_pca(pooling)
st.write("Uses {}-dimensional vectors".format(pca.components_.shape[0]))


if mode == "Sentence":

    input_sentence = st.text_input('Enter a sentence for similarity search', 'The virus can spread rapidly via different transimission vectors.')


    filter_by = "None" #st.selectbox('Filter results based on:', ('None', 'Boolean query', 'Token query', 'Syntactic query')) 
    query_type = "syntactic" if "syntactic" in filter_by.lower() else "boolean" if "boolean" in filter_by.lower() else "token" if "token" in filter_by.lower() else None
    filter_by_spike = query_type is not None

    if query_type == "syntactic":
        filter_query = st.text_input('SPIKE query', 'The [subj:l coronavirus] [copula:w is] prevalent among [w:e bats].')
    elif query_type == "boolean":
       filter_query = st.text_input('SPIKE query', 'virus lemma=persist on')
    elif query_type == "token":
       filter_query = st.text_input('SPIKE query', 'novel coronavirus')
    
    if query_type is not None:

        filter_size = st.slider('Max number of results', 1, 10000, 3000)

        results_df = spike_queries.perform_query(filter_query, dataset_name = "covid19", num_results = filter_size, query_type = query_type)
        results_sents = np.array(results_df["sentence_text"].tolist())
        results_ids = [hash(s) for s in results_sents]

        st.write("Found {} matches".format(len(results_ids)))
    else:

        number_of_sentence_results  = st.slider('Number of results', 1, 10000, 5000) #int(st.text_input('Number of results',  100))
        #number_of_clusters = st.slider("Number of clusters", 2, 512, 50)
    number_of_clusters = st.slider("Number of clusters", 2, 512 if query_type is None else len(results_ids), 50 if query_type is None else len(results_ids)//3)
            
show_results = True
start = st.button('Run')


if start:
 if mode == "Sentence":

    #input_sentence = st.text_input('Input sentence', 'The virus can spread rapidly via different transimission vectors.')
    encoding = pca.transform(bert.encode([input_sentence], [1], batch_size = 1, strategy = pooling, fname = "dummy.txt", write = False))#.squeeze()
    #st.write("filter by spike: ", filter_by_spike)
    #st.write(encoding.shape)
    #st.write(index.d)
    show_all_for_each_cluster = []

    if not filter_by_spike:
        D,I = index.search(np.ascontiguousarray(encoding), number_of_sentence_results)
        results_encodings = np.array([index.reconstruct(int(i)) for i in I.squeeze()])
        results_sents = np.array([sents[i] for i in I.squeeze()])
 
        clust_ids, clust_centroids = perform_clustering(results_encodings, number_of_clusters)
        
        for clust_id in sorted(set(clust_ids.flatten())):
          idx = clust_ids == clust_id
          relevant = results_sents[idx]
          relevant_vecs = results_encodings[idx]

          dists_to_centroid = sklearn.metrics.pairwise_distances(relevant_vecs, [clust_centroids[clust_id]])[:,0]
          idx_sorted = dists_to_centroid.argsort()
          closest_sent = relevant[idx_sorted[0]]

          st.subheader("Cluster {}".format(clust_id))
          df_results  = pd.DataFrame(relevant)
          html = build_html(closest_sent, relevant[idx_sorted])
          st.markdown(html, unsafe_allow_html = True)           
          #st.write(relevant[:3])

    else:
        
        encoding_of_spike_results = np.array([index.reconstruct(id2ind[i]) for i in results_ids if i in id2ind])
        sims = sklearn.metrics.pairwise.cosine_similarity(encoding, encoding_of_spike_results)
        idx_sorted = sims.argsort()[0]
        spike_sents_sorted = results_sents[idx_sorted][::-1]
        I = np.array([[id2ind[hash(s)] for s in spike_sents_sorted if hash(s) in id2ind]])

 if show_results:
    pass
    #results = [sents[i] for i in I.squeeze()]
    #st.write("Performed query of type '{}'. Similarity search results:".format(mode))
    #st.write(st.table(results))
    
