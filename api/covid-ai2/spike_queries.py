import requests
import pandas as pd
import streamlit as st

COVID_URL = "http://35.242.203.108:5000/api/3/search/query" 
COVID_BASE_URL = "http://35.242.203.108:5000"
    
PUBMED_URL = "http://34.89.172.235:5000/api/3/search/query"
PUBMED_BASE_URL = "http://34.89.172.235:5000"



def get_tsv_url(response: requests.models.Response, results_limit: int, base_url) -> str:
    
    tsv_location = response.headers["tsv-location"]
    tsv_url = base_url + tsv_location +"?sentence_text=True&capture_indices=True&sentence_id=True&limit={}".format(results_limit)
    return tsv_url
    
@st.cache(allow_output_mutation=True)    
def perform_query(query: str, dataset_name: str = "pubmed", num_results: int = 10, query_type : str = "syntactic", remove_duplicates: bool = True) -> pd.DataFrame:

   template = """{{
  "queries": {{"{query_type}": "{query_content}"}},
  "data_set_name": "{dataset_name}"
}}"""
        
   query = template.format(query_content = query, dataset_name = dataset_name, query_type = query_type)
   headers = {'content-type': 'application/json'}
   if dataset_name == "pubmed":
        url, base_url = PUBMED_URL, PUBMED_BASE_URL
   elif dataset_name == "covid19":
        url, base_url = COVID_URL, COVID_BASE_URL
   
   response = requests.post(url, data = query, headers = headers)
   tsv_url = get_tsv_url(response, results_limit = num_results, base_url = base_url)
   df = pd.read_csv(tsv_url, sep = "\t")
   if remove_duplicates:
        df = df.drop_duplicates("sentence_text")
   return df
