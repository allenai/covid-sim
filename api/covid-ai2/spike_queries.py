import requests
import pandas as pd
import streamlit as st

COVID_URL = "https://spike.staging.apps.allenai.org/api/3/search/query" #"http://35.242.203.108:5000/api/3/search/query"
COVID_BASE_URL = "https://spike.staging.apps.allenai.org" #"http://35.242.203.108:5000"

PUBMED_URL = "http://34.89.172.235:5000/api/3/search/query"
PUBMED_BASE_URL = "http://34.89.172.235:5000"

WIKIPEDIA_URL = "https://spike.staging.apps.allenai.org/api/3/search/query"
WIKIPEDIA_BASE_URL = "https://spike.staging.apps.allenai.org"


def get_tsv_url(response: requests.models.Response, results_limit: int, base_url) -> str:
    tsv_location = response.headers["tsv-location"]
    tsv_url = base_url + tsv_location + "?sentence_text=True&capture_indices=True&sentence_id=True&limit={}".format(
        results_limit)
    return tsv_url


def perform_query(query_str: str, dataset_name: str = "pubmed", num_results: int = 10, query_type: str = "syntactic",
                  remove_duplicates: bool = True, lucene_query="") -> pd.DataFrame:
    template = """{{
  "queries": {{"{query_type}": "{query_content}", "lucene": "{lucene_query}"}},
  "data_set_name": "{dataset_name}"
}}"""

    template = """{{
      "queries": {{"{query_type}": "{query_content}", "lucene": "{lucene_query}"}},
      "data_set_name": "{dataset_name}"
    }}"""


    query = template.format(query_content=query_str, dataset_name=dataset_name, query_type=query_type, lucene_query=lucene_query)
    #st.write("******************")
    #st.write(query)
    #st.write("******************")

    headers = {'content-type': 'application/json'}
    if dataset_name == "pubmed":
        url, base_url = PUBMED_URL, PUBMED_BASE_URL
    elif dataset_name == "covid19":
        url, base_url = COVID_URL, COVID_BASE_URL
    elif dataset_name == "wiki":
        url, base_url = WIKIPEDIA_URL, WIKIPEDIA_BASE_URL

    response = requests.post(url, data=query.encode('utf-8'), headers=headers)
    try:
        tsv_url = get_tsv_url(response, results_limit=num_results, base_url=base_url)
    except Exception as e:
        st.write("Invalid SPIKE query. Please check query content and/or its type.")
        raise e
        

    df = pd.read_csv(tsv_url, sep="\t")

        
    # if remove_duplicates:
    #     df = df.drop_duplicates("sentence_text")
    return df
