import pandas as pd
import numpy as np
import spike_queries
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='download covid dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-results', dest='num_results', type=int,
                        default=5000,
                        help='how many results to download')
    parser.add_argument('--output-filename', dest='output_filename', type=str,
                        default="results.tsv",
                        help='results filename')
    parser.add_argument('--query', dest='query', type=str,
                        default="the [subj virus] [verb does] [obj something]",
                        help='query to perform')
    parser.add_argument('--query-type', dest='query_type', type=str,
                        default="syntactic",
                        help='query type')                         
    args = parser.parse_args() 
    
    df = spike_queries.perform_query(args.query, num_results = args.num_results, query_type = args.query_type)
    df.to_csv(args.output_filename, sep = "\t")
