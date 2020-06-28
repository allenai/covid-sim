import argparse
import faiss
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import subprocess
import tqdm
import pickle

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])
    
def do_pca(num_vecs = 1e6, fname = "output.jsonl", variance = 0.98):

        vecs = []
        with open(fname, "r", encoding = "utf-8") as f:
        
                for i,line in enumerate(f):
                
                        data_dict = eval(line)
                        data_dict["vec"] = np.array([float(x) for x in data_dict["vec"].split(" ")]).astype("float32")
                        vecs.append(data_dict["vec"])                            
                        if i > num_vecs: break
                
        vecs = np.array(vecs)
        print("Fitting PCA to variance {}...".format(variance))
        pca = PCA(n_components = 0.98)
        vecs_pca = pca.fit_transform(vecs)
        print("Done. Number of dimensions: {}".format(vecs_pca.shape[1]))         

        return pca


def add_to_index(index, vecs, pca, cosine = False):

    vecs = np.array(vecs).astype("float32")
    vecs = pca.transform(vecs)
    if cosine:
        vecs /= np.linalg.norm(vecs, axis = 1, keepdims = True)
    index.add(np.ascontiguousarray(vecs))

def index_vectors(similarity_type, fitted_pca, fname):

        dims = fitted_pca.components_.shape[0]
        if similarity_type == "cosine" or similarity_type == "dot_product":       
            index = faiss.IndexFlatIP(dims)
        elif similarity_type == "l2":
            index = faiss.IndexFlatL2(dims)
        else:
            raise Exception("Unsupported metric.")
        
        vecs = []
        
        print("Loading vecs & indexing...")
        
        with open(fname, "r", encoding = "utf-8") as f:
        
            for i,line in tqdm.tqdm(enumerate(f), total = 2.4*1e6):
            
                vec = eval(line)["vec"]
                vec = [float(x) for x in vec.split(" ")]
                vecs.append(vec)
                if i > 150000: break
                
                if (len(vecs) > 2048):
                    add_to_index(index,vecs,pca, similarity_type == "cosine")            
                    vecs = []
                
        if len(vecs) > 0:
        
            add_to_index(index, vecs, pca, similarity_type == "cosine")
                
        print("Done indexing, Saving to file")
        index_fname = fname.rsplit(".", 1)[0]+".index"
        faiss.write_index(index, index_fname)
        return index
        
        
if __name__ == "__main__":
 
        parser = argparse.ArgumentParser(description='balanced brackets generation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--fname', dest='fname', type=str,
                        default="output.mean-cls.jsonl")
        parser.add_argument('--num_vecs_pca', dest='num_vecs_pca', type=int,
                        default=150 * 1e3)
        parser.add_argument('--pca_variance', dest='pca_variance', type=float,
                        default=0.985)   
        parser.add_argument('--similarity_type', dest='similarity_type', type=str,
                        default="cosine")  
                                                                                     
        args = parser.parse_args()
        pca = do_pca(args.num_vecs_pca, args.fname, args.pca_variance)
        pca_filename = args.fname.rsplit(".", 1)[0]+".pca.pickle"
     
        with open(pca_filename, "wb") as f:
           pickle.dump(pca, f)
        index = index_vectors(args.similarity_type, pca, args.fname)
