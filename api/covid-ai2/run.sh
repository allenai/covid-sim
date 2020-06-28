#git clone https://github.com/Shaul1321/covid.git
python3 download_data.py --num-results 50000 --output-filename results.tsv
python3 run_bert.py --input-filename results.tsv --device cpu --pooling cls --output_fname output-cls.jsonl
python3 run_bert.py --input-filename results.tsv --device cpu --pooling mean-cls --output_fname output-mean-cls.jsonl
python3 build_index.py --fname output-cls.jsonl --num_vecs_pca 100000 --pca_variance 0.985 --similarity_type cosine
python3 build_index.py --fname output-mean-cls.jsonl --num_vecs_pca 100000 --pca_variance 0.985 --similarity_type cosine
zip demo_data.zip output-cls.index output-mean-cls.index output-cls.pca.pickle output-mean-cls.pca.pickle results.tsv
