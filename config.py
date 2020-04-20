import os

# directories
EMBEDDINGS_DIR = os.path.join("data", "embeddings")
CORPUS_DIR = os.path.join("data", "corpora")
EVAL_DATA_DIR = os.path.join("data", "eval", "FinSemEvl", "FinSemEvl")

# how many groups of words per category in the intrusion task?
n_samples = 20