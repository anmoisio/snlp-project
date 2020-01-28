import numpy as np
import os
import csv
from gensim.models import Word2Vec, KeyedVectors

EMBEDDINGS_DIR = os.path.join("data", "embeddings")
EVAL_DATA_DIR = os.path.join("data", "eval", "FinSemEvl", "FinSemEvl", "analogy")
# eval_file = os.path.join(EVAL_DATA_DIR, "ANA_antonymic_adjectives.txt")
eval_file = os.path.join(EVAL_DATA_DIR, "ANA_female-male.txt")

model_filename = os.path.join(EMBEDDINGS_DIR, "20190509_yle-wikipedia_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin")
w2v_model = KeyedVectors.load_word2vec_format(model_filename, binary=True)


with open(eval_file, "r") as f:
    # split file into lines and each line into words to create list nested inside a list
    data = [line.split() for line in f.read().splitlines()]

correct = 0
incorrect = 0
OOV_line = 0 # out of vocabulary
for line in data:
    try:
        result = w2v_model.most_similar(positive=[line[0], line[2]], negative=[line[1]], topn=1)
        if result[0][0] == line[3]:
            correct += 1
        else:
            incorrect += 1
        # print(line, result)
    except KeyError as err:
        OOV_line += 1
        # print(err)


print("N of lines: {}, correct: {}, incorrect: {}, %: {}, lines with OOV: {}".format(len(data), correct, incorrect, correct/(correct+incorrect), OOV_line))