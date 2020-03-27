import numpy as np
import os
import glob
from gensim.models import Word2Vec, KeyedVectors

EMBEDDINGS_DIR = os.path.join("data", "embeddings")
EVAL_DATA_DIR = os.path.join("data", "eval", "FinSemEvl", "FinSemEvl", "analogy")

#model_filename = "20190509_yle_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin"
# model_filename = "20190509_yle-wikipedia_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin"
# model_filename = "fin-word2vec-lemma.bin"

# model_file = os.path.join(EMBEDDINGS_DIR, model_filename)
model_file = "w2w_model.model"
# print("Using the word2vec model:", model_filename)
print("Loading word2vec model...")
# w2v_model = KeyedVectors.load_word2vec_format(model_file, binary=True)
w2v_model = Word2Vec.load(model_file)
print("Word2vec model loaded.")

correct_total = 0
incorrect_total = 0
OOV_line_total = 0 # out of vocabulary
n_lines = 0
for eval_file in glob.glob(os.path.join(EVAL_DATA_DIR, "*.txt")): # for each file in dir
    print("Analogy task using the data in:", eval_file)
    with open(eval_file, "r") as f:
        # split file into lines and each line into words to create list nested inside a list
        data = [line.split() for line in f.read().splitlines()]

    correct = 0
    incorrect = 0
    OOV_line = 0 
    for line in data:
        try:
            # the analogy task:
            # line[0] = line[1] + line[2] - line[3]
            # e.g., leveä = kapea + pitkä - lyhyt
            # or    äiti = isä + tyttö - poika
            # or    bangkok = thaimaa + tallinna - viro
            result = w2v_model.most_similar(positive=[line[1], line[2]], negative=[line[3]], topn=1)
            if result[0][0] == line[0]:
                correct += 1
            else:
                incorrect += 1
            # print(line, result)
        except KeyError as err:
            OOV_line += 1
            # print(err)

    try:
        print("N of lines: {n}, correct: {correct} ({per:.2f}%), lines with OOV: {oov}".format( \
            n=len(data), correct=correct, per=correct*100/(correct+incorrect), oov=OOV_line))
    except ZeroDivisionError:
        print("N of lines: {n}, correct: {correct}, lines with OOV: {oov}".format( \
            n=len(data), correct=correct, oov=OOV_line))
    print()
    
    correct_total += correct
    incorrect_total += incorrect
    OOV_line_total += OOV_line
    n_lines += len(data)

print("Total scores:")
print("N of lines: {n}, correct: {correct} ({per:.2f}%), lines with OOV: {oov} ({oovper:.2f}%)".format( \
    n=n_lines, correct=correct_total, per=correct_total*100/(correct_total+incorrect_total), \
        oov=OOV_line_total, oovper=OOV_line_total*100/n_lines))