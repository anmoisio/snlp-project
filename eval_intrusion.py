import numpy as np
import os
import csv
import glob
from gensim.models import Word2Vec, KeyedVectors
from random import sample, shuffle, seed
seed(12345) # change integer to get different random samples

EMBEDDINGS_DIR = os.path.join("data", "embeddings")
EVAL_DATA_DIR = os.path.join("data", "eval", "FinSemEvl", "FinSemEvl", "intrusion")

model_filename = "20190509_yle_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin"
# model_filename = "20190509_yle-wikipedia_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin"
# model_filename = "fin-word2vec-lemma.bin"

model_file = os.path.join(EMBEDDINGS_DIR, model_filename)
print("Using the word2vec model:", model_filename)
print("Loading word2vec model...")
w2v_model = KeyedVectors.load_word2vec_format(model_file, binary=True)
print("Word2vec model loaded.")

# how many groups of words per category
n_samples = 10 

# dict of task samples
# {'intrusion/animals.txt': [([cat, dog, basketball], basketball), ([horse, Sweden, cow], Sweden)], etc}
all_samples = {} 

for eval_file in glob.glob(os.path.join(EVAL_DATA_DIR, "*.txt")): # for all files in the dir
    with open(eval_file, "r") as f:
        # split file into lines and each line into words to create lists nested inside a list
        data = [line for line in f.read().splitlines()]

    samples = []
    for i in range(n_samples):
        word_sample = sample(data, 5) # 5 words from this category
        while True:
            # sample one of the other categories to get the intrusion word
            file_intrusion = sample(glob.glob(os.path.join(EVAL_DATA_DIR, "*.txt")), 1)
            # make sure it's a different category
            if file_intrusion != eval_file: break 

        with open(file_intrusion[0], "r") as f:
            intrusion_words = [line for line in f.read().splitlines()]

        intrusion = sample(intrusion_words, 1)[0] # one word from the intrusion category
        word_sample.append(intrusion)
        shuffle(word_sample)

        samples.append((word_sample, intrusion))

    all_samples[eval_file] = samples

correct_total = 0
incorrect_total = 0
OOV_line_total = 0
n_total = 0
for category, word_samples in all_samples.items():
    print("Intrusion task using the data in:", category)
    correct = 0
    incorrect = 0
    OOV_line = 0
    
    for word_sample in word_samples:
        print(word_sample)
        try:
            result = w2v_model.doesnt_match(word_sample[0])
            print("output: ", result)
            if result == word_sample[1]:
                correct += 1
            else:
                incorrect += 1
        except ValueError as err:
            OOV_line += 1
            # print(err)

    n_total += len(word_samples)
    correct_total += correct
    incorrect_total += incorrect
    OOV_line_total += OOV_line

    try:
        print("N of tasks: {n}, correct: {correct} ({per:.2f}%), tasks with OOV: {oov}".format( \
            n=len(word_samples), correct=correct, per=correct*100/(correct+incorrect), oov=OOV_line))
    except ZeroDivisionError:
        print("N of tasks: {n}, correct: {correct}, tasks with OOV: {oov}".format( \
            n=len(data), correct=correct, oov=OOV_line))
    print()

print("Total scores:")
try:
    print("N of tasks: {n}, correct: {correct} ({per:.2f}%), tasks with OOV: {oov}".format( \
        n=n_total, correct=correct_total, per=correct_total*100/(correct_total+incorrect_total), oov=OOV_line_total))
except ZeroDivisionError:
    print("N of tasks: {n}, correct: {correct}, tasks with OOV: {oov}".format( \
        n=n_total, correct=correct_total, oov=OOV_line_total))
print()
