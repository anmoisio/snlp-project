import csv
import os
from gensim.models import Word2Vec, KeyedVectors

EMBEDDINGS_DIR = os.path.join("data", "embeddings")

model_filename = "20190509_yle_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin"
# model_filename = "20190509_yle-wikipedia_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin"
# model_filename = "fin-word2vec-lemma.bin"

model_file = os.path.join(EMBEDDINGS_DIR, model_filename)
print("Using the word2vec model:", model_filename)
print("Loading word2vec model...")
w2v_model = KeyedVectors.load_word2vec_format(model_file, binary=True)
print("Word2vec model loaded.")

wordlist_file = os.path.join("data", "eval", "nearest_neighbor_wordlist.csv")

with open(wordlist_file, 'r', encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    finnish_words = [row[0] for row in csv_reader]

for word in finnish_words:
    print("word:", word, "nearest neighbour:", w2v_model.most_similar(word))
