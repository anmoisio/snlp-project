import numpy as np
import os
import csv
from gensim.models import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

EMBEDDINGS_DIR = os.path.join("data", "embeddings")

def print_csv_rows(file_name, n_rows):
    """
    print n_rows from a .csv file
    """
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            print(row)
            if n_rows == i: break

print_csv_rows(os.path.join("data","20190509_yle_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.csv"), 30)
'''
model_filename = os.path.join(EMBEDDINGS_DIR, "20190509_yle-wikipedia_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin")
w2v_model = KeyedVectors.load_word2vec_format(model_filename, binary=True)

def print_a_minus_b_plus_c(a, b, c):
    """
    calculate: king - man + woman = ?
    i.e., man is to king what woman is to ?
    """
    result = w2v_model.most_similar(positive=[a, c], negative=[b], topn=10)
    print("'{}' minus '{}' plus '{}' equals:".format(a, b, c))
    for r in result:
        print(r)
	
print_a_minus_b_plus_c('kuningas', 'mies', 'nainen')


"""
Calculate PCA and create a graph of the embedding space.
Label the word vectors in 'wordlist'.
"""
X = w2v_model[w2v_model.wv.vocab]
pca = PCA(n_components=2)
w2v_result = pca.fit_transform(X)
w2v_words = list(w2v_model.wv.vocab)
plt.figure()
plt.scatter(w2v_result[:, 0], w2v_result[:, 1], marker='.')
wordlist = ["mies", "kuningas", "nainen", "kuningatar"]
for word in wordlist:
	i = w2v_words.index(word)
	plt.annotate(word, xy=(w2v_result[i, 0], w2v_result[i, 1]))
plt.title("")

plt.show()
'''