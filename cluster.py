from sklearn import cluster
from sklearn import metrics
import numpy as np
import os
import glob
from gensim.models import Word2Vec, KeyedVectors

EMBEDDINGS_DIR = os.path.join("data", "embeddings")
EVAL_DATA_DIR = os.path.join("data", "eval", "FinSemEvl", "FinSemEvl", "analogy")

#model_filename = "20190509_yle_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin"
# model_filename = "20190509_yle-wikipedia_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin"
model_filename = "fin-word2vec-lemma.bin"

model_file = os.path.join(EMBEDDINGS_DIR, model_filename)
print("Using the word2vec model:", model_filename)
print("Loading word2vec model...")
w2v_model = KeyedVectors.load_word2vec_format(model_file, binary=True)
print("Word2vec model loaded.")

# 
# from: https://ai.intelligentonlinetools.com/ml/k-means-clustering-example-word2vec/

N_CLUSTERS = 8
vocabulary = w2v_model[w2v_model.vocab][0:1000]
kmeans = cluster.KMeans(n_clusters=N_CLUSTERS)
kmeans.fit(vocabulary)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
 
print("Cluster id labels for inputted data")
print(labels)
print("Centroids data")
print(centroids)
 
print("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print(kmeans.score(vocabulary))
 
silhouette_score = metrics.silhouette_score(vocabulary, labels, metric='euclidean')
 
print ("Silhouette_score: ")
print (silhouette_score)