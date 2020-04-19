import os
import csv
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def combine_corpora():
    file1 = os.path.join("data", "corpora", "a-iltalehti-2020-02-28_normalized_split.txt")
    file2 = os.path.join("data", "corpora", "wikipedia2008_fi_lemmatized.txt")
    combined = os.path.join("data", "corpora", "iltalehti-2020-02-28_wikipedia.txt")

    with open(file1, 'r', encoding='utf-8') as f:
        corpus1 = f.read()

    with open(file2, 'r', encoding='utf-8') as f:
        corpus2 = f.read()

    with open(combined, 'w', encoding='utf-8') as f:
        f.write(corpus1)
        f.write('\n')
        f.write(corpus2)

def print_clusters():
    file_path = os.path.join("clusters", "wikipedia_clusters_full_k20.txt")

    with open(file_path, "r") as f:
        clusters = f.read().splitlines()

    clustered = [line.split() for line in clusters]

    # separate clusters into different lists
    n_clusters = 20
    clusters = [[] for i in range(n_clusters)]
    for line in clustered[1:]:
        clusters[int(line[1])].append(line[0])

    for c in clusters:
        print(c[:50]) # print 50 words from each cluster


def split_sentences():
    with open(os.path.join('data', 'corpora', 'a-iltalehti-2020-02-28_normalized.txt'), 'r', encoding='utf-8') as f:
        corpus = f.read().split('.')

    with open(os.path.join('data', 'corpora', 'a-iltalehti-2020-02-28_normalized_split.txt'), 'w', encoding='utf-8') as f:
        for line in corpus:
            f.write(line)
            f.write('\n')

def print_csv_rows(file_name, n_rows):
    """
    print n_rows from a .csv file
    """
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            print(row)
            if n_rows == i: break

def print_a_minus_b_plus_c(a, b, c, w2v_model):
    """
    calculate: king - man + woman = ?
    i.e., man is to king what woman is to ?
    """
    result = w2v_model.most_similar(positive=[a, c], negative=[b], topn=10)
    print("'{}' minus '{}' plus '{}' equals:".format(a, b, c))
    for r in result:
        print(r)

def plot_pca(w2v_model):
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
