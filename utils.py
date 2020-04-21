import os
import csv
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import config
import glob

def capitalize_data():
    """
    Capitalize proper nouns for analogy and intrusion evaluation
    """
    for eval_file in glob.glob(os.path.join(config.EVAL_DATA_DIR, 
                                            "intrusion", "*.txt")):
        if "countries" in eval_file or "cities" in eval_file or \
            "philosophers" in eval_file:
            #Use capitalized_file to write to new file
            capitalized_file = eval_file[:-4]+"_capitalized"+".txt"
            #print(eval_file)
            with open(eval_file, "r", encoding="utf-8") as f:
                # split file into lines and each line into words
                # to create list nested inside a list
                data = [line.split() for line in f.read().splitlines()]
                capitalized_data=""
                for line in data:
                    for word_idx in range(len(line)):
                        line[word_idx] = line[word_idx].capitalize()
                    capitalized_data += " ".join(line) + "\n"
            with open(capitalized_file, 'w', encoding='utf-8') as f:
                f.write(capitalized_data)
                
    for eval_file in glob.glob(os.path.join(config.EVAL_DATA_DIR, 
                                            "analogy", "*.txt")):
        if "country" in eval_file or "city" in eval_file:
            #Use capitalized_file to write to new file
            capitalized_file = eval_file[:-4]+"_capitalized"+".txt"
            #print(eval_file)
            with open(eval_file, "r", encoding="utf-8") as f:
                # split file into lines and each line into words
                # to create list nested inside a list
                data = [line.split() for line in f.read().splitlines()]
                capitalized_data=""
                for line in data:
                    if "currency" in eval_file:
                        for word_idx in range(0,len(line),2):
                            line[word_idx] = line[word_idx].capitalize()
                    elif "capital" in eval_file:
                        for word_idx in range(len(line)):
                            line[word_idx] = line[word_idx].capitalize()
                    elif "hockey" in eval_file:
                        for word_idx in range(len(line)):
                            line[word_idx] = line[word_idx].capitalize()
                            if word_idx % 2 == 0:
                                if len(line[word_idx])<5:
                                    line[word_idx] = line[word_idx].upper()
                                elif line[word_idx][-2:] == "pa":
                                    line[word_idx] = line[word_idx][:-2] + \
                                        line[word_idx][-2:].capitalize()
                    capitalized_data += " ".join(line) + "\n"
            with open(capitalized_file, 'w', encoding='utf-8') as f:
                f.write(capitalized_data)
                
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
