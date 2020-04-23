import os
import csv
import json
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

def remove_divider():
    with open(os.path.join('data', 'corpora', 'wikipedia2008_fi_lemmatized.txt'), 'r', encoding='utf-8') as f:
        corpus = f.read().replace('|','')

    with open(os.path.join('data', 'corpora', 'wikipedia_new.txt'), 'w', encoding='utf-8') as f:
        f.write(corpus)

def combine_corpora():
    file1 = os.path.join("data", "corpora", "wikipedia_new.txt")
    file2 = os.path.join("data", "corpora", "iltalehti_new.txt")
    combined = os.path.join("data", "corpora", "iltalehti-wikipedia_new.txt")

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


def split_LM_corpus():
    file1 = os.path.join("data", "corpora", "iltalehti_new.txt")

    with open(file1, 'r', encoding='utf-8') as f:
        corpus = f.readlines()

    divide = 10
    truncated = corpus[ : int(len(corpus) / divide)]

    train_portion = 0.8
    train_len = int(len(truncated)*train_portion)
    train = truncated[1 : train_len]
    rest = truncated[train_len : ]

    val = rest[1:int(len(rest)*0.5)]
    test = rest[int(len(rest)*0.5):]

    with open(os.path.join("data", "corpora", "train.txt"), 'w', encoding='utf-8') as f:
        for line in train:
            f.write(line)

    with open(os.path.join("data", "corpora", "valid.txt"), 'w', encoding='utf-8') as f:
        for line in val:
            f.write(line)

    with open(os.path.join("data", "corpora", "test.txt"), 'w', encoding='utf-8') as f:
        for line in test:
            f.write(line)
            
def split_sentences():
    # DEPRECATED. Included in normalise()
    with open(os.path.join('data', 'corpora', 
                           'a-iltalehti-2020-02-28_normalized.txt'),
              'r', encoding='utf-8') as f:
        corpus = f.read().split('.')

    with open(os.path.join('data', 'corpora', 
                           'a-iltalehti-2020-02-28_normalized_split.txt'), 
              'w', encoding='utf-8') as f:
        for line in corpus:
            f.write(line)
            f.write('\n')

def normalise():
    """
    Normalise corpus
    """
    import libvoikko
    #Define a Voikko class for Finnish
    analyzer = libvoikko.Voikko(u"fi")
    
    #Open the text file
    print("Reading the input text file...")
    with open(os.path.join('data', 'corpora', 
                           'a-iltalehti-2020-02-28.txt'),
              'r', encoding='utf-8') as f:
        text = f.read()
    
    #Print text
    #print("TEXT BEFORE NORMALISATION")
    #print(text)
    
    #Remove numbers
    #text = ''.join(c for c in text if not c.isdigit())
    
    #Tokenize & remove punctuation
    #tokenizer = RegexpTokenizer(r'\w+')
    #text = tokenizer.tokenize(text)
    
    #Tokenize
    tokenizer = word_tokenize
    print("Tokenizing...")
    text = word_tokenize(text)
    text_length = len(text)
    
    #Lemmatize tokens
    
    pbar = tqdm(total=text_length, ascii=True, desc = 'Lemmatizing...',
                position=0,unit='keys', unit_scale=True)
    for idx, word in enumerate(text):
        #Check if word is found from dictionary
        if analyzer.analyze(word):
            #Lemmatize the word. analyze() function returns
            #various info for the word
            
            #Check if word starts with lowercase
            if word[0].islower():   
                
                #Check if there are more than 1 possible lemmas in the vocabulary
                if len(analyzer.analyze(word))>1:
                    #Esclude classes paikannimi, sukunimi, etunimi, nimi
                    analyzed = [element for element in analyzer.analyze(word) if
                                'paikannimi' not in element.values() and
                                'sukunumi' not in element.values() and
                                'etunumi' not in element.values() and
                                'nimi' not in element.values()]
                    
                    #Avoid an error if it turns out to be empty list after
                    #excluding these classes
                    if len(analyzed)>0:
                        text[idx] = analyzed[0]['BASEFORM'].lower()
                    else:
                        text[idx] = analyzer.analyze(word)[0]['BASEFORM'].lower()
                
                #Pick the lowercased lemma directly if there is only one lemma
                #for the query word
                else:
                    text[idx] = analyzer.analyze(word)[0]['BASEFORM'].lower()
            
            #The word is capitalized => proper noun or/and the first word of a
            #sentence. Pick the lemma without applying lowercasing.
            else:
                text[idx] = analyzer.analyze(word)[0]['BASEFORM']
        pbar.update(1)
    
    #Print normalized text
    #print("TEXT AFTER NORMALISATION")    
    #print(' '.join(text))
    
    #Write tokenized text to a new text file
    print("\nWriting the normalized text to a txt file...")
    with open(os.path.join('data', 'corpora', 
                           'a-iltalehti-2020-02-28_normalized.txt'),
              'r', encoding='utf-8') as f:
        
        #Write the whole text in one line
        #f.write(' '.join(text))
        
        #Write one sentence per line
        for sentence in ' '.join(text).split('.'):
            f.write(sentence)
            f.write('.\n')

def parse_yle():
    """ Parse Yle news .json corpus"""
    yle_dir = os.path.join(config.CORPUS_DIR, "ylenews-fi-2011-2018-src", "data", "fi")

    # use mode 'a' if you parse many files
    with open(os.path.join(config.CORPUS_DIR, 'yle_small.txt'), 'w', encoding="utf-8") as save_file:
        ## use wildcards * to parse many files
        # for filename in glob.glob(os.path.join(yle_dir, "*", "*", "*.json")):
        for filename in glob.glob(os.path.join(yle_dir, "2018", "01", "0001.json")): 
            with open(filename, 'r', encoding="utf-8") as f:
                json_file = json.load(f)
                for article in json_file['data']:
                    for item in article['content']:
                        if item['type'] == 'text':
                            save_file.write(item['text'])
