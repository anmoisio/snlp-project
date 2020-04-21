import utils
import config
from train import train_model
from evaluate import *
from gensim.models.keyedvectors import FastTextKeyedVectors
import os

# select Word2Vec/Fasttext
# model_type = 'Word2Vec'
model_type = 'FastText'

# save type, i.e. suffix of the model file
model_file_type = ".bin"
# model_file_type = ".model"
# model_file_type = ".txt"

# train new or use pretrained?
TRAIN = True

if TRAIN:
    """
    default_args = {
        'sentences':None,
        'corpus_file':None,
        'size':100,
        'alpha':0.025, 
        'window':5,
        'min_count':5,
        'max_vocab_size':None,
        'sample':0.001,
        'seed':1,
        'workers':3,
        'min_alpha':0.0001,
        'sg':0,
        'hs':0,
        'negative':5,
        'ns_exponent':0.75,
        'cbow_mean':1,
        'iter':5,
        'null_word':0,
        'trim_rule':None,
        'sorted_vocab':1,
        'batch_words':10000,
        'compute_loss':False,
        'callbacks':(),
        'max_final_vocab':None,
        # fasttext default train parameters
        'min_n':3,         # min length of char ngram
        'max_n':6,         # max length of char ngram
        'bucket':2000000,  # number of buckets
        'compatible_hash':True}
    """

    # train parameters
    arguments = {
        'size': 100,
        'alpha': 0.025,
        'window': 5,
        'min_count': 5,
        'workers': 3,
        'sg': 1,
        'negative': 5,
        'iter': 5,
        # fasttext train parameters
        'min_n':3,
        'max_n':6}

    print_parameters = {
        'size',
        'alpha',
        'window',
        'min_count',
        'sg',
        'negative',
        'iter',}

    # corpus file
    # corpus_name = "a-iltalehti-2020-02-28_normalized_split"
    # corpus_name = "wikipedia2008_fi_lemmatized"
    # corpus_name = "iltalehti-2020-02-28_wikipedia"
    corpus_name = "test_corpus"

    train_data_file = os.path.join(config.CORPUS_DIR, model_type + "_" + corpus_name + ".txt")

    # loop to train and evaluate multiple models
    for min_count in range(5):
        arguments['min_count'] = min_count + 1

        # create a string from the args for the file name
        # if fasttext, print also n-gram lengths
        if model_type == 'FastText':
            print_parameters.update({'min_n','max_n'})

        arg_string = ""
        for param, arg in arguments.items():
            if param in print_parameters:
                arg_string += param + "=" + str(arg) + ","
        arg_string = arg_string[:-1] # delete last comma

        # model file
        model_filename = corpus_name + "_" + arg_string
        # remember to switch the right model_file_type
        model_file = os.path.join(config.EMBEDDINGS_DIR, model_filename + model_file_type)

        model = train_model(model_type, arguments, model_file, train_data_file)

        # evaluate
        result_string = intrusion(model)
        result_string += analogy(model)
        result_string += nearest_neighbours(model)

        result_file = os.path.join("results", model_type + model_filename + "_results.txt")

        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(result_string)

else:
    # use a pretrained model
    # model_filename = "20190509_yle_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5"
    # model_filename = "20190509_yle-wikipedia_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5"
    # model_filename = "fin-word2vec-lemma"
    model_filename = "wikipedia2008_fi_lemmatized_size=200,alpha=0.025,window=5,min_count=2,sg=1,negative=15,iter=5"

    # remember to switch the right model_file_type
    model_file = os.path.join(config.EMBEDDINGS_DIR, model_filename + model_file_type)

    print("Using the "+model_type+" model:", model_filename)
    print("Loading "+model_type+" model...")
    if model_type == 'Word2Vec':
        try:
            model = KeyedVectors.load_word2vec_format(model_file, binary=True)
        except UnicodeDecodeError:
            model = KeyedVectors.load_word2vec_format(model_file, binary=False)
    else:
        model = FastTextKeyedVectors.load(model_file)
    print(model_type+" model loaded.")

    # evaluate
    result_string = intrusion(model)
    result_string += analogy(model)
    result_string += nearest_neighbours(model)

    result_file = os.path.join("results", model_type + model_filename + "_results.txt")

    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(result_string)

