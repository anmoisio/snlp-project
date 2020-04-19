import os

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
                # 'hashfxn':<built-in function hash>,
                'iter':5,
                'null_word':0,
                'trim_rule':None,
                'sorted_vocab':1,
                'batch_words':10000,
                'compute_loss':False,
                'callbacks':(),
                'max_final_vocab':None
}

# fasttext default train parameters
default_args_fasttext = {
                'min_n':3,         # min length of char ngram
                'max_n':6,         # max length of char ngram
                'bucket':2000000,  # number of buckets
                'compatible_hash':True
}
default_args_fasttext.update(default_args)
default_args_fasttext.pop('compute_loss')

# word2vec train parameters
arguments = {
            'size': 100,
            'alpha': 0.025,
            'window': 5,
            'min_count': 5,
            'workers': 3,
            'sg': 1,
            'negative': 5,
            'iter': 5,
}

# fasttext train parameters
arguments_fasttext = {
            'min_n':3,
            'max_n':6,
}
arguments_fasttext.update(arguments)


# create a string from the args for the file name
print_parameters = {
            'size',
            # 'alpha',
            'window',
            'min_count',
            'sg',
            'negative',
            'iter',
}
arg_string = ""
for param, arg in arguments.items():
    if param in print_parameters:
        arg_string += param + "=" + str(arg) + ","
arg_string = arg_string[:-1] # delete last comma

# directories
EMBEDDINGS_DIR = os.path.join("data", "embeddings")
CORPUS_DIR = os.path.join("data", "corpora")
EVAL_DATA_DIR = os.path.join("data", "eval", "FinSemEvl", "FinSemEvl")

# corpus file
corpus_name = "a-iltalehti-2020-02-28_normalized_split.txt"
# corpus_name = "wikipedia2008_fi_lemmatized.txt"
# corpus_name = "iltalehti-2020-02-28_wikipedia.txt"

train_data_file = os.path.join(CORPUS_DIR, corpus_name)

# Word2Vec/Fasttext
w2vec = False

# model file
binary = True
if binary:
    file_suffix = ".bin"
else:
    file_suffix = ".txt"

# model_filename = corpus_name + arg_string + file_suffix

# model_filename = "20190509_yle_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin"
# model_filename = "20190509_yle-wikipedia_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin"
# model_filename = "fin-word2vec-lemma.bin"
model_filename = "iltalehti-wikipedia-dim=300,window=5,mincount=2,sg=1,negative=15.bin"

model_file = os.path.join(EMBEDDINGS_DIR, model_filename)


# eval
# how many groups of words per category
n_samples = 10

result_file = os.path.join("results", model_filename + "_results.txt")