'''
Mostly from:
https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
'''
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim import utils
import os

train_data_file = os.path.join("data", "corpora", "a-iltalehti-2020-02-28_normalized_split.txt")

# yield lines one by one so that memory is not filled
class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __iter__(self):
        for i, line in enumerate(open(train_data_file, 'r', encoding='utf-8')):
            # assume there's one document per line, tokens separated by whitespace
            yield line.split()

            # if i > 1000: break

corpus_iterator = MyCorpus()
model = Word2Vec(sentences=corpus_iterator, size=100, window=5, min_count=5, workers=4, sg=1, negative=15)
# model.wv.save_word2vec_format("word2vec-test.txt", binary=False)
model.save("w2w_model.model")