'''
Mostly from:
https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
'''
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import datapath
from gensim import utils
import os


train_data_file = os.path.join("data", "corpora", "iltalehti-2020-02-28_wikipedia.txt")
# train_data_file = os.path.join("data", "corpora", "test.txt")
save_file = os.path.join("data", "embeddings", "iltalehti-wikipedia-dim=200,window=5,mincount=5,sg=1,negative=5.model")

# yield lines one by one so that memory is not filled
class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __iter__(self):
        for i, line in enumerate(open(train_data_file, 'r', encoding='utf-8')):
            # assume there's one document per line, tokens separated by whitespace
            yield line.split()
            # if i > 1000: break

corpus_iterator = MyCorpus()

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1

## default values
# sentences=None,
# corpus_file=None,
# size=100,
# alpha=0.025, 
# window=5,
# min_count=5,
# max_vocab_size=None,
# sample=0.001,
# seed=1,
# workers=3,
# min_alpha=0.0001,
# sg=0,
# hs=0,
# negative=5,
# ns_exponent=0.75,
# cbow_mean=1,
# hashfxn=<built-in function hash>,
# iter=5,
# null_word=0,
# trim_rule=None,
# sorted_vocab=1,
# batch_words=10000,
# compute_loss=False,
# callbacks=(),
# max_final_vocab=None
model = Word2Vec(
                sentences=corpus_iterator,
                size=200,
                alpha=0.025,
                window=5,
                min_count=5,
                workers=4,
                sg=1,
                negative=5,
                compute_loss=True,
                callbacks=[callback()],
            )
# model.wv.save_word2vec_format("word2vec-test.txt", binary=False)
model.save(save_file)