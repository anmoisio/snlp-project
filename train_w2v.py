'''
Mostly from:
https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
'''
from gensim.models import Word2Vec, FastText
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import datapath
from gensim import utils
import os
import config

# yield lines one by one so that memory is not filled
class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __iter__(self):
        for i, line in enumerate(open(config.train_data_file, 'r', encoding='utf-8')):
            # assume there's one document per line, tokens separated by whitespace
            yield line.split()
            # if i > 1000: break

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

corpus_iterator = MyCorpus()
def train_model():
    print("Train model using corpus", config.train_data_file)
    print("with parameters:")
    for param, arg in config.arguments.items():
        print(param, arg)

    model = Word2Vec(
                    sentences = corpus_iterator,
                    size =      config.arguments['size'],
                    alpha =     config.arguments['alpha'],
                    window =    config.arguments['window'],
                    min_count = config.arguments['min_count'],
                    workers =   config.arguments['workers'],
                    sg =        config.arguments['sg'],
                    negative =  config.arguments['negative'],
                    compute_loss=True,
                    callbacks = [callback()],
                )

    model.wv.save_word2vec_format(config.model_file, binary=config.binary)
    print("Model trained. Saved in file", config.model_file)
    print("Vocab size:", len(model.wv.vocab))

    return model

def train_fasttext_model():
    print("Train model using corpus", config.train_data_file)
    print("with parameters:")
    for param, arg in config.arguments.items():
        print(param, arg)

    model = FastText(
                    sentences = corpus_iterator,
                    size =      config.arguments_fasttext['size'],
                    alpha =     config.arguments_fasttext['alpha'],
                    window =    config.arguments_fasttext['window'],
                    min_count = config.arguments_fasttext['min_count'],
                    workers =   config.arguments_fasttext['workers'],
                    sg =        config.arguments_fasttext['sg'],
                    negative =  config.arguments_fasttext['negative'],
                    min_n =     config.arguments_fasttext['min_n'],
                    max_n =     config.arguments_fasttext['max_n'],
                )

    model.wv.save(config.model_file)#, binary=config.binary)
    print("Model trained. Saved in file", config.model_file)
    print("Vocab size:", len(model.wv.vocab))

    return model