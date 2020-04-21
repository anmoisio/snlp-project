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

# yield lines one by one so that memory is not filled
class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, train_data_file):
        self.train_data_file = train_data_file
    
    def __iter__(self):
        for i, line in enumerate(open(self.train_data_file, 'r', encoding='utf-8')):
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

def train_model(model_type, arguments, model_file, train_data):
    corpus_iterator = MyCorpus(train_data)

    print("Train "+model_type+" model using corpus", train_data)
    print("with parameters:")
    for param, arg in arguments.items():
        print(param, arg)
    
    if model_type == 'Word2Vec':
        model = Word2Vec(
                        sentences = corpus_iterator,
                        size =      arguments['size'],
                        alpha =     arguments['alpha'],
                        window =    arguments['window'],
                        min_count = arguments['min_count'],
                        workers =   arguments['workers'],
                        sg =        arguments['sg'],
                        negative =  arguments['negative'],
                        iter =      arguments['iter'],
                        compute_loss=True,
                        callbacks = [callback()],
                    )
        model.wv.save_word2vec_format(model_file, binary=True)

    elif model_type == 'FastText':
        model = FastText(
                        sentences = corpus_iterator,
                        size =      arguments['size'],
                        alpha =     arguments['alpha'],
                        window =    arguments['window'],
                        min_count = arguments['min_count'],
                        workers =   arguments['workers'],
                        sg =        arguments['sg'],
                        negative =  arguments['negative'],
                        iter =      arguments['iter'],
                        min_n =     arguments['min_n'],
                        max_n =     arguments['max_n'],
                    )
        model.wv.save(model_file, binary=True)

    print("Model trained. Saved in file", model_file)
    print("Vocab size:", len(model.wv.vocab))

    return model