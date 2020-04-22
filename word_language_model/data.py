import os
from io import open
import torch

class Dictionary(object):
    def __init__(self, w2v_model=None):
        self.word2idx = {}
        self.idx2word = []

        if w2v_model:
            # add the embedding vocab to this dictionary
            for word in list(w2v_model.wv.vocab.keys()):
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1

            print("vocab size:", len(list(w2v_model.wv.vocab.keys())))

    def add_word(self, word, oovs):
        
        if word not in self.word2idx:
            # print("OOV word:", word)
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            oovs += 1
        
        return self.word2idx[word], oovs


    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, w2v_model):
        self.dictionary = Dictionary(w2v_model)
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        
    def tokenize(self, path):
        oovs = 0
        
        # """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() # + ['<eos>']
                for word in words:
                    _, oovs =  self.dictionary.add_word(word, oovs)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() # + ['<eos>']
                ids = []
                for word in words:
                    try:
                        ids.append(self.dictionary.word2idx[word])
                    except KeyError:
                        # print("OOV word removed from corpus:", word)
                        pass
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
