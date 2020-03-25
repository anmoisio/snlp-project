from gensim.models import Word2Vec
import os

train_data_file = os.path.join("data", "corpora", "a-iltalehti-2020-02-28_normalized.txt")


def read_in_chunks(file_object, chunk_size=1024):
    """
    https://stackoverflow.com/questions/519633/lazy-method-for-reading-big-file-in-python
    Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k.
    """
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


with open(train_data_file, 'r') as f:
    for piece in read_in_chunks(f):
        process_data(piece)


model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)

model.save("word2vec-test.model")