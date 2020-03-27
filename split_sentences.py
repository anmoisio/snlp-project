import os

with open(os.path.join('data', 'corpora', 'a-iltalehti-2020-02-28_normalized.txt'), 'r', encoding='utf-8') as f:
    corpus = f.read().split('.')

with open(os.path.join('data', 'corpora', 'a-iltalehti-2020-02-28_normalized_split.txt'), 'w', encoding='utf-8') as f:
    for line in corpus:
        f.write(line)
        f.write('\n')