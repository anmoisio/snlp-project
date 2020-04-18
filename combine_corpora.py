import os

file1 = os.path.join("data", "corpora", "a-iltalehti-2020-02-28_normalized_split.txt")
file2 = os.path.join("data", "corpora", "wikipedia2008_fi_lemmatized.txt")
combined = os.path.join("data", "corpora", "iltalehti-2020-02-28_wikipedia.txt")

with open(file1, 'r', encoding='utf-8') as f:
    corpus1 = f.read()

with open(file2, 'r', encoding='utf-8') as f:
    corpus2 = f.read()

with open(combined, 'w', encoding='utf-8') as f:
    f.write(corpus1)
    f.write('\n')
    f.write(corpus2)