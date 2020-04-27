# Finnish word embedding training and evaluation
ELEC-E5550 Statistical Natural Language Processing project

Group 21: Joel Lindfors, Yaroslav Getman, Anssi Moisio

## Instructions
Download the data (embeddings, corpora, evaluation data sets) from the course directory /work/courses/unix/T/ELEC/E5550/data/:
```
scp -r <username>@kosh.aalto.fi:/work/courses/unix/T/ELEC/E5550/data/ .
```
If you are on an Aalto computer, you can also link the directory.

Download pre-trained embeddings:
- YLE: https://data.ylestatic.fi/releases/20190509_yle_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin
- YLE + Wikipedia: https://data.ylestatic.fi/releases/20190509_yle-wikipedia_word2vec_cbow_fi_lr=0.05,dim=100,ws=5,epoch=5,neg=5,mincount=5.bin

Move embeddings to data/embeddings/

Download Yle corpus:
- https://korp.csc.fi/download/YLE/fi/2011-2018-src/ylenews-fi-2011-2018-src


Stats:
---
- fin-word2vec-lemma.bin:
    - vocab size: 2 208 293
    - dim: 300
- Iltalehti corpus:
    - tokens: 20 314 343
    - vocab size: 388 444
