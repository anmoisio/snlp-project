# ELEC-E5550 Statistical Natural Language Processing project: word2vec embeddings
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