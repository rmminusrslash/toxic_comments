#!/bin/bash

echo "Downlaoding Glove embeddings vectors in variant 'Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download'"
curl -L http://nlp.stanford.edu/data/glove.twitter.27B.zip -o glove.twitter.27B.zip


echo "Downloading fasttext word embeddings in variant '2 million word vectors trained with subword information on Common Crawl (600B tokens)'."
curl https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip -o crawl-300d-2M-subword.zip  
