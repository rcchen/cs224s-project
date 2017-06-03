#!/bin/bash

# This file will contain scripts for downloading the train, dev, and test data
# needed to train and evaluate the models.

# Glove vectors ((2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download)

DATASETS_DIR="var/data/vocab"

mkdir -p $DATASETS_DIR

cd $DATASETS_DIR

curl http://nlp.stanford.edu/data/glove.twitter.27B.zip

if hash wget 2>/dev/null; then
  wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
else
  curl -O http://nlp.stanford.edu/data/glove.twitter.27B.zip
fi
unzip glove.twitter.27B.zip
rm glove.twitter.27B.zip