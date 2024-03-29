#!/bin/bash
mkdir data
mkdir data/train
mkdir data/test
mkdir data/valid

FILES=data_uncompressed/train_chunk*
for f in $FILES
do
  echo "Unzipping $f file..."
  unzip -j $f -d data/train
done

unzip test.zip -d data/test
unzip valid.zip -d data/valid
