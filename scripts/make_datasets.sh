#!/bin/bash
echo "Making vocabulary"
python -m img2latex.data.make_vocab data/raw/im2latex_formulas.norm.csv data/processed/ --add-special --col-name formulas

echo "Processing train"
python -m img2latex.data.preprocess data/raw/im2latex_train.csv ./data/interim/im2latex.train.csv --vocab ./data/processed/token2id.json --col-name formulas --add-padding

echo "Processing val"
python -m img2latex.data.preprocess data/raw/im2latex_validate.csv ./data/interim/im2latex.val.csv --vocab ./data/processed/token2id.json --col-name formulas --add-padding

echo "Processing test"
python -m img2latex.data.preprocess data/raw/im2latex_test.csv ./data/interim/im2latex.test.csv --vocab ./data/processed/token2id.json --col-name formulas --add-padding