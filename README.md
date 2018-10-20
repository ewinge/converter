# Tools for converting lexical semantic representations

This project contains software written for my MSc thesis "Word
embedding models as graphs: conversion and evaluation".  The purpose
of this software is to convert between graph and vector lexical
semantic representations, and to evaluate the results.

## Evaluation

For evaluation, we use gensim with gold standard data files from the
wordvectors.org evaluation suite. These are found at
<https://github.com/mfaruqui/eval-word-vectors>.  Three of these data
files use spaces as field separators. This must be changed to tabs, to
work with gensim.

## Word embedding models

Our software has been tested with word embeddings from
<http://vectors.nlpl.eu/repository/>.  We have mainly used model 11.

## convert.py

Lexical semantic model conversion script. Takes as input a model in
either word2vec or graphml format. Outputs the model converted to the
opposite format.

## evaluate.py

Script for evaluating the conversion results. Takes as parameter a
word2vec word embedding model. This model is converted to a graph
model, and then back to a word embedding model. The resulting word
embeddings are evaluated with SimLex-999. The script expects to find
the gold standard data files from
<https://github.com/mfaruqui/eval-word-vectors> in the directory
`word-sim`.

Example command line, using the threshold method:

`python3 evaluate.py Gigaword/skip-gram.txt Threshold -s -max 10000`

This requires an unzipped word embedding model at the given path.  The
evaluation results are stored in a text file.

## evaluation-suite.py

Script for evaluating the conversion using all data sets from the
wordvectors.org evaluation suite.  Example command, evaluating the kNN
method with data sets in the directory `word-sim`:

`python3 evaluation-suite.py kNN word-sim/ -s`

With large models, this might take some hours to complete.  The
results are stored in a text file with a name like
suite-kNN-400000.txt.

## SemEval-2013.py

Evaluation script for performing and evaluating word sense induction
based on SemEval-2013 task 11.  Can be used with either a word
embedding model or a graph model.  Example command line: `python3
SemEval-2013.py Wikipedia/wikipedia_merged_w2v.txt.gz 20 56 -n 10000`
This runs the evaluation with thresholds from 0.20 to 0.56, limiting
the model to the first 10,000 word

## WSI.py

Contains methods for doing word sense induction based on the HyperLex
or spinglass graph clustering algorithms.

## centrality.py

Script for evaluating converted graphs by performing hypernym
discovery in WordNet graphs.

`python centrality.py gigaword/skip-gram.txt`

## visualize.py

Helper script for plotting visualizations of graph models.

## graph_tools.py

Some functions for working with graphs


