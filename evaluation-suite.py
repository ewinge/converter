#!/usr/bin/env python3
# encoding: UTF-8
from __future__ import print_function
from __future__ import division
import logging
import argparse
import sys
import timeit
import os
import psutil
import numpy
from gensim import corpora, models, matutils
from convert import *


class Evaluator(object):
    def get_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--verbose", help="verbose output", action="store_true")
        parser.add_argument("-s", "--save", help="save output", action="store_true")
#         parser.add_argument("infile", help="word2vec input file")
        parser.add_argument("mode", choices=["Threshold", "kNN", "Variable-k"], help="Algorithm to use")
        parser.add_argument("directory", help="Directory containing gold standard evaluation data")
        parser.add_argument("-max", help="maximum number of embeddings to read", type=int, default=400000)
        return parser.parse_args()

    def evaluate(self):
        gold_files = sorted(os.listdir(self.options.directory))
#         gold_files = ["EN-SIMLEX-999.txt"]
        # Column headings
        output = "Dataset\t"
        for modelname in self.model_names:
            output += "%s-orig\t" % modelname
            output += "%s-conv\t" % modelname
        print(output.strip())
        # data
        for filename in gold_files:
            path = os.path.join(self.options.directory, filename)
            output = filename[3:-4] + "\t"
            for modelname in self.model_names:
                pearson, spearman, unknown = self.originals[modelname].evaluate_word_pairs(path)
                output += str(spearman[0]) + "\t"
                pearson, spearman, unknown = self.converted[modelname].evaluate_word_pairs(path)
                output += str(spearman[0]) + "\t"
            print(output.strip())

    def main(self):
        if self.options.verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#         self.model_names = ["skip-gram.txt"]
        self.model_names = ["skip-gram.txt", "fasttext.vec", "glove.txt"]
        self.originals = {}
        self.converted = {}
        if self.options.mode == "kNN":
            k = 600
        else:
            k = 60  # variable-k
        for name in self.model_names:
            self.originals[name] = models.KeyedVectors.load_word2vec_format("Gigaword/" + name, binary=False, limit=self.options.max)
            graph = VSM2graph(self.originals[name], self.options.mode, k=k)
            self.converted[name] = reduce_dimensions(graph2VSM(graph))
        if self.options.save:
            filename = "suite-%s.txt" % self.label
            with open(filename, "w") as outfile:
                sys.stdout = outfile
                self.evaluate()
        else:
            self.evaluate()

    def __init__(self):
        self.options = self.get_arguments()
        self.process = psutil.Process(os.getpid())
        self.label = "%s-%d" % (self.options.mode, self.options.max)


if __name__ == '__main__':
    Evaluator().main()
