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
        parser.add_argument("infile", help="word2vec input file")
        parser.add_argument("mode", choices=["Threshold", "kNN", "Variable-k"], help="Algorithm to use")
    #     parser.add_argument("-out", help="output file name")
        parser.add_argument("-max", help="maximum number of embeddings to read", type=int, default=10000)
    #     parser.add_argument("-threshold", help="minimum similarity threshold", type=float, default=-2)
        return parser.parse_args()

    def VSM2graph_eval(self):
        if self.options.mode == "kNN":
            self.graph = VSM2graph(self.original_model, self.options.mode, k=self.parameter)
        elif self.options.mode == "Variable-k":
            self.graph = VSM2graph(self.original_model, self.options.mode, k=self.parameter)
        else:
            self.graph = VSM2graph(self.original_model, self.options.mode, threshold=self.parameter)

    def graph2VSM_eval(self):
        self.VSM = graph2VSM(self.graph)

    def reduction_eval(self):
        self.reduced = reduce_dimensions(self.VSM)

    def evaluate(self):
        gold_data = "word-sim/EN-SimLex-999.txt"
        parameters = []
        if self.options.mode == "kNN":
            #             parameters = [15, 20, 25, 30, 40, 50, 60, 70, 100, 150, 200, 300]
            if self.options.max > 20000:
                parameters = [25, 30, 40, 50, 60, 70, 100, 150, 200, 300, 400, 600, 1000]
            else:
                parameters = [15, 20, 25, 30, 40, 50, 60, 70, 100, 150, 200, 300]
        elif self.options.mode == "Variable-k":
            if self.options.max > 20000:
                parameters = [40, 60, 80, 100, 120, 160, 200, 240, 280, 400, 600, 1000]
            else:
                parameters = [20, 40, 60, 80, 100, 120, 160, 200, 240, 280]
        else:
            # threshold method
            if self.options.max <= 10000:
                parameters = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, -1]
            else:
                parameters = [0.55, 0.5, 0.45, 0.425, 0.4, 0.39, 0.38, 0.36, 0.35, 0.3, 0.25]
            if self.options.max > 80000:
                parameters = [0.35]
#                 parameters = [0.5]
#             parameters = [i / 10.0 for i in range(9, -1, -1)]
#             parameters[-1] = -1.0  # a few vectors might have negative similarity
#         print(parameters)
#         print("Running gensim evaluate_word_pairs. Results are Pearson/Spearman/unknown%")
        pearson, spearman, unknown = self.original_model.evaluate_word_pairs(gold_data)
        original_spearman = spearman[0]
#         print("original model:  %4.3f/%4.3f/%4.3f" % (pearson[0], spearman[0], unknown))
        print()
        print(r"\begin{table}[tp]")
#         print(r"\centering")
#         print(r"\footnotesize")
        print(r"\begin{adjustbox}{center}")
        print(r"\evaltable{")
#         print(r"&\multicolumn{3}{c}{Spearman $\rho$} & & & \multicolumn{3}{c}{Time used (s)}\\")
#         print(r"\cmidrule{2-4}")
#         print(r"\cmidrule{7-9}")
        if self.options.mode == "kNN":
            print(r"k &", end="")
        elif self.options.mode == "Variable-k":
            print(r"kmax &", end="")
        else:
            print(r"epsilon &", end="")
        print(r" rc & rcro & e & d & tv2g & tg2v & tr & m\\")
        for self.parameter in parameters:
            if self.options.mode == "Threshold":
                print("%-6.2f &" % self.parameter, end="")
            else:
                print("%-6d &" % self.parameter, end="")
            graph_time = timeit.Timer(self.VSM2graph_eval).repeat(repeat=1, number=1)
            VSM_time = timeit.Timer(self.graph2VSM_eval).repeat(repeat=1, number=1)
            reduction_time = timeit.Timer(self.reduction_eval).repeat(repeat=1, number=1)

            pearson, spearman, unknown = self.reduced.evaluate_word_pairs(gold_data)
            print("%6.6f & %6.6f &" % (spearman[0], spearman[0] / original_spearman), end="")
            print(r"%7.6e& " % self.graph.ecount(), end="")
            print(r"%7.6e&" % (self.graph.density()), end="")
            print(r"%7.3f& %7.3f& %7.3f& %5d\\" % (graph_time[0], VSM_time[0], reduction_time[0], self.process.memory_info().vms / (1024 * 1024)))
        print(r"}")
        print("\end{adjustbox}")
#         print(r"\caption*{%.1f\%% OOV, $\rho_{o} = %.3f$}" % (unknown, original_spearman))
#         print(r"\caption{%s method on %s terms from %s}" % (self.options.mode, self.options.max, self.options.infile))
        print(r"\caption{%s mode on %s terms, %.1f\%% OOV, $\rho_o = %.3f$}" % (self.options.mode, len(self.VSM.index2word), unknown, original_spearman))
        print(r"\label{tab:%s}" % self.label)
        print("\end{table}")

    def main(self):
        #     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.original_model = models.KeyedVectors.load_word2vec_format(self.options.infile, binary=False, limit=self.options.max)
        if self.options.save:
            filename = "%s.txt" % self.label
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
