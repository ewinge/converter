#!/usr/bin/env python3
# encoding: UTF-8
from __future__ import print_function
import argparse
import itertools
from operator import itemgetter, attrgetter, methodcaller
import math
import numpy
import scipy
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackError
from igraph import Graph, mean, plot, load
import gensim
from gensim import corpora, models, matutils
from sklearn.decomposition import TruncatedSVD
import logging
from graph_tools import get_subgraph


def make_VSM(terms, matrix):
    """
    Make a KeyedVectors VSM from a matrix
    Params:
        terms list containing the vocabulary
        matrix numpy array containing the similarity matrix
    """
    model = models.KeyedVectors()
    model.syn0 = matrix
    model.index2word = terms
    for i, term in enumerate(model.index2word):
        model.vocab[term] = models.keyedvectors.Vocab(index=i)
    return model


def graph2VSM(graph):
    """
    Convert an igraph model to a VSM
    Params:
        model the igraph model
    """
    terms = graph.vcount()
    edges = graph.ecount()

    # Use sparse matrix representation if the matrix is less than 50% full
#     if terms > 20000 and graph.density() < 0.5:
    if graph.density() < 0.5:
        #         print("%d terms, using scipy.sparse" % terms)
        matrix = scipy.sparse.lil_matrix((terms, terms))
    else:
        #         print("%d terms, using numpy.zeros" % terms)
        matrix = numpy.zeros((terms, terms))

    # Entries on the diagonal indicate self similarity, and should be 1
    for i in range(terms):
        matrix[i, i] = 1.0

    # Get all edge weights from the graph, and add both directions
    for edge in graph.es:
        matrix[edge.source, edge.target] = edge["weight"]
        matrix[edge.target, edge.source] = edge["weight"]

#     print(matrix)
    # Make gensim corpus
#     return matutils.Dense2Corpus(matrix)

    model = make_VSM(graph.vs["name"], matrix)
    return model


def VSM2graph(model, mode, threshold=-1, k=None):
    """
    Convert a gensim KeyedVectors VSM to a graph model
    Params:
        model a gensim KeyedVectors model
        mode the conversion method to use: threshold, kNN or variable-k
        threshold (float) the minimum similarity required for an edge, used only with threshold method
        k (int) the number  of nearest neighbors that should get edges, used with the kNN methods
    """
    terms = model.index2word

    # Create nodes
    graph = Graph(len(terms))
    graph.vs["name"] = terms

    # cache for reciprocal kNN function
    cache = {}

    def get_kNN(term):
        if term not in cache:
            cache[term] = model.most_similar(term, topn=k)
        return cache[term]

    # Create edges
    if mode == "kNN":
        graph.add_edges((term, term2) for term in terms
                        for term2 in (pair[0] for pair in model.most_similar(term, topn=k)))
        # eliminate duplicate edges
        graph.simplify()
    elif mode == "reciprocal-kNN":
        graph.add_edges((term, term2) for term in terms
                        for term2 in (pair[0] for pair in get_kNN(term))
                        if term in (pair[0] for pair in get_kNN(term2)))
    elif mode == "Variable-k":
        graph.add_edges((term, term2) for index, term in enumerate(terms)
                        for term2 in (pair[0] for pair in model.most_similar(term,
                                                                             topn=math.ceil(k / math.log10(10 + index)))))
        graph.simplify()
    elif mode == "Threshold":
        # edges are undirected, only create for alphabetically ordered nodes
        graph.add_edges((term, term2) for term, term2 in itertools.product(terms, repeat=2)
                        if term < term2 and model.similarity(term, term2) >= threshold)
    else:
        raise ValueError("Unknown mode: %s" % mode)

    # Set weights
#     graph.es["weight"] = 0       # default weight
    for edge in graph.es:
        term0 = graph.vs[edge.source]["name"]
        term1 = graph.vs[edge.target]["name"]
        edge['weight'] = model.similarity(term0, term1)

    return graph


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="verbose output", action="store_true")
    parser.add_argument("infile", help="input model, word2vec or graphml")
    parser.add_argument("out", help="output file name")
    parser.add_argument("mode", choices=["Threshold", "kNN", "Variable-k"], help="Algorithm to use")
    parser.add_argument("-max", help="maximum terms", type=int, default=10000)
    parser.add_argument("-k", help="k or k-max", type=int, default=25)
    parser.add_argument("-threshold", help="minimum similarity threshold", type=float, default=-1)
    return parser.parse_args()


def reduce_dimensions(model, dimensions=300):
    """
    Reduce model dimensionality by Truncated SVD
    Params:
        model gensim keyedvectors vector space model
        dimensions (int) number of dimensions after reduction
    """
    matrix = model.syn0
    size = len(model.index2word)
    if size <= dimensions:
        # model already is smaller than the given dimensions
        return model
    try:
        SVD = TruncatedSVD(algorithm="arpack",  # arpack or randomized
                           n_components=dimensions,
                           n_iter=5,
                           random_state=None)
        truncated = SVD.fit_transform(matrix)
    except ArpackError:
        # Arpack failed, fall back to randomized
        SVD = TruncatedSVD(algorithm="randomized",  # arpack or randomized
                           n_components=dimensions,
                           n_iter=5,
                           random_state=None)
        truncated = SVD.fit_transform(matrix)
    return make_VSM(model.index2word, truncated)


if __name__ == '__main__':
    def verbose(*args):
        if options.verbose:
            print(*args)

    options = get_arguments()
    if options.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Which direction are we converting?
    if options.infile.endswith(".graphmlz") or options.infile.endswith(".graphml"):
        graph = load(options.infile)
        print("Graph summary:")
        print(graph.summary())
        VSM = graph2VSM(graph)
        reduced = reduce_dimensions(VSM)
        reduced.save_word2vec_format(options.out + ".txt")
        print("file saved: " + options.out + ".txt")
#         verbose("Simlex-999 evaluation, VSM:\n")
#         gold_data = "word-sim/EN-SimLex-999.txt"
#         reduced.evaluate_word_pairs(gold_data)
    else:
        original_model = models.KeyedVectors.load_word2vec_format(options.infile, binary=False, limit=options.max)
        graph = VSM2graph(original_model, options.mode, threshold=options.threshold, k=options.k)
        print("Graph summary:")
        print(graph.summary())
        try:
            graph.write(options.out + ".graphmlz")
            print("file saved: " + options.out + ".graphmlz")
        except SystemError:
            # full disk /tmp
            graph.write(options.out + ".graphml")
            print("file saved: " + options.out + ".graphml")
