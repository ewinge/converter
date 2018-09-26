#!/usr/bin/env python3
# encoding: UTF-8
from __future__ import print_function
import argparse
import itertools
from operator import itemgetter, attrgetter, methodcaller
import math
import numpy
import scipy
from igraph import Graph, plot, load
import igraph
import gensim
from gensim import corpora, models, matutils
import logging
from graph_tools import get_subgraph
from convert import VSM2graph, graph2VSM


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="verbose output", action="store_true")
    parser.add_argument("mode", choices=["Threshold", "kNN", "Variable-k", "file"], help="Algorithm to use")
    parser.add_argument("infile", help="input file")
    parser.add_argument("-max", help="maximum dimensions", type=int, default=2000)
    parser.add_argument("-k", help="k or k-max", type=int, default=25)
    parser.add_argument("-threshold", help="minimum similarity threshold", type=float, default=0.4)
    return parser.parse_args()


def plot_neighborhood(graph, term, outFile, radius=1, width=1200, height=1000, clustering=True, display=False):
    """
    Create a graphic plot of the neighborhood centered on term
    """
    graph = get_subgraph(graph, term, radius)
    plot_graph(graph, outFile, width, height, clustering, display)


def plot_graph(graph, outFile, width=1200, height=800, clustering=True, display=False, layout="grid_fr"):
    """
    Create a graphic plot  of the graph
    Params:
        graph, igraph Graph or VertexClustering object
        outFile,  string containing filename for storing results
    """
    # Remove duplicate edges
#     graph = graph.simplify()
#     layout = graph.layout("large")  # root=root.index)
    if not "label" in graph.vs.attributes():
        graph.vs["label"] = graph.vs["name"]
    layout = graph.layout(layout)
    target = None if display else outFile
    args = {"layout": layout, "bbox": (width, height), "margin": 80, "target": target, "vertex_label_dist": 1.5, "edge_color": "gray", "vertex_label_size": 20}
    if clustering:
        if isinstance(clustering, igraph.Clustering):
            output = plot(clustering, **args)
        else:
            clusters = graph.community_spinglass()
#         clusters = graph.community_optimal_modularity()
#         clusters = graph.community_multilevel(return_levels  =  False)
#         clusters = graph.community_label_propagation()
#         clusters = graph.community_leading_eigenvector()
#         clusters = graph.community_infomap()
#         clusters = graph.community_walktrap()  # hierarchic
#         clusters = graph.community_edge_betweenness(directed=False)#hierarchic
            output = plot(clusters, **args)
    else:
        output = plot(graph, **args)
    if display:
        output.save(outFile)


def main():
    options = get_arguments()
    if options.mode == "file":
        graph = load(options.infile)
    else:
        original_model = models.KeyedVectors.load_word2vec_format(options.infile, binary=False, limit=options.max)
        graph = VSM2graph(original_model, options.mode, threshold=options.threshold, k=options.k)
    print(graph.summary())
    print("graph density:", graph.density())
    degree = mean(graph.degree())
    print("mean degree:", degree)
    print("clustering coefficient:", graph.transitivity_undirected(), "expected(random):", 2 * degree / graph.vcount())
#     print("average path length:", graph.average_path_length(), "expected (random):", math.log(graph.vcount()) / math.log(degree))
    for term in "apple fire line queen".split():
        plot_neighborhood(graph, term, clustering=True)
#         plot_neighborhood(graph, term, clustering=True, display=True)


if __name__ == '__main__':
    for mode in ["threshold", "kNN"]:
        if mode == "threshold":
            graph = load("Wikipedia/skip-gram-threshold-10.graphmlz")
        else:
            graph = load("Wikipedia/skip-gram-kNN-K25-10000.graphmlz")

        for term in "apple fire queen".split():
            outFile = "temp/images/%s-%s.PDF" % (term, mode)
            plot_neighborhood(graph, term, outFile, height=800)
        # smaller plots
        for term in "line".split():
            outFile = "temp/images/%s-%s.PDF" % (term, mode)
            plot_neighborhood(graph, term, outFile, height=600)
