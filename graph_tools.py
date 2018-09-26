#!/usr/bin/env python3
# encoding: UTF-8
from __future__ import print_function
import argparse
import itertools
from operator import itemgetter, attrgetter, methodcaller
import math
import numpy
import scipy
from igraph import Graph, mean, plot
import gensim
from gensim import corpora, models, matutils
from sklearn.decomposition import TruncatedSVD
import logging


def igraph_test():
    import igraph.test
    igraph.test.run_tests()
    exit()


def largest_connected_component(graph):
    """
    Extract a subgraph  containing the graphs largest connected component
    """
    components = graph.components()
    component_sizes = components.sizes()
    largest_index = numpy.argmax(component_sizes)
    return graph.subgraph(components[largest_index])


def get_subgraph(graph, center, radius=1):
    """
    Extract  a subgraph around center
    Params:
        graph an igraph Graph  object
        center label or id  of center node
        radius  (int) Maximum unweighted distance of neighbors to include
    """
    neighbors = {graph.vs.find(center).index}
    # Expand iteratively
    for i in range(radius):
        neighbors.update({next_neighbor for neighbor in neighbors for next_neighbor in graph.neighbors(neighbor)})
#         print(graph.vs[neighbors]["name"])
    subgraph = graph.subgraph(neighbors)
    # Labels for plotting
    subgraph.vs["label"] = subgraph.vs["name"]
    subgraph.vs["color"] = "yellow"
    subgraph.vs.find(center)["color"] = "red"
    return subgraph


def create_local_graph(model, threshold, terms):
    """
    Create  a local neighborhood graph from a VSM
    Params:
        model a gensim KeyedVectors model
        threshold to use  for edges
        terms to include in the graph
    """
    dictionary = model.index2word
    terms = [term for term in terms if term in dictionary]
    graph = Graph(len(terms))
    graph.vs["name"] = list(terms)
    graph.add_edges((term, term2) for (term, term2) in itertools.product(terms, repeat=2)  # if term < term2)
                    if term < term2 and model.similarity(term, term2) >= threshold)
    # Set edge weights
    graph.es["weight"] = 0  # default weight
    for edge in graph.es:
        term0 = graph.vs[edge.source]["name"]
        term1 = graph.vs[edge.target]["name"]
        #         print("%s-%s" % (term0, term1))
        edge['weight'] = model.similarity(term0, term1)
    #     print(graph.es["weight"])
    #     print(graph)
#     print(graph.summary())
    return graph
