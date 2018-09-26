#!/usr/bin/env python3
# encoding: UTF-8
from __future__ import print_function
import argparse
import itertools
from operator import itemgetter, attrgetter, methodcaller
import math
import numpy
import scipy
from igraph import Graph, mean, plot, InternalError
import logging
import visualize
import graph_tools


def get_subgraph(graph, term, context, max_neighbors=0):
    """
    Get the neighborhood of term in graph. Also, include all nodes
    corresponding to terms in the context.

    Params:
        graph, an igraph Graph object
        term, the term that is the center of the neighborhood
        context, list all terms to include in the subgraph
    Returns:
        subgraph as igraph Graph object
    """
    try:
        root = graph.vs.find(term).index
    except ValueError as e:
        print("OOV:", term, e)
        return
    nodes = {root}

    # Find max_neighbors most similar neighbors
    edges = graph.es[graph.incident(root)]
    edges = sorted(edges, key=itemgetter("weight"), reverse=True)[:max_neighbors]
    nodes.update(edge.source if edge.target == root else edge.target for edge in edges)
    print("number of neighbors:", len(nodes) - 1)
#     print(nodes)
    # add all context terms
    for word in context:
        try:
            nodes.add(graph.vs.find(word).index)
        except ValueError:
            pass
#             print("context word OOV:", word)
    graph = graph.subgraph(nodes)
    # Inject edges between root and context terms?
#     if max_neighbors == 0:
#         root = graph.vs.find(term).index
#         context = [node.index for node in graph.vs]
#         context.remove(root)
#         neighbors = graph.neighbors(root)
#         missing = [node for node in context if node not in neighbors]
#         for node in missing:
#             graph.add_edge(root, node, weight=0.1)
#         print("%d missing  edges added, previously %d neighbors" % (len(missing), len(neighbors)))
    # Labels for plotting
    graph.vs["label"] = graph.vs["name"]
    graph.vs["color"] = "red"
    graph.vs.find(term)["color"] = "yellow"
    return graph


def induce(graph, term, algorithm):
    if algorithm == "HyperLex":
        return HyperLex(graph, term)
    else:
        return IgraphClustering(graph, term, algorithm)


def induce_from_VSM(model, term, context=[], algorithm="HyperLex", threshold=0.4, max_neighbors=50):
    print("term:", term)
    print("max neighbors:", max_neighbors)
    try:
        if max_neighbors > 0:
            neighbors = set(neighbor for neighbor, score in model.most_similar(term, topn=max_neighbors))
        else:
            neighbors = set()
            # Query model to make sure term is not OOV
            if term not in model.vocab:
                raise KeyError(term, "OOV")
    except KeyError as e:
        print("OOV:", e)
        return induce(None, term, algorithm)
    neighbors.add(term)

    for word in context:
        if word in model.vocab:
            neighbors.add(word)
    print("neighbors:", len(neighbors))
    graph = graph_tools.create_local_graph(model, threshold, neighbors)
    return induce(graph, term, algorithm)


def induce_from_graph(graph, term, context=[], algorithm="leading_eigenvector", max_neighbors=50):
    # select nodes for subgraph
    graph = get_subgraph(graph, term, context, max_neighbors)
    return induce(graph, term, algorithm)


class IgraphClustering(object):
    def __init__(self, graph, term, algorithm="leading_eigenvector"):
        self.senses = []
        self.graph = graph
        self.term = term
        self.term_clusters = self.get_clusters(graph, algorithm)
#         print("Clustering induced  sense:", self.term_clusters)
        print("Clustered term:", term)

    def get_clusters(self, graph, algorithm):
        if graph is None:
            return
        else:
            graph = graph_tools.largest_connected_component(graph)
            print("largest connected component:", graph.summary())
            try:
                if algorithm == "spinglass":
                    clustering = graph.community_spinglass(weights="weight")
#                     clustering = graph.community_spinglass(weights=graph.es["weight"])
#                     clustering = graph.community_spinglass(spins=8)
                elif algorithm == "leading_eigenvector":
                    clustering = graph.community_leading_eigenvector()
                elif algorithm == "optimal_modularity":
                    clustering = graph.community_optimal_modularity()
                else:
                    raise ValueError("Unsupported clustering:", algorithm)
            except InternalError as e:
                print("Clustering error term", self.term, e)
                return
#             visualize.plot_graph(graph, "temp/images/%s.PDF" % self.term, clustering=clustering, width=2000, height=1500, layout="fr")
            return [[graph.vs[index]["name"] for index in cluster] for cluster in clustering]

    def disambiguate(self, term, context):
        """
        Use  the clustering to disambiguate the term.
        This method checks the number of occurrences of words
        from each cluster, and chooses the cluster with the most occurrences.
        """
        # sanity check
        if term != self.term:
            raise Exception("Incorrect term: %s != %s" % (self.term, term))
        if not self.term_clusters:
            return 0

        score = numpy.zeros(len(self.term_clusters))
        for index, cluster in enumerate(self.term_clusters):
            for word in context:
                if word in cluster:
                    score[index] += 1
        return numpy.argmax(score)


class HyperLex(object):
    def __init__(self, graph, term, min_degree=10):
        self.senses = []
        self.term = term
        self.min_degree = min_degree
        self.tree = self.build_tree(graph, term)

    def get_distance(self, path, tree):
        distance = 0
        for index in range(len(path) - 1):
            source = path[index]
            target = path[index + 1]
            edge = tree.get_eid(source, target)
            distance += tree.es[edge]["weight"]
        return distance

    def build_tree(self, graph, term):
        """
        Do word sense induction on the term, based on the graph
        Params:
            graph an igraph graph
            term to undergo WSI
        """
        if not graph:
            return

#         mean_degree = mean(graph.degree())
#         max_degree = max(graph.degree())
#         self.min_degree = max(max_degree * 0.2, mean_degree * 1, 10)
#         self.min_degree = max(mean_degree * 1, 10)
        print("HyperLex hub minimum degree:", self.min_degree)
#         graph = graph_tools.largest_connected_component(graph)
        # HyperLex algorithm
        graph.vs["used"] = [False for node in graph.vs]
        root = graph.vs.find(term)
        root["used"] = True
#         plot(graph, layout="kk")
        # Remove all edges from root
        graph.delete_edges((root, neighbor) for neighbor in graph.neighbors(root))

        # HyperLex  uses inverse weights
        graph.es["weight"] = [1 - weight for weight in graph.es["weight"]]

        # Find senses
        sense_neighbors = []
        candidates = sorted([vertex for vertex in graph.vs if vertex.degree() >= self.min_degree], key=methodcaller("degree"), reverse=True)
        for hub in candidates:
            if hub["used"]:
                #                 print("used:", hub["label"])
                continue  # skip used vertex
            sense_neighbors.append([graph.vs[neighbor]["name"] for neighbor in graph.neighbors(hub)])
            self.senses.append(hub["name"])
            graph.add_edge(root, hub, weight=0)
            hub["used"] = True
            graph.vs[graph.neighbors(hub)]["used"] = True

        tree = graph.spanning_tree(weights=[weight for weight in graph.es["weight"]])

        # create score vectors
        for vertex in tree.vs:
            path = tree.get_shortest_paths(root.index, vertex)
            path = path[0][1:]  # flatten, remove root
            if path:
                sense = path[0]
                sense_label = tree.vs[sense]["name"]
                vector = numpy.zeros(len(self.senses))
                distance = self.get_distance(path, tree)
                vector[self.senses.index(sense_label)] = 1 / (1 + distance)
                vertex["score"] = vector
            else:
                vertex["score"] = None
#         tree.vs["label"] = tree.vs["score"]
#         layout = graph.layout("large", root=root.index)
#         plot(tree, layout=layout, bbox=(1200, 1000), margin=20)
        print("HyperLex induced %d senses for %s" % (len(self.senses), term))
        for index, sense in enumerate(self.senses):
            pass
#             print("sense:", sense, "nodes:", sense_neighbors[index])
        return tree

    def disambiguate(self, term, context):
        """
        Use  the HyperLex tree to disambiguate the term
        """
#         print("disambiguate:", context)

        # sanity checks
        if term != self.term:
            raise Exception("Incorrect term: %s != %s" % (self.term, term))
        if self.tree is None:  # term OOV, no tree constructed
            return 0
        if len(self.senses) == 0:  # no senses found
            return 0

        total = numpy.zeros(len(self.senses))
        for word in context:
            try:
                vertex = self.tree.vs.find(word)
                score = vertex["score"]
#                 print("word:", word, "total:", total, "score:", score)
                if score is not None:
                    #                     print("Term found:", word)
                    total += score
                else:
                    pass
#                     print("error: score not found:", word)
            except ValueError:
                pass
#                 print("error: context term not found:", word)
        sense = numpy.argmax(total)
#         print(total)
        return sense
#         return "%s#%d" % (term, sense)


if __name__ == '__main__':
    import convert
