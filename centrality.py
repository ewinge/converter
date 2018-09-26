#!/usr/bin/env python3
# encoding: UTF-8
from __future__ import print_function
from __future__ import division
import argparse
from operator import itemgetter
import itertools
import numpy
import graph_tools
from igraph import Graph, mean, load, InternalError
import nltk
from nltk.corpus import wordnet
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from gensim import models
import visualize
# from WSI import get_subgraph

# Global variables
graph = None


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="verbose output", action="store_true")
    parser.add_argument("-s", "--save", help="save graph plots", action="store_true")
    parser.add_argument("--arbitrary", help="break ties arbitrarily", action="store_true")
    parser.add_argument("infile", help="input file")
    parser.add_argument("-ties", help="Accept ties with this many members", type=int, default=1)
    return parser.parse_args()


def get_hyponyms(synset):
    """
    Get the WordNet hyponyms that are in the model
    """
    hyponyms = synset.hyponyms()
#     hyponym_terms = [hyponym.lemmas()[0].name() for hyponym in hyponyms]
#     hyponym_terms = [term for term in hyponym_terms if term in dictionary]
#     return hyponym_terms
    lemmas = set()
    for hyponym in hyponyms:
        # Use only one lemma  per hyponym
        lemmas.update(get_lemmas(hyponym)[:1])
    return lemmas


def get_lemmas(synset):
    """
    Get the lemmas of synset that are in the model
    """
    return [lemma.name() for lemma in synset.lemmas() if lemma.name() in dictionary]


def is_midfrequent(synset):
    lemmas = get_lemmas(synset)
    return lemmas and lemmas[0] in midfrequent


def get_rank(word):
    """
    Get the  frequency rank of the word  in the word embedding model
    """
    return model.vocab[word].index


def get_synsets(count):
    try:
        synsets = wordnet.all_synsets("n")
#         synsets = itertools.islice(wordnet.all_synsets("n"), 10000)
    except LookupError:
        nltk.download("wordnet")
        print("WordNet downloaded, please restart program")
        exit()
    hypernyms = [synset for synset in synsets if len(get_hyponyms(synset)) >= 5 and is_midfrequent(synset)]
    numpy.random.seed(1)
    print("hypernyms:", len(hypernyms))
#     return numpy.random.choice(hypernyms, count, replace=False)
    return hypernyms


def is_equal(list):
    # Convert to set to check that all entries are equal
    return len(set(list)) <= 1


def get_score(hypernym, centrality, max_ties):
    if max_ties <= 1:
        return hypernym == centrality[0][0]
    else:
        centrality_scores = [score for _, score in centrality[: max_ties]]
        candidates = [candidate for candidate, _ in centrality[: max_ties]]
        return (is_equal(centrality_scores) and hypernym in candidates) or get_score(hypernym, centrality, max_ties - 1)


def save_plot(method, graph, hypernym, center):
    graph.vs["color"] = "yellow"
    # include centrality in node label
    if "centrality" in graph.vs.attributes():
        graph.vs["label"] = ["%s (%d)" % (node["name"], node["centrality"]) for node in graph.vs]
    if hypernym == center:
        graph.vs.find(center)["color"] = "green"
        graph.vs.find(center)["shape"] = "diamond"
    else:
        if center:
            graph.vs.find(center)["color"] = "blue"
            graph.vs.find(center)["shape"] = "square"
        graph.vs.find(hypernym)["color"] = "red"
        graph.vs.find(hypernym)["shape"] = "up-triangle"
    visualize.plot_graph(graph, "temp/images/centrality/%s/%s.PDF" % (method, hypernym), clustering=False, layout="fr")  # or kk


def wordnet_subgraph(model, threshold, synset, method, plot=False):
    """
    Create  a local graph containing the synset lemma and its hyponyms from WordNet

    Params:
        model a gensim KeyedVectors model
        threshold for  edge inclusion
        synset a WordNet synset
    """
    global fully_connected, ties, graph
    hypernym = get_lemmas(synset)[0]
    hyponym_terms = get_hyponyms(synset)
#     print("synset: %s model contains  %d of %d  hyponyms" % (synset, len(hyponym_terms), len(synset.hyponyms())))
#     print(synset, term, hyponym_terms)
    hyponym_terms.add(hypernym)  # add hypernym
    graph = graph_tools.create_local_graph(model, threshold, hyponym_terms)
    centrality = get_centrality(method)
    center, center_centrality = centrality[0]
    score = get_score(hypernym, centrality, options.ties)
    if plot:
        # stats for  fully connected graphs
        if graph.density() == 1:
            fully_connected.append(score)
        else:
            # Stats excluding fully connected graphs
            excluding_fully_connected.append(score)
        # calculate baseline
        baseline_random.append(1 / len(hyponym_terms))
        most_frequent = sorted(hyponym_terms, key=get_rank)
#         print(list((word, get_rank(word)) for word in most_frequent))
        baseline_rank.append(hypernym == most_frequent[0])
        # make plot
        if options.save:
            save_plot(method, graph, hypernym, center)
        # log statistics
        words = [word for word, _ in centrality]
        try:
            index = words.index(hypernym)
            hypernym_centrality = centrality[index][1]
            difference = center_centrality - hypernym_centrality
#                 log("synset: %s model contains  %d of %d  hyponyms" % (synset, len(hyponym_terms), len(synset.hyponyms())))
            snip = centrality[:index + 2]
            log(hypernym, center, hypernym_centrality, center_centrality, difference, snip)
            if (hypernym != center and hypernym_centrality == center_centrality) or (hypernym == center and hypernym_centrality == centrality[1][1]):
                # tie
                #                 print("tie:", hypernym, center, hypernym_centrality, center_centrality, difference, snip)
                ties.append(score)
        except ValueError as e:
            pass
#             print("exception:", e)
    return score


def get_centrality(method):
    """
    Get centrality for all nodes in the graph

    Params:
        method - centrality measure to use
    """
    global graph
    if graph.ecount() == 0:
        # No edges
        return [(None, None)]
    # Invert weights, so similar words are closer
#     graph.es["weight"] = [1 - weight for weight in graph.es["weight"]]
    graph.es["weight"] = [1 / weight for weight in graph.es["weight"]]
    if method == "betweenness":
        #         graph.vs["centrality"] = graph.betweenness()
        graph.vs["centrality"] = graph.betweenness(weights="weight")
    elif method == "pagerank":
        graph.vs["centrality"] = graph.personalized_pagerank(weights="weight")
    elif method == "degree":
        graph.vs["centrality"] = graph.vs.degree()
    else:
        raise ValueError("Unsupported centrality measure: %s" % method)
    scores = [(node["name"], node["centrality"]) for node in graph.vs]
    if not options.arbitrary:
        # Sort by rank/frequency first
        scores = sorted(scores, key=lambda item: get_rank(item[0]))
    # Then sort by centrality
    scores = sorted(scores, key=itemgetter(1), reverse=True)
#     graph.vs["label"] = ["%s (%s)" % (node["name"], node["centrality"]) for node in graph.vs]
    return scores


def log(*args):
    print(*args, sep="\t", file=output)


def printtable(*args):
    print(*args, sep="\t", file=tableout)


if __name__ == '__main__':
    options = get_arguments()
    tiebreak = "arbitrary" if options.arbitrary else "rank"
    model_name = options.infile.split("/")[0]
    print("loading model %s from %s" % (options.infile, model_name))
    model = models.KeyedVectors.load_word2vec_format(options.infile, binary=False)
#     model = models.KeyedVectors.load_word2vec_format(options.infile, binary=False, limit=80000)
    dictionary = set(model.index2word)
    # Only get mid frequent terms
    midfrequent = set(model.index2word[1000:100000])
    synsets = get_synsets(150)
    figure = plot.figure(figsize=(8, 5.5))
    plot.axis(ymax=0.5)
    thresholds = numpy.linspace(0, 1, num=100)
#     for method in ["pagerank"]:
#     for method in ["betweenness"]:
    variant_description = "%s-%s-%s" % (model_name, tiebreak, options.ties)  # for logging
    tablefile = "../data/table-%s-%s.tsv" % (model_name, tiebreak)
    with open(tablefile, "w") as tableout:
        printtable("Model", "Centrality", "Accuracy", "Best-epsilon", "FC", "acc-FC", "acc-no-FC", "ties", "acc-ties")
        for method in ["pagerank", "degree", "betweenness"]:
            logfile = "../data/%s-%s.tsv" % (variant_description, method)
            with open(logfile, "w") as output:
                log("Hypernym", "Center", "Hypernym-Centrality", "Center-Centrality", "Difference", "Centrality")
                scores = []
                for threshold in thresholds:
                    score = [wordnet_subgraph(model, threshold, synset, method) for synset in synsets]
                    score = numpy.average(score)
                    scores.append(score)
                plot.plot(thresholds, scores, label=method)
                best_score = numpy.max(scores)
                best_threshold = thresholds[numpy.argmax(scores)]

                # Plot graphs for best threshold
                fully_connected = []
                excluding_fully_connected = []
                ties = []
                baseline_random = []
                baseline_rank = []
                for synset in synsets:
                    wordnet_subgraph(model, best_threshold, synset, method, plot=True)
                log("best  threshold: %.3f, score: %.3f\t\t\t\t\t" % (best_threshold, best_score))
                log("ties: %d, score ties: %.3f\t\t\t\t\t" % (len(ties), numpy.average(ties)))
                log("fully connected: %d, score fully connected: %.3f\t\t\t\t\t" % (len(fully_connected), numpy.average(fully_connected)))
                log("not fully connected: %d, score excluding fully connected: %.3f\t\t\t\t\t" % (len(excluding_fully_connected), numpy.average(excluding_fully_connected)))
                log("random baseline: %.3f\t\t\t\t\t" % (numpy.average(baseline_random)))
                log("most frequent baseline: %.3f\t\t\t\t\t" % (numpy.average(baseline_rank)))
                printtable(model_name, method, best_score, best_threshold, len(fully_connected), numpy.average(
                    fully_connected), numpy.average(excluding_fully_connected), len(ties), numpy.average(ties))
        # plot baselines
        plot.plot(thresholds, [numpy.average(baseline_random) for threshold in thresholds], linestyle="--", label="random baseline")
        plot.plot(thresholds, [numpy.average(baseline_rank) for threshold in thresholds], linestyle=":", label="frequency baseline")
        plot.xlabel("Threshold")
        plot.ylabel("Average score")
        plot.legend(title="Centrality measure")
        plot.savefig("temp/images/centrality-%s.pdf" % variant_description)
