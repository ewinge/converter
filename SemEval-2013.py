#!/usr/bin/env python3
# encoding: UTF-8
from __future__ import print_function
import argparse
import math
import time
import numpy
from lxml import etree
from igraph import Graph, mean, load
# from html2text import html2text
# from StringIO import StringIO
from WSI import induce_from_VSM, induce_from_graph
from pynlp import StanfordCoreNLP
from gensim import models
import string
"""Script for evaluating graph-based word sense induction"""

# Stanford CoreNLP setup
annotators = "tokenize, ssplit, pos, lemma, ner"
nlp = StanfordCoreNLP(annotators=annotators)
annotated = {}


def list_plaintext(list):
    return [get_plaintext(item) for item in list]


def get_plaintext(text):
    return text.replace("<b>", "").replace("</b>", "").replace("...", "").strip()


def fix_NER_glue(text):
    """
    SemEval-2013 uses _ as glue, while our corpus uses ::
    Example:
    >>> fix_NER_glue("some_name")
    'some::name'
    """
    return text.replace("_", "::")


def is_punctuation(term):
    """
    Check whether a string consists of a single punctuation mark
    Example:
    >>> is_punctuation(".")
    True
    >>> is_punctuation("word")
    False
    """
    return len(term) == 1 and term in string.punctuation


def annotate_document(text):
    """
    Tokenize and annotate document  using Stanford CoreNLP
    Results are cached for further use
    """
    if text not in annotated:
        annotated[text] = nlp(text)
    return annotated[text]


def process_topic(id, term, algorithm, threshold, max_neighbors):
    filename = "SemEval-2013/2010/%s.n.xml" % term
    path = "/dataSet/%s.n.test" % term
    data = etree.parse(filename)
    contexts = []
    for i in range(1, 65):
        item_path = path + "/%s.n.%d" % (term, i)
        # title and snippet are concatenated
        title = data.xpath(item_path + "/@title")
        title = title[0] if title else ""
        snippet = data.xpath(item_path + "/TargetSentence/text()")
        # some target sentences are empty
        snippet = snippet[0] if snippet else ""
        context = title + " " + snippet
        contexts.append(context)
#     print(term, len(contexts), context)
    term = fix_NER_glue(term)
    contexts = list_plaintext(contexts)
    # Tokenize and lemmatize  with CoreNLP
    all_tokens = set()  # Set of all tokens occurring in all contexts
    contexts_lemmas = []
    for context in contexts:
        document = annotate_document(context)
        current = [token.lemma.lower() for sentence in document
                   for token in sentence if not is_punctuation(token.lemma)]
        contexts_lemmas.append(current)
        all_tokens.update(current)
#     print(all_tokens)
    if mode == "graph":
        senses = induce_from_graph(graph, term, all_tokens, algorithm, max_neighbors)
    else:
        senses = induce_from_VSM(model, term, all_tokens, algorithm, threshold, max_neighbors)

    for index, context in enumerate(contexts_lemmas):
        sense = senses.disambiguate(term, context)
        # subTopicID    resultID
        print("%d.%d\t%d.%d" % (id, sense + 1, id, index + 1), file=output)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="verbose output", action="store_true")
    parser.add_argument("infile", help="input file")
#     parser.add_argument("clustering", help="clustering algorithm to use")
    parser.add_argument("min", help="minimum similarity threshold", type=int, default=30)
    parser.add_argument("max", help="maximum similarity threshold", type=int, default=56)
    parser.add_argument("-step", help="similarity threshold step size", type=int, default=2)
    parser.add_argument("-n", help="maximum number of embeddings to read", type=int, default=220000)
#     parser.add_argument("-m", "--max_neighbors", help="max #neighbors in WSI", type=int, default=50)
    return parser.parse_args()


def remove_edges(graph, threshold):
    "Delete all edges below threshold from graph"
    edges = graph.es.select(weight_lt=threshold)
    print("removing edges:", len(edges))
    edges.delete()
    return graph


if __name__ == '__main__':
    options = get_arguments()
    filename = "_".join(options.infile.replace("/", "_").split(".")[:-1])

    if options.infile.endswith(".graphmlz"):
        graph = load(options.infile)
        mode = "graph"
#         if options.threshold:
#             graph = remove_edges(graph, options.threshold)
    else:
        model = models.KeyedVectors.load_word2vec_format(options.infile, binary=False, limit=options.n)
        print("Loaded %d embeddings from %s" % (len(model.index2word), options.infile))
        mode = "VSM"
    # Get topics
    topics = numpy.genfromtxt("SemEval-2013/topics.txt", dtype=None, skip_header=1)
#     for algorithm in ["HyperLex", "leading_eigenvector", "spinglass"]:
#     for algorithm in ["spinglass"]:
#     for algorithm in ["HyperLex"]:
    for algorithm in ["HyperLex", "spinglass"]:
        for threshold in range(options.min, options.max, options.step):
            threshold = threshold / 100
            if mode == "graph":
                graph = remove_edges(graph, threshold)

            for max_neighbors in range(1):
                for i in range(5 if algorithm == "spinglass" else 1):  # spinglass is stochastic, use mean ARI of 5 runs
                    starttime = time.time()
                    with open("SemEval-2013/output/%s-%s-%.2f-%d-%d.tsv" % (filename, algorithm, threshold, max_neighbors, i), "w") as output:
                        print("subTopicID\tresultID", file=output)
                        for topic in topics:
                            id = topic[0]
                            term = topic[1].decode()
                            process_topic(id, term, algorithm, threshold, max_neighbors)
                    endtime = time.time()
                print("%s algorithm with threshold %.2f completed in %f seconds" % (algorithm, threshold, endtime - starttime))
