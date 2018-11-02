#!/usr/bin/env python3
# encoding: UTF-8
from __future__ import print_function
import matplotlib.pyplot as plot
import math
import sys
import numpy
import re


def plot_data(name_template, mode, algorithm, max_neighbors=0, **plotting_options):
    for neighbors in range(0, max_neighbors + 1, 5):
        data = {}
        for threshold in range(0, 56):
            ARI = get_ARI(name_template, threshold, neighbors, algorithm)
            if ARI:
                data[threshold / 100] = ARI
#             print(ARI)
        plot_ARI(data, neighbors, mode, max_neighbors, algorithm, **plotting_options)
#     plot.savefig(("../temp/images/ARI-%s-%s.pdf" % (algorithm, mode)).replace(" ", "-"))


def get_ARI(name_template, threshold, neighbors, algorithm):
    results = []
    for index in range(5):
        filename = name_template % (algorithm, threshold / 100, neighbors, index)
#         print(filename)
        try:
            with open(filename, "r") as input:
                for line in input:
                    match = re.match("average Adj Rand Index = (.*)", line)
                    if match:
                        results.append(float(match.group(1)))
        except FileNotFoundError as e:
            pass
    if results:
        #         print("threshold: %.2f, neighbors: %d, mean:%.4f, standard deviation: %.4f" % (threshold / 100, neighbors, numpy.mean(results), numpy.std(results)))
        #         print(results)
        return numpy.mean(results), numpy.std(results)


def plot_ARI(data, neighbors, mode, max_neighbors, algorithm, **plotting_options):
    #     plot.figure(figsize=(6.5, 4))
    x = [value for value in data.keys()]
    y = [value for value, deviation in data.values()]
    e = [deviation for value, deviation in data.values()]
    plot.xlabel("Threshold")
    plot.ylabel("Adjusted Rand Index")
    if max_neighbors > 0:
        plot.plot(x, y, label=neighbors)
        plot.legend(title="Neighbors")
    else:
        label = "%s %s" % (algorithm, mode)
        if algorithm == "HyperLex":
            plot.plot(x, y, label=label, **plotting_options)
        else:
            plot.errorbar(x, y, e, label=label, **plotting_options)
#         plot.plot(x, y, label=mode)
        plot.legend()
#     plot.title("Evaluation of %s based WSI" % (mode))
#     plot.savefig(("../temp/images/ARI-%s-%s.pdf" % (mode, neighbors)).replace(" ", "-"))
#     plot.show()
    print("Evaluation of %s %s based WSI" % (mode, algorithm))
    print("best ARI:%f, threshold: %f" % (numpy.max(y), x[numpy.argmax(y)]))


if __name__ == '__main__':
    figure = plot.figure(figsize=(6, 10))
    plot_data("output-multiple/Wikipedia_skip-gram-merged-220-%s-%.2f-%d-%d.tsv.log", "", "spinglass", linestyle=":", marker=".")
    plot_data("output-multiple/Wikipedia_wikipedia_merged_w2v_txt-%s-%.2f-%d-%d.tsv.log", "on-the-fly", "spinglass", linestyle=":", marker="x")
    plot_data("output-full/Wikipedia_wikipedia_merged_w2v_txt-%s-%.2f-%d-%d.tsv.log", "on-the-fly, full model", "spinglass", linestyle=":", marker="h")

    plot_data("output-HyperLex/Wikipedia_skip-gram-merged-220-%s-%.2f-%d-%d.tsv.log", "", "HyperLex", linestyle="-", marker=None)
    plot_data("output-HyperLex/Wikipedia_wikipedia_merged_w2v_txt-%s-%.2f-%d-%d.tsv.log", "on-the-fly", "HyperLex", linestyle="-.", marker=None)
    plot_data("output-full/Wikipedia_wikipedia_merged_w2v_txt-%s-%.2f-%d-%d.tsv.log", "on-the-fly, full model", "HyperLex", linestyle="--", marker=None)

#     plot_data("output/Wikipedia_skip-gram-merged-220-%s-%.2f-%d-%d.tsv.log", "20", "HyperLex", marker=None)
#     plot_data("output/Wikipedia_wikipedia_merged_w2v_txt-%s-%.2f-%d-%d.tsv.log", "20 on-the-fly", "HyperLex", marker=None)
#     plot.show()
    figure.tight_layout()
    plot.savefig("../temp/images/ARI-results.pdf")
