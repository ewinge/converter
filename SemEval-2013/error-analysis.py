#!/usr/bin/python3
from operator import itemgetter
import matplotlib.pyplot as plot
import math
import sys
import numpy
from sklearn.metrics import adjusted_rand_score as ari


def read_data(inputfile):
    data = {}
    for line in open(inputfile, 'r').readlines():
        if line.strip()[0].isdigit():
            (subtopic, result) = line.strip().split('\t')
#             data[result] = float(subtopic)
            data[result] = subtopic
    return data


def histogram(data, label):
    plot.figure(figsize=(6.5, 4))
    plot.hist(data)
    plot.xlabel("# clusters")
    plot.ylabel("frequency")
    plot.title(label)
    print(label, "mean:", numpy.mean(data))
    plot.axvline(numpy.mean(data), color='black', linestyle='dashed', linewidth=2, label="Mean")
    plot.legend()
    plot.tight_layout()
    plot.savefig(("../temp/images/%s.pdf" % label).replace(" ", "-"))
#     plot.show()


# goldfile = sys.argv[1]
# predfile = sys.argv[2]
goldfile = "STRel.txt"
gold = read_data(goldfile)

HyperLex = read_data("output-HyperLex/Wikipedia_wikipedia_merged_w2v_txt-HyperLex-0.36-0-0.tsv")
spinglass = read_data("output-multiple/Wikipedia_wikipedia_merged_w2v_txt-spinglass-0.36-0-0.tsv")
# spinglass = read_data("output/Wikipedia_wikipedia_merged_w2v_txt-spinglass-0.40.tsv")
# spinglass = read_data("output/Wikipedia_wikipedia_merged_w2v_txt-spinglass-complete.tsv")
# spinglass = read_data("output/Wikipedia_skip-gram-merged-80-spinglass-0.40.tsv")

results = list(gold.keys())
scores = []
# print(results[0])
gold_count = []
HyperLex_count = []
spinglass_count = []

for index in range(1, 101):
    goldtopics = [gold[r] for r in results if math.floor(float(r)) == index]
    hyper_topics = [HyperLex[r] for r in results if math.floor(float(r)) == index]
    spinglass_topics = [spinglass[r] for r in results if math.floor(float(r)) == index]
    gold_count.append(len(set(goldtopics)))
    HyperLex_count.append(len(set(hyper_topics)))
    spinglass_count.append(len(set(spinglass_topics)))
#     print(hyper_topics)
#     print(goldtopics)
#     exit()
#     total_clusters.append(max(int(number.split(".")[1]) + 1 for number in hyper_topics))
    scores.append((index, ari(goldtopics, spinglass_topics)))

# HyperLex_count = [count for count in HyperLex_count if count > 1]
# spinglass_count = [count for count in spinglass_count if count > 1]
histogram(gold_count, "Gold standard")
histogram(HyperLex_count, "HyperLex")
histogram(spinglass_count, "Spinglass")
# histogram(total_clusters, "total")
print(scores)
print(gold_count)
print(HyperLex_count)
# print(total_clusters)
# print("ARI:", numpy.average(scores))
scores = sorted(scores, key=itemgetter(1))
topics = numpy.genfromtxt("topics.txt", dtype=None, skip_header=1)
print(scores)
# Top 5
for id, score in scores[-1:-6:-1]:
    index = id - 1
    topic = topics[index]
#     print("%s &  %d &  %d  &  %.3f" % (topic[1], topic[0], id, score))
    print(r"%s  &  %d  &  %.3f& %d & %d\\" % (topic[1].decode().replace("_", "::"), id, score, spinglass_count[index], gold_count[index]))

print("\nBottom 5")
for id, score in scores[: 5]:
    index = id - 1
    topic = topics[index]
#     print("%s &  %d &  %d  &  %.3f" % (topic[1], topic[0], id, score))
    print(r"%s  &  %d  &  %.3f& %d & %d\\" % (topic[1].decode().replace("_", "::"), id, score, spinglass_count[index], gold_count[index]))
