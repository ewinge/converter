#!/bin/sh
#python3 evaluate.py Gigaword/skip-gram.txt Threshold -s -max 5000
python3 evaluate.py Gigaword/skip-gram.txt Threshold -s -max 10000
python3 evaluate.py Gigaword/skip-gram.txt Threshold -s -max 15000
python3 evaluate.py Gigaword/skip-gram.txt Threshold -s -max 20000
python3 evaluate.py Gigaword/skip-gram.txt Threshold -s -max 40000
python3 evaluate.py Gigaword/skip-gram.txt Threshold -s -max 80000
python3 evaluate.py Gigaword/skip-gram.txt Threshold -s -max 400000
cat Threshold-10000.txt Threshold-15000.txt Threshold-20000.txt Threshold-40000.txt Threshold-80000.txt Threshold-400000.txt >  ../thesis/threshold.tex
#cat Threshold-10000.txt Threshold-15000.txt Threshold-20000.txt Threshold-40000.txt Threshold-80000.txt  >  ../thesis/threshold.tex
#cp threshold.tex ../thesis/
#scp threshold.tex ewinge@sauron:inf/master/repo/thesis
