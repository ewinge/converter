#!/bin/sh
#python3 evaluate.py Gigaword/skip-gram.txt kNN -s -max 5000
python3 evaluate.py Gigaword/skip-gram.txt kNN -s -max 10000
python3 evaluate.py Gigaword/skip-gram.txt kNN -s -max 15000
python3 evaluate.py Gigaword/skip-gram.txt kNN -s -max 20000
python3 evaluate.py Gigaword/skip-gram.txt kNN -s -max 40000
python3 evaluate.py Gigaword/skip-gram.txt kNN -s -max 80000
python3 evaluate.py Gigaword/skip-gram.txt kNN -s -max 400000
cat kNN-10000.txt kNN-15000.txt kNN-20000.txt kNN-40000.txt kNN-80000.txt kNN-400000.txt >  ../thesis/kNN.tex
#cp kNN.tex ../thesis/
#scp kNN.tex ewinge@sauron:inf/master/repo/thesis
