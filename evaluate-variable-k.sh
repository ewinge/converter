#!/bin/sh
#python3 evaluate.py Gigaword/skip-gram.txt Variable-k -s -max 5000
python3 evaluate.py Gigaword/skip-gram.txt Variable-k -s -max 10000
python3 evaluate.py Gigaword/skip-gram.txt Variable-k -s -max 15000
python3 evaluate.py Gigaword/skip-gram.txt Variable-k -s -max 20000
python3 evaluate.py Gigaword/skip-gram.txt Variable-k -s -max 40000
python3 evaluate.py Gigaword/skip-gram.txt Variable-k -s -max 80000
python3 evaluate.py Gigaword/skip-gram.txt Variable-k -s -max 400000
cat Variable-k-10000.txt Variable-k-15000.txt Variable-k-20000.txt Variable-k-40000.txt Variable-k-80000.txt Variable-k-400000.txt >  ../thesis/Variable-k.tex
#cp Variable-k.tex ../thesis/
#scp Variable-k.tex ewinge@sauron:inf/master/repo/thesis
