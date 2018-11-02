#!/bin/sh
for var in "$@"
do
    echo "$var"
    java -jar WSI-Evaluator.jar ../ $var
    cp result.log $var.log
#    git add -f $var.log
#    git add -f $var
done