#!/bin/bash

echo "Solving Problem 1 with all the search algorithms"
pypy3 run_search.py -p 1 -s 1 2 3 4 5 6 7 8 9 10 11 > results/results_p1.txt

echo "Solving Problem 2 with all the search algorithms"
pypy3 run_search.py -p 2 -s 1 2 3 4 5 6 7 8 9 10 11 > results/results_p2.txt

echo "Solving Problem 3 on a one-by-one basis"
echo "s 1"
pypy3 run_search.py -p 3 -s 1 >> results/results_p3.txt
echo "s 3"
pypy3 run_search.py -p 3 -s 3 >> results/results_p3.txt
echo "s 4"
pypy3 run_search.py -p 3 -s 4 >> results/results_p3.txt
echo "s 5"
pypy3 run_search.py -p 3 -s 5 >> results/results_p3.txt
echo "s 6"
pypy3 run_search.py -p 3 -s 6 >> results/results_p3.txt
echo "s 7"
pypy3 run_search.py -p 3 -s 7 >> results/results_p3.txt
echo "s 8"
pypy3 run_search.py -p 3 -s 8 >> results/results_p3.txt
echo "s 9"
pypy3 run_search.py -p 3 -s 9 >> results/results_p3.txt
echo "s 10"
pypy3 run_search.py -p 3 -s 10 >> results/results_p3.txt
echo "s 11"
pypy3 run_search.py -p 3 -s 11 >> results/results_p3.txt

echo "Solving Problem 4 on a one-by-one basis"
echo "s 1"
pypy3 run_search.py -p 4 -s 1 >> results/results_p4.txt
echo "s 3"
pypy3 run_search.py -p 4 -s 3 >> results/results_p4.txt
echo "s 4"
pypy3 run_search.py -p 4 -s 4 >> results/results_p4.txt
echo "s 5"
pypy3 run_search.py -p 4 -s 5 >> results/results_p4.txt
echo "s 6"
pypy3 run_search.py -p 4 -s 6 >> results/results_p4.txt
echo "s 7"
pypy3 run_search.py -p 4 -s 7 >> results/results_p4.txt
echo "s 8"
pypy3 run_search.py -p 4 -s 8 >> results/results_p4.txt
echo "s 9"
pypy3 run_search.py -p 4 -s 9 >> results/results_p4.txt
echo "s 10"
pypy3 run_search.py -p 4 -s 10 >> results/results_p4.txt
echo "s 11"
pypy3 run_search.py -p 4 -s 11 >> results/results_p4.txt


