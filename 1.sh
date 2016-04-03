#!/bin/bash

printf "Part A \n-------------------------------------------------\n
We will consider 10 cases in which, for each case we will take some pixels per cell.The accuracy for each of the few cases are given below:\n"

for i in {3..10}
do
   printf "\nCase $i: Testing folder is $i:\n-------------------------------------------------\n"
   python 1a.py $i | awk '/---------/{y=1;next}y'
done