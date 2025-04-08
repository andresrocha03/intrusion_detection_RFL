#!/bin/bash

echo "Starting tests"

for i in `seq 1 10`; do
    echo "Starting test $i"
    ./run.sh 5 101 5 1 $i
done


