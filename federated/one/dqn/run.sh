#!/bin/bash

echo "Starting server"
python server.py --num_clients=$1 --num_rounds=$2 --learning_rate=$3 --gamma=$4 --seed=$5&
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 1 $1`; do
    cpu_index=$((i - 1))
    echo "Starting client $i on CPU $cpu_index"
    taskset -c $cpu_index python client.py  --id=${i} --num_clients=$1 --learning_rate=$3 --gamma=$4 --seed=$5&
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# Wait for all background processes to complete
wait