#!/bin/bash

batch_sizes=(8 16)
learning_rates=(1e-3 5e-3)
filters_options=(
  "128 64 32 16"
  "64 32"
)
kernel_sizes=("3 3" "5 5")
epochs=10

for batch_size in "${batch_sizes[@]}"; do
  for learning_rate in "${learning_rates[@]}"; do
    for filters in "${filters_options[@]}"; do
      for kernel_size in "${kernel_sizes[@]}"; do
        # Convert filters into separate arguments (e.g., --filters 128 --filters 64 --filters 32 --filters 16)
        FILTER_ARGS=""
        for filter in $filters; do
          FILTER_ARGS="$FILTER_ARGS --filters $filter"
        done
        echo "Running with batch_size=$batch_size, learning_rate=$learning_rate, filters=$filters, kernel_size=$kernel_size, epochs=$epochs"
        
        eval "python main.py --number_of_epoches $epochs --batch_size $batch_size --initial_learning_rate $learning_rate $FILTER_ARGS --kernel_size $kernel_size"
      done
    done
  done
done
