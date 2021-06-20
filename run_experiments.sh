#!/bin/sh

utd="1 2 4"

for value in $utd; do
  nohup python -u train.py --updates_per_step $value --isDropout &> output_$value.log &
done