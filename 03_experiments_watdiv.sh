#!/bin/bash

source ./variables.sh
source ./run_testbeds.sh

declare -a datasets=${DATASETS_WATDIV[@]}
query_dir="queries/watdiv"
shape_dir=${SHAPES_WATDIV[@]}

run_testbeds datasets[@] $query_dir shape_dir[@] CONFIGS[@] "watdiv"

