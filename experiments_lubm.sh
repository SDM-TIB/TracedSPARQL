#!/bin/bash

source ./variables.sh
source ./run_testbeds.sh

declare -a datasets=${DATASETS_LUBM[@]}
query_dir="queries/lubm"
shape_dir=${SHAPES_LUBM[@]}

run_testbeds datasets[@] $query_dir shape_dir[@] CONFIGS[@] "lubm"

