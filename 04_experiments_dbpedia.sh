#!/bin/bash

source ./variables.sh
source ./run-testbeds.sh

declare -a datasets=${DATASETS_DBPEDIA[@]}
query_dir="queries/dbpedia"
shape_dir=${SHAPES_DBPEDIA[@]}

declare -a con=($CONFIG_BASELINE
                $CONFIG_VALSPARQL)

run_testbeds datasets[@] $query_dir shape_dir[@] con[@] "dbpedia"

