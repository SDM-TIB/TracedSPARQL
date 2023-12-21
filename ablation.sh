#!/bin/bash

source ./variables.sh
source ./run_testbeds.sh

declare -a configs_ablation=("ablation/baseline"
                             "ablation/opt-1"
                             "ablation/opt-2"
                             "ablation/opt-3"
                             "ablation/tracedsparql")

declare -a datasets=($DOCKER_BENCHMARK_LUBM_MKG)
query_dir="queries/ablation"
declare -a shape_dir=${SHAPES_LUBM[1]}
run_testbeds datasets[@] $query_dir shape_dir[@] configs_ablation[@] "ablation"

declare -a datasets=($DOCKER_BENCHMARK_WATDIV_MKG)
query_dir="queries/watdiv-ablation"
declare -a shape_dir=${SHAPES_WATDIV[0]}
run_testbeds datasets[@] $query_dir shape_dir[@] configs_ablation[@] "ablation-watdiv"

declare -a datasets=($DOCKER_BENCHMARK_DBPEDIA)
query_dir="queries/dbpedia-ablation"
declare -a shape_dir=${SHAPES_DBPEDIA[0]}
run_testbeds datasets[@] $query_dir shape_dir[@] configs_ablation[@] "ablation-dbpedia"

