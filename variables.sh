#!/bin/bash
# setting variables to be used in other scripts

# number of runs
RUNS=5

# Docker container names
DOCKER_BENCHMARK_LUBM_SKG="tracedsparql_lubm_skg"
DOCKER_BENCHMARK_LUBM_MKG="tracedsparql_lubm_mkg"
DOCKER_BENCHMARK_LUBM_LKG="tracedsparql_lubm_lkg"

declare -a DATASETS_LUBM=($DOCKER_BENCHMARK_LUBM_SKG
                          $DOCKER_BENCHMARK_LUBM_MKG
                          $DOCKER_BENCHMARK_LUBM_LKG)

DOCKER_ENGINE="tracedsparql_engine"

# API configurations
CONFIG_BASELINE="baseline"
CONFIG_BASELINE_S2S="s2s_baseline"
CONFIG_TRACEDSPARQL="tracedsparql"
CONFIG_TRACEDSPARQL_S2S="s2s_tracedsparql"

declare -a CONFIGS=($CONFIG_BASELINE
                    $CONFIG_BASELINE_S2S
                    $CONFIG_TRACEDSPARQL
                    $CONFIG_TRACEDSPARQL_S2S)

# SHACL networks
declare -a SHAPES_LUBM=("/shapes/lubm/network1"
                        "/shapes/lubm/network2")

# mapping from data sets to source description
declare -A RDFMTS=( [$DOCKER_BENCHMARK_LUBM_SKG]="rdfmts-lubm-skg.json"
                    [$DOCKER_BENCHMARK_LUBM_MKG]="rdfmts-lubm-mkg.json"
                    [$DOCKER_BENCHMARK_LUBM_LKG]="rdfmts-lubm-lkg.json" )

# mapping from data sets to ports
declare -A PORTS=( [$DOCKER_BENCHMARK_LUBM_SKG]="15000"
                   [$DOCKER_BENCHMARK_LUBM_MKG]="15001"
                   [$DOCKER_BENCHMARK_LUBM_LKG]="15002" )

