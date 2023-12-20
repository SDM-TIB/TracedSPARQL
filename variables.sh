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

DOCKER_BENCHMARK_WATDIV_SKG="tracedsparql_watdiv_500k"
DOCKER_BENCHMARK_WATDIV_MKG="tracedsparql_watdiv_10M"
DOCKER_BENCHMARK_WATDIV_LKG="tracedsparql_watdiv_100M"

declare -a DATASETS_WATDIV=($DOCKER_BENCHMARK_WATDIV_SKG
                            $DOCKER_BENCHMARK_WATDIV_MKG
                            $DOCKER_BENCHMARK_WATDIV_LKG)

DOCKER_BENCHMARK_DBPEDIA="tracedsparql_dbpedia"
declare -a DATASETS_DBPEDIA=($DOCKER_BENCHMARK_DBPEDIA)

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
declare -a SHAPES_WATDIV=("/shapes/watdiv/network1"
                          "/shapes/watdiv/network2")
declare -a SHAPES_DBPEDIA=("/shapes/dbpedia")

# mapping from data sets to source description
declare -A RDFMTS=( [$DOCKER_BENCHMARK_LUBM_SKG]="rdfmts-lubm-skg.json"
                    [$DOCKER_BENCHMARK_LUBM_MKG]="rdfmts-lubm-mkg.json"
                    [$DOCKER_BENCHMARK_LUBM_LKG]="rdfmts-lubm-lkg.json"
                    [$DOCKER_BENCHMARK_WATDIV_SKG]="rdfmts-watdiv-500k.json"
                    [$DOCKER_BENCHMARK_WATDIV_MKG]="rdfmts-watdiv-10M.json"
                    [$DOCKER_BENCHMARK_WATDIV_LKG]="rdfmts-watdiv-100M.json"
                    [$DOCKER_BENCHMARK_DBPEDIA]="rdfmts-dbpedia.json" )

# mapping from data sets to ports
declare -A PORTS=( [$DOCKER_BENCHMARK_LUBM_SKG]="15000"
                   [$DOCKER_BENCHMARK_LUBM_MKG]="15001"
                   [$DOCKER_BENCHMARK_LUBM_LKG]="15002"
                   [$DOCKER_BENCHMARK_WATDIV_SKG]="15003"
                   [$DOCKER_BENCHMARK_WATDIV_MKG]="15004"
                   [$DOCKER_BENCHMARK_WATDIV_LKG]="15005"
                   [$DOCKER_BENCHMARK_DBPEDIA]="15006" )

