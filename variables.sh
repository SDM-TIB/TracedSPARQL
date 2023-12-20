#!/bin/bash
# setting variables to be used in other scripts

# number of runs
RUNS=5

# Docker container names
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

