#!/bin/bash
# running all experiments automatically

./01_preparation.sh
./02_experiments_lubm.sh
./03_experiments_watdiv.sh
./04_experiments_dbpedia.sh
./05_ablation_study.sh
./06_cleanup.sh
